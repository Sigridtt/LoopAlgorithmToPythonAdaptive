"""
AdaptiveManager: Encapsulates all two-layer adaptation logic.

This separates adaptive concerns from the simglucose controller.
The controller just calls manage_step() each iteration.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import timedelta
import logging

from .autotune import (
    tune_parameter,
    AutotuneConfig,
)
from .autosens import (
    AutosensConfig, AutosensBuffer, AutosensPoint,
    compute_autosens, apply_autosens_to_isf, apply_autosens_to_basal, apply_autosens_to_target, 
)

from .autotune_prep import prepare_for_autotune_isf, AutotunePrepConfig
logger = logging.getLogger(__name__)

from .loop_oref_mapping import prepare_isf_glucose_data

@dataclass
class AdaptiveState:
    """State for one patient's adaptation."""
    isf: float
    isf_pump: float
    cr: float
    cr_pump: float
    basal_pr_hr: float
    basal_pr_hr_pump: float
    autosens_ratio: float = 1.0
    
    # Timing
    sim_start: object = None
    last_adapted: object = None
    
    # History
    df_history: pd.DataFrame = None
    json_history: list = None
    isf_history: list = None
    cr_history: list = None
    basal_history: list = None
    autosens_log: list = None
    autosens_buffer: object = None
    

    
    def __post_init__(self):
        if self.df_history is None:
            self.df_history = pd.DataFrame(columns=["CGM"])
        if self.json_history is None:
            self.json_history = []
        if self.isf_history is None:
            self.isf_history = []
        if self.cr_history is None:
            self.cr_history = []
        if self.basal_history is None:
            self.basal_history = []
        if self.autosens_log is None:
            self.autosens_log = []
 

class AdaptiveManager:
    """
    Manages two-layer adaptation (autotune + autosens) for multiple patients.
    
    The controller calls:
        1. manage_step() for each 5-min timestep
        2. get_*_history() to retrieve results
    """
    
    def __init__(
        self,
        target: float = 100, #oref default
        warmup_days: float = 1,
        adaptation_interval_hours: float = 24,
        n_autotune_iterations: int = 1,
        max_window_days: int = 3,
        autotune_cfg: Optional[AutotuneConfig] = None,
        autosens_cfg: Optional[AutosensConfig] = None,
        autotune_prep_cfg: Optional[AutotunePrepConfig] = None,
    ):
        self.target = target   
        self.warmup_duration = timedelta(days=warmup_days)
        self.adaptation_interval = timedelta(hours=adaptation_interval_hours)
        self.max_window_days = max_window_days
        self.autotune_cfg = autotune_cfg or AutotuneConfig()
        self.autosens_cfg = autosens_cfg or AutosensConfig()
        self.autotune_prep_cfg = autotune_prep_cfg or AutotunePrepConfig()
        self.patients: Dict[str, AdaptiveState] = {}
    
    def initialize_patient(
        self,
        name: str,
        datetime: object,
        isf_pump: float,
        cr_pump: float,
        basal_pr_hr_pump: float,
    ):
        """Initialize adaptive state for a new patient."""
        self.patients[name] = AdaptiveState(
            isf=isf_pump,
            isf_pump=isf_pump,
            cr=cr_pump,
            cr_pump=cr_pump,
            basal_pr_hr=basal_pr_hr_pump,
            basal_pr_hr_pump=basal_pr_hr_pump,
            sim_start=datetime,
            last_adapted=datetime,
            autosens_buffer=AutosensBuffer(self.autosens_cfg),
        )
    
    def manage_step(
        self,
        name: str,
        datetime: object,
        glucose: float,
        json_autotune: dict
    ) -> Tuple[float, float, float, float, float]:
        """
        Execute one step of adaptation:
        1. Log raw observation
        2. Categorize glucose data (autotune-prep)
        3. Apply autosens scaling (layer 2)
        4. Check if autotune should trigger (layer 1)
        
        Returns: (effective_isf, effective_basal, effective_cr, effective_min_bg, effective_max_bg)
        """
        
        if name not in self.patients:
            raise ValueError(f"Patient {name} not initialized. Call initialize_patient() first.")
        
        state = self.patients[name]
        
        # Log raw observation (idempotent for duplicate latest timestamps)
        is_new_row = self._append_row(state, datetime, glucose)

        # Keep json history aligned with df_history length
        if is_new_row:
            state.json_history.append(json_autotune)
        elif state.json_history:
            state.json_history[-1] = json_autotune
        else:
            state.json_history.append(json_autotune)

        # Enrich with a small context window for stable avgDelta/BGI/deviation,
        # then write only derived columns back onto the newest row.
        CONTEXT_ROWS = 5
        bgi = 0.0
        deviation = 0.0
        cob = 0.0
        can_update_autosens = False

        latest = state.df_history.iloc[-1]
        cgm_value = latest.get("CGM") if "CGM" in latest else np.nan

        if len(state.df_history) >= CONTEXT_ROWS and not pd.isna(cgm_value):
            context_df = state.df_history.iloc[-CONTEXT_ROWS:]
            context_json_history = state.json_history[-CONTEXT_ROWS:]
            df_ctx_enriched, _ = prepare_isf_glucose_data(
                context_df,
                loop_algorithm_input=json_autotune,
                basal=state.basal_pr_hr,
                isf=state.isf,
                cr=state.cr,
                json_history=context_json_history,
            )

            # Update only derived columns in the last row; keep raw CGM untouched.
            last_idx = state.df_history.index[-1]
            for col in df_ctx_enriched.columns:
                if col != "CGM":
                    state.df_history.at[last_idx, col] = df_ctx_enriched.iloc[-1][col]

            latest = state.df_history.iloc[-1]
            bgi = float(latest["BGI"]) if "BGI" in latest and not pd.isna(latest["BGI"]) else 0.0
            deviation = float(latest["deviation"]) if "deviation" in latest and not pd.isna(latest["deviation"]) else 0.0
            cob = float(latest["COB"]) if "COB" in latest and not pd.isna(latest["COB"]) else 0.0
            can_update_autosens = True
        

        
        # ────────────────────────────────────────────────────────────────
        # Compute autosens (every 5 min, using categorized data)
        # ────────────────────────────────────────────────────────────────

        if can_update_autosens:
            state.autosens_buffer.push(AutosensPoint(
                deviation=deviation,
                bgi=bgi,
                cob=cob,
                glucose=float(glucose),
            ))

            autosens_result = compute_autosens(state.autosens_buffer, self.autosens_cfg)
            state.autosens_ratio = autosens_result['ratio']
        else:
            autosens_result = {
                'ratio': state.autosens_ratio,
                'sign_mode': 'insufficient_context',
                'n_points_8h': len(state.autosens_buffer.points_8h()),
                'n_points_24h': len(state.autosens_buffer.points_24h()),
            }
        
        state.autosens_log.append({
            'datetime': datetime,
            'ratio': autosens_result['ratio'],
            'sign_mode': autosens_result.get('sign_mode', 'unknown'),
            'n_points_8h': autosens_result['n_points_8h'],
            'n_points_24h': autosens_result['n_points_24h'],
            'bgi': bgi,
            'deviation': deviation,
            'cob': cob,
        })
        
        # ────────────────────────────────────────────────────────────────
        # Apply autosens scaling (layer 2)
        # ────────────────────────────────────────────────────────────────
        effective_isf = apply_autosens_to_isf(state.isf, state.autosens_ratio)
        effective_basal = apply_autosens_to_basal(state.basal_pr_hr, state.autosens_ratio)
        effective_cr = state.cr  # CR doesn't scale with autosens in oref0's current implementation
        effective_min_bg, effective_max_bg = self._apply_autosens_to_target(
            autosens_ratio=state.autosens_ratio,
        )
        
        # ────────────────────────────────────────────────────────────────
        # Check if autotune should trigger (layer 1)
        # ────────────────────────────────────────────────────────────────
        self._maybe_adapt(name, datetime)
        
        return effective_isf, effective_basal, effective_cr, effective_min_bg, effective_max_bg
    # ------------------------------------------------------------------
    #   Autosens helpers
    # ------------------------------------------------------------------
    
   
    
   
    def _append_row(self, state: AdaptiveState, datetime, glucose) -> bool:
        """Append new CGM row or update the latest row if timestamp repeats.

        Returns
        -------
        bool
            True if a new row was appended, False if the latest row was updated.
        """
        ts = pd.to_datetime(datetime, utc=True)
        new_row = pd.DataFrame(
            [{"CGM": glucose}],
            index=[ts]
        )

        if state.df_history.empty:
            state.df_history = new_row
            return True
        if state.df_history.index[-1] == ts:
            state.df_history.at[ts, "CGM"] = glucose
            return False
        else:
            state.df_history = pd.concat([state.df_history, new_row])
            return True
    
    def _apply_autosens_to_target(
        self,
        autosens_ratio: float,
        temp_target_active: bool = False,  # ← NEW
    ) -> Tuple[float, float]:
        """
        Adjust BG target based on autosens ratio (oref0 behavior).
        
        Delegates to the standalone apply_autosens_to_target() function
        from autosens.py to keep all autosens logic in one place.
        If temp target is active, don't adjust with autosens.
        This prevents double-adjustment (once for temp target, once for autosens).
        
        Parameters
        ----------
        autosens_ratio : float
            Current autosens ratio (0.7 to 1.2)
        
        Returns
        -------
        Tuple[float, float]
            (effective_min_bg, effective_max_bg) adjusted targets
        """
        if temp_target_active:
            # Return unadjusted bounds
            return self.target - 10, self.target + 10
    
        # Define symmetric target bounds around self.target
        target_min = self.target - 10
        target_max = self.target + 10
        
        # Delegate to standalone function from autosens.py
        return apply_autosens_to_target(
            target_min=target_min,
            target_max=target_max,
            autosens_ratio=autosens_ratio,
            sensitivity_raises_target=True,
            resistance_lowers_target=False,
        )
    # ------------------------------------------------------------------
    #   Autotune trigger
    # ------------------------------------------------------------------
    
    def _maybe_adapt(self, name: str, datetime: object):
        """
        Trigger autotune if warmup has elapsed and enough data is available.
        
        Layer 1 (slow): updates state.isf, state.cr, state.basal_pr_hr permanently.
        These become the new baseline that autosens (layer 2) scales on top of.
        
        All three parameters are tuned from the same categorized data batch,
        computed once per adaptation cycle using current tuned values for BGI/deviation.
        """
        
        state = self.patients[name]
        
        elapsed = datetime - state.sim_start
        if elapsed < self.warmup_duration:
            return
        
        if datetime - state.last_adapted < self.adaptation_interval:
            return
        
        if not state.json_history or len(state.df_history) < 288:  # 1 day minimum
            return
        
        # ────────────────────────────────────────────────────────────────
        # Categorize glucose data (once, before both layers)
        # ────────────────────────────────────────────────────────────────
        # Rebuild prep config from current state so BGI/deviation are computed
        # with the values autotune has learned, not the original pump values.
        prep_cfg = AutotunePrepConfig(
            basal_rate=state.basal_pr_hr,
            isf=state.isf,
            carb_ratio=state.cr,
            min_5m_carbimpact=self.autotune_prep_cfg.min_5m_carbimpact,
            categorize_uam_as_basal=self.autotune_prep_cfg.categorize_uam_as_basal,
        )

        categorized = prepare_for_autotune_isf(
            state.df_history,
            loop_algorithm_input=self._merge_json_inputs(state.json_history),
            cfg=prep_cfg,
            json_history=state.json_history,
        )


        basalGlucoseData = categorized['basalGlucoseData']
        ISFGlucoseData = categorized['ISFGlucoseData']
        CSFGlucoseData = categorized['CSFGlucoseData']

        logger.info(
            f"[Autotune] {name} @ {datetime} | "
            f"basal={len(basalGlucoseData)} ISF={len(ISFGlucoseData)} "
            f"CSF={len(CSFGlucoseData)} pts"
        )
        
        # ── Safety gates: require minimum points per category ─────────────
        # These are the same thresholds oref0 uses before allowing a parameter
        # to move. If a category is thin it is better to leave that parameter
        # unchanged than to update it from noisy data.
        ISF_MIN_PTS   = self.autotune_cfg.min_points   # default 10
        BASAL_MIN_PTS = 50    # basal needs a long quiet window to be meaningful
        CR_MIN_PTS    = self.autotune_cfg.min_points    # CSF 5-min points, not episodes

        old_isf   = state.isf
        old_cr    = state.cr
        old_basal = state.basal_pr_hr

        # ──────────────────────────────────────────────────────────────
        # Autotune ISF (uses ISFGlucoseData)
        # ──────────────────────────────────────────────────────────────
        # Uses ISFGlucoseData: quiet periods where insulin is the only driver.
        # Each point has deviation = avgDelta - BGI computed from the IOB model.
        if len(ISFGlucoseData) >= ISF_MIN_PTS:
            try:
                result = tune_parameter(
                    param_name="ISF",
                    current_value=state.isf,
                    glucose_data=ISFGlucoseData,   # already has BGI + deviation
                    pump_value=state.isf_pump,
                    cfg=self.autotune_cfg,
                )
                state.isf = result["newValue"]
                state.isf_history.append({
                    "datetime":    datetime,
                    "old":         old_isf,
                    "new":         state.isf,
                    "p50_ratio":   result["p50_ratio"],
                    "data_points": len(ISFGlucoseData),
                    "reason":      result["reason"],
                })
                logger.info(f"[Autotune] {name} ISF {old_isf:.3f} → {state.isf:.3f} "
                            f"(p50={result['p50_ratio']}, n={len(ISFGlucoseData)})")
            except Exception as e:
                logger.warning(f"[Autotune] {name} ISF failed: {e}")
        else:
            logger.info(f"[Autotune] {name} ISF skipped: "
                        f"only {len(ISFGlucoseData)} pts (need {ISF_MIN_PTS})")

        # ──────────────────────────────────────────────────────────────
        # Autotune basal (uses basalGlucoseData)
        # ──────────────────────────────────────────────────────────────
        # Uses basalGlucoseData: periods with no active carbs and low IOB.
        # Needs more points than ISF because basal signal is weaker.
        if len(basalGlucoseData) >= BASAL_MIN_PTS:
            try:
                result = tune_parameter(
                    param_name="Basal",
                    current_value=state.basal_pr_hr,
                    glucose_data=basalGlucoseData,
                    pump_value=state.basal_pr_hr_pump,
                    cfg=self.autotune_cfg,
                )
                state.basal_pr_hr = result["newValue"]
                state.basal_history.append({
                    "datetime":    datetime,
                    "old":         old_basal,
                    "new":         state.basal_pr_hr,
                    "p50_ratio":   result["p50_ratio"],
                    "data_points": len(basalGlucoseData),
                    "reason":      result["reason"],
                })
                logger.info(f"[Autotune] {name} Basal {old_basal:.4f} → {state.basal_pr_hr:.4f} "
                            f"(p50={result['p50_ratio']}, n={len(basalGlucoseData)})")
            except Exception as e:
                logger.warning(f"[Autotune] {name} Basal failed: {e}")
        else:
            logger.info(f"[Autotune] {name} Basal skipped: "
                        f"only {len(basalGlucoseData)} pts (need {BASAL_MIN_PTS})")

            
        # ──────────────────────────────────────────────────────────────
        # Autotune CE (uses CSFGlucoseData)
        # ──────────────────────────────────────────────────────────────
        # Uses CSFGlucoseData: 5-min points during carb absorption windows.
        # Tuned with a lower adjustment_fraction (0.5 vs 1.0) because meal
        # absorption variance contaminates the deviation signal — a fast-
        # absorbing meal is indistinguishable from an incorrect CR.
        # This makes CR converge more slowly but avoids overcorrecting from
        # a single atypical meal.
        if len(CSFGlucoseData) >= CR_MIN_PTS:
            try:
                cr_cfg = AutotuneConfig(
                    min_points=self.autotune_cfg.min_points,
                    adjustment_fraction=self.autotune_cfg.cr_adjustment_fraction,  # 0.5
                    autosens_max=self.autotune_cfg.autosens_max,
                    autosens_min=self.autotune_cfg.autosens_min,
                    min_bgi_abs=self.autotune_cfg.min_bgi_abs,
                )
                result = tune_parameter(
                    param_name="CR",
                    current_value=state.cr,
                    glucose_data=CSFGlucoseData,
                    pump_value=state.cr_pump,
                    cfg=cr_cfg,
                )
                state.cr = result["newValue"]
                state.cr_history.append({
                    "datetime":    datetime,
                    "old":         old_cr,
                    "new":         state.cr,
                    "p50_ratio":   result["p50_ratio"],
                    "data_points": len(CSFGlucoseData),
                    "reason":      result["reason"],
                })
                logger.info(f"[Autotune] {name} CR {old_cr:.3f} → {state.cr:.3f} "
                            f"(p50={result['p50_ratio']}, n={len(CSFGlucoseData)})")
            except Exception as e:
                logger.warning(f"[Autotune] {name} CR failed: {e}")
        else:
            logger.info(f"[Autotune] {name} CR skipped: "
                        f"only {len(CSFGlucoseData)} pts (need {CR_MIN_PTS})")

        # ── Housekeeping ──────────────────────────────────────────────────
        state.last_adapted = datetime

        # Trim history to max_window_days so memory doesn't grow unbounded.
        # Drop from the front (oldest) in whole-day increments.
        max_rows = self.max_window_days * int(24 * 60 / 5)
        if len(state.df_history) > max_rows:
            state.df_history  = state.df_history.iloc[-max_rows:]
            state.json_history = state.json_history[-max_rows:]

   
    
    def _merge_json_inputs(self, json_history: list) -> dict:
        """Merge JSON snapshots into one coherent input."""
        if not json_history:
            return None
        
        merged = dict(json_history[-1])
        merged['doses'] = list(merged.get('doses', []) or [])
        
        # Merge doses
        seen_starts = {d.get('startDate') for d in merged['doses']}
        for json_input in reversed(json_history[:-1]):
            for dose in (json_input.get('doses') or []):
                start = dose.get('startDate')
                if start and start not in seen_starts:
                    merged['doses'].append(dose)
                    seen_starts.add(start)
        merged['doses'].sort(key=lambda d: d.get('startDate', ''))
        
        # Merge carbEntries
        seen_carb_dates = set()
        all_carb_entries = []
        for json_input in json_history:
            for entry in (json_input.get('carbEntries') or []):
                if float(entry.get('grams', 0)) > 0:
                    date = entry.get('date')
                    if date and date not in seen_carb_dates:
                        all_carb_entries.append(entry)
                        seen_carb_dates.add(date)
        all_carb_entries.sort(key=lambda e: e.get('date', ''))
        merged['carbEntries'] = all_carb_entries
        
        return merged
    
    # ------------------------------------------------------------------
    #   Public API
    # ------------------------------------------------------------------
    
    def get_isf_history(self, patient_name: str) -> list:
        if patient_name not in self.patients:
            raise KeyError(f"Patient '{patient_name}' not found")
        return self.patients[patient_name].isf_history
    
    def get_current_isf(self, patient_name: str) -> float:
        if patient_name not in self.patients:
            raise KeyError(f"Patient '{patient_name}' not found")
        return self.patients[patient_name].isf
    
    def get_cr_history(self, patient_name: str) -> list:
        if patient_name not in self.patients:
            raise KeyError(f"Patient '{patient_name}' not found")
        return self.patients[patient_name].cr_history
    
    def get_current_cr(self, patient_name: str) -> float:
        if patient_name not in self.patients:
            raise KeyError(f"Patient '{patient_name}' not found")
        return self.patients[patient_name].cr
    
    def get_basal_history(self, patient_name: str) -> list:
        if patient_name not in self.patients:
            raise KeyError(f"Patient '{patient_name}' not found")
        return self.patients[patient_name].basal_history
    
    def get_current_basal(self, patient_name: str) -> float:
        if patient_name not in self.patients:
            raise KeyError(f"Patient '{patient_name}' not found")
        return self.patients[patient_name].basal_pr_hr
    
    def get_autosens_log(self, patient_name: str) -> list:
        if patient_name not in self.patients:
            raise KeyError(f"Patient '{patient_name}' not found")
        return self.patients[patient_name].autosens_log
    
    def reset(self):
        """Clear all state."""
        self.patients = {}