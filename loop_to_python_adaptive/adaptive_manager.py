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
    run_autotune_isf_iterations,
    run_autotune_cr_iterations,
    run_autotune_basal_iterations,
    AutotuneConfig,
)
from .autosens import (
    AutosensConfig, AutosensBuffer, AutosensPoint,
    compute_autosens, apply_autosens_to_isf, apply_autosens_to_basal, apply_autosens_to_target
)

logger = logging.getLogger(__name__)


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
    log_rows: list = None
    json_history: list = None
    isf_history: list = None
    cr_history: list = None
    basal_history: list = None
    autosens_log: list = None
    autosens_buffer: object = None
    
    # Categorized data (refreshed each step)
    categorized_data: dict = None
    
    def __post_init__(self):
        if self.log_rows is None:
            self.log_rows = []
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
        if self.categorized_data is None:
            self.categorized_data = {
                'CSFGlucoseData': [],
                'ISFGlucoseData': [],
                'basalGlucoseData': [],
                'UAMGlucoseData': [],
                'CRData': [],
            }

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
    ):
        self.target = target   
        self.warmup_duration = timedelta(days=warmup_days)
        self.adaptation_interval = timedelta(hours=adaptation_interval_hours)
        self.n_autotune_iterations = n_autotune_iterations
        self.max_window_days = max_window_days
        self.autotune_cfg = autotune_cfg or AutotuneConfig()
        self.autosens_cfg = autosens_cfg or AutosensConfig()
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
        json_autotune: dict,
        df_observations: pd.DataFrame,
        env_sample_time: int,
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
        
        # Log raw observation
        self._log_row(state, datetime, glucose)
        
        # Accumulate json for autotune
        state.json_history.append(json_autotune)
        
        # ────────────────────────────────────────────────────────────────
        # Categorize glucose data (once, before both layers)
        # ────────────────────────────────────────────────────────────────
        categorized = self._prepare_categorized_data(name, state)
        
        # Store categorized data in state for autotune to use
        state.categorized_data = categorized
        
        # ────────────────────────────────────────────────────────────────
        # Compute autosens (every 5 min, using categorized data)
        # ────────────────────────────────────────────────────────────────
        bgi = self._get_latest_bgi(json_autotune)
        deviation = self._get_latest_deviation(df_observations, bgi)
        cob = float(get_active_carbs(json_autotune))
        
        state.autosens_buffer.push(AutosensPoint(
            deviation=deviation,
            bgi=bgi,
            cob=cob,
            glucose=float(glucose),
        ))
        
        autosens_result = compute_autosens(state.autosens_buffer, self.autosens_cfg)
        state.autosens_ratio = autosens_result['ratio']
        
        state.autosens_log.append({
            'datetime': datetime,
            'ratio': autosens_result['ratio'],
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
    
    def _get_latest_bgi(self, json_data: dict) -> float:
        """Extract BGI from Loop prediction."""
        try:
            values, _ = get_prediction_values_and_dates(json_data)
            if values and len(values) >= 2:
                return float(values[1]) - float(values[0])
        except Exception:
            pass
        return 0.0
    
    def _get_latest_deviation(self, df_observations: pd.DataFrame, bgi: float) -> float:
        """Compute deviation = avgDelta - BGI."""
        try:
            cgm = pd.to_numeric(df_observations['CGM'], errors='coerce').dropna()
            if len(cgm) < 5:
                return 0.0
            recent = cgm.iloc[-5:]
            avg_delta = recent.diff().dropna().mean()
            return float(avg_delta - bgi)
        except Exception:
            return 0.0
    
    def _log_row(self, state: AdaptiveState, datetime: object, cgm: float):
        """Log raw observation."""
        state.log_rows.append({
            'date': datetime,
            'CGM': cgm,
        })
    
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
        """Trigger autotune if conditions are met."""
        
        state = self.patients[name]
        
        elapsed = datetime - state.sim_start
        if elapsed < self.warmup_duration:
            return
        
        if datetime - state.last_adapted < self.adaptation_interval:
            return
        
        if not state.json_history or len(state.log_rows) < 288:  # 1 day minimum
            return
        
        # Use cached categorized data from manage_step()
        categorized = state.categorized_data
        basalGlucoseData = categorized['basalGlucoseData']
        ISFGlucoseData = categorized['ISFGlucoseData']
        CSFGlucoseData = categorized['CSFGlucoseData']
        CRData = categorized['CRData']
        
        # Check if we have enough data in each category
        if len(basalGlucoseData) < 50:
            logger.warning(f"[{name}] Not enough basal data ({len(basalGlucoseData)} points)")
            return
        if len(ISFGlucoseData) < 20:
            logger.warning(f"[{name}] Not enough ISF data ({len(ISFGlucoseData)} points)")
            return
        
        print(
            f"\n[Autotune] {name} @ {datetime} — "
            f"Basal: {len(basalGlucoseData)}, ISF: {len(ISFGlucoseData)}, "
            f"CSF: {len(CSFGlucoseData)}, CR: {len(CRData)}"
        )
        
        try:
            old_isf = state.isf
            old_cr = state.cr
            old_basal = state.basal_pr_hr
            new_isf = old_isf
            new_cr = old_cr
            new_basal = old_basal
            
            # ──────────────────────────────────────────────────────────────
            # Autotune ISF (uses ISFGlucoseData)
            # ──────────────────────────────────────────────────────────────
            try:
                if len(ISFGlucoseData) >= 20:
                    isf_df = pd.DataFrame(ISFGlucoseData)
                    result_isf = run_autotune_isf_iterations(
                        df_windows=[isf_df],
                        loop_algorithm_inputs=[self._merge_json_inputs(state.json_history)],
                        pump_isf=state.isf_pump,
                        isf_current=state.isf,
                        n_iterations=self.n_autotune_iterations,
                        cfg=self.autotune_cfg,
                        json_history_list=[state.json_history],
                    )
                    new_isf = result_isf['finalISF']
                    state.isf = new_isf
                    state.isf_history.append({
                        'datetime': datetime,
                        'old': old_isf,
                        'new': new_isf,
                        'data_points': len(ISFGlucoseData),
                    })
            except Exception as e:
                logger.warning(f"[{name}] ISF autotune failed: {e}")
            
            # ──────────────────────────────────────────────────────────────
            # Autotune CR (uses CRData)
            # ──────────────────────────────────────────────────────────────
            try:
                if len(CRData) >= 2:
                    cr_df = pd.DataFrame(CRData)
                    result_cr = run_autotune_cr_iterations(
                        df_windows=[cr_df],
                        loop_algorithm_inputs=[self._merge_json_inputs(state.json_history)],
                        pump_cr=state.cr_pump,
                        cr_current=state.cr,
                        n_iterations=self.n_autotune_iterations,
                        cfg=self.autotune_cfg,
                        json_history_list=[state.json_history],
                    )
                    new_cr = result_cr['finalCR']
                    state.cr = new_cr
                    state.cr_history.append({
                        'datetime': datetime,
                        'old': old_cr,
                        'new': new_cr,
                        'data_points': len(CRData),
                    })
            except Exception as e:
                logger.warning(f"[{name}] CR autotune failed: {e}")
            
            # ──────────────────────────────────────────────────────────────
            # Autotune Basal (uses basalGlucoseData)
            # ──────────────────────────────────────────────────────────────
            try:
                if len(basalGlucoseData) >= 50:
                    basal_df = pd.DataFrame(basalGlucoseData)
                    result_basal = run_autotune_basal_iterations(
                        df_windows=[basal_df],
                        loop_algorithm_inputs=[self._merge_json_inputs(state.json_history)],
                        pump_basal=state.basal_pr_hr_pump,
                        basal_current=state.basal_pr_hr,
                        n_iterations=self.n_autotune_iterations,
                        cfg=self.autotune_cfg,
                        json_history_list=[state.json_history],
                    )
                    new_basal = result_basal['finalBasal']
                    state.basal_pr_hr = new_basal
                    state.basal_history.append({
                        'datetime': datetime,
                        'old': old_basal,
                        'new': new_basal,
                        'data_points': len(basalGlucoseData),
                    })
            except Exception as e:
                logger.warning(f"[{name}] Basal autotune failed: {e}")
            
            print(
                f"[Autotune] {name}: "
                f"ISF {old_isf:.2f} → {new_isf:.2f}, "
                f"CR {old_cr:.1f} → {new_cr:.1f}, "
                f"Basal {old_basal:.2f} → {new_basal:.2f}"
            )
            
            state.last_adapted = datetime
            
            # Roll window
            one_day_rows = int(24 * 60 / 5)
            while len(state.log_rows) > self.max_window_days * one_day_rows:
                state.log_rows = state.log_rows[one_day_rows:]
                state.json_history = state.json_history[one_day_rows:]
        
        except Exception as e:
            logger.warning(f"[{name}] Autotune failed: {e}")
            state.last_adapted = datetime
            state.log_rows = []
            state.json_history = []

    def _prepare_categorized_data(
        self,
        name: str,
        state: AdaptiveState,
    ) -> dict[str, list[dict]]:
        """
        Prepare and categorize glucose data into:
        - CSFGlucoseData (meal/carb absorption)
        - ISFGlucoseData (insulin sensitivity periods)
        - basalGlucoseData (basal periods)
        - UAMGlucoseData (unannounced meals)
        - CRData (carb ratio periods)
        """
        df = pd.DataFrame(state.log_rows).set_index('date')
        df.index.name = 'date'
        
        result = prepare_for_autotune_isf(
            df,
            loop_algorithm_input=self._merge_json_inputs(state.json_history),
            cfg=self.autotune_prep_cfg,
            json_history=state.json_history,
        )
        
        return {
            'CSFGlucoseData': result['CSFGlucoseData'],
            'ISFGlucoseData': result['ISFGlucoseData'],
            'basalGlucoseData': result['basalGlucoseData'],
            'UAMGlucoseData': result['UAMGlucoseData'],
            'CRData': result['CRData'],
        }
    def _build_daily_log_df(self, log_rows: list) -> pd.DataFrame:
        """Build DataFrame from raw log rows."""
        df = pd.DataFrame(log_rows).set_index('date')
        df.index.name = 'date'
        return df
    
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
        return self.patients.get(patient_name, AdaptiveState(
            isf=0, isf_pump=0, cr=0, cr_pump=0, basal_pr_hr=0, basal_pr_hr_pump=0
        )).isf_history
    
    def get_current_isf(self, patient_name: str) -> float:
        return self.patients.get(patient_name, AdaptiveState(
            isf=0, isf_pump=0, cr=0, cr_pump=0, basal_pr_hr=0, basal_pr_hr_pump=0
        )).isf
    
    def get_cr_history(self, patient_name: str) -> list:
        return self.patients.get(patient_name, AdaptiveState(
            isf=0, isf_pump=0, cr=0, cr_pump=0, basal_pr_hr=0, basal_pr_hr_pump=0
        )).cr_history
    
    def get_current_cr(self, patient_name: str) -> float:
        return self.patients.get(patient_name, AdaptiveState(
            isf=0, isf_pump=0, cr=0, cr_pump=0, basal_pr_hr=0, basal_pr_hr_pump=0
        )).cr
    
    def get_basal_history(self, patient_name: str) -> list:
        return self.patients.get(patient_name, AdaptiveState(
            isf=0, isf_pump=0, cr=0, cr_pump=0, basal_pr_hr=0, basal_pr_hr_pump=0
        )).basal_history
    
    def get_current_basal(self, patient_name: str) -> float:
        return self.patients.get(patient_name, AdaptiveState(
            isf=0, isf_pump=0, cr=0, cr_pump=0, basal_pr_hr=0, basal_pr_hr_pump=0
        )).basal_pr_hr
    
    def get_autosens_log(self, patient_name: str) -> list:
        return self.patients.get(patient_name, AdaptiveState(
            isf=0, isf_pump=0, cr=0, cr_pump=0, basal_pr_hr=0, basal_pr_hr_pump=0
        )).autosens_log
    
    def reset(self):
        """Clear all state."""
        self.patients = {}