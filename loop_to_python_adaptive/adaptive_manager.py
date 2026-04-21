"""
AdaptiveManager: Encapsulates all two-layer adaptation logic.
1: buckets are prepared using autotune_prep module
2: Autosens is computed every 5 minutes and applied on top of current ISF/basal/target
3: Autotune triggers every 24 hours (configurable), updating ISF and basal
4: Sick detection via sustained hyperglycemia (closed-loop compatible signal)

Feature flags allow running in 5 ablation conditions:
  A: enable_autotune=False, enable_autosens=False  → pure Loop baseline
  B: enable_autotune=True,  enable_autosens=False  → autotune only
  C: enable_autotune=False, enable_autosens=True   → autosens only
  D: enable_autotune=True,  enable_autosens=True   → full adaptive (no sick detection)
  E: enable_autotune=True,  enable_autosens=True   → full adaptive (with sick detection)
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import timedelta
import logging

from .autotune import (
    AutotuneConfig,
    run_autotune,
    tune_parameter,
)
from .autosens import (
    AutosensConfig, AutosensBuffer, AutosensPoint,
    compute_autosens, apply_autosens_to_isf, apply_autosens_to_basal,
    apply_autosens_to_target, _is_unexpected_rise
)
from .autotune_prep import categorized_buckets, AutotunePrepConfig, IncrementalAutotunePrep

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveState:
    """State for one patient's adaptation."""
    isf: float
    isf_pump: float
    cr: float
    cr_pump: float
    basal: float
    basal_pump: float
    autosens_ratio: float = 1.0
    sick: bool = False
    sick_start: Optional[pd.Timestamp] = None
    last_sick_check: Optional[pd.Timestamp] = None
    _resistance_streak: int = 0

    sim_start: object = None
    last_autotune: object = None

    df_history: pd.DataFrame = None
    json_history: list = None
    isf_history: list = None
    cr_history: list = None
    basal_history: list = None
    autosens_log: list = None
    autosens_buffer: object = None
    incremental_engine: object = None
    prepared_buckets: list = None

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
        if self.prepared_buckets is None:
            self.prepared_buckets = {}


class AdaptiveManager:
    """
    Manages two-layer adaptation (autotune + autosens) for multiple patients.

    Feature flags:
        enable_autotune       — run autotune daily parameter updates
        enable_autosens       — run autosens ratio scaling every 5 min
        enable_sick_detection — block autotune during detected illness
    """

    def __init__(
        self,
        target: float = 100,
        warmup_days: float = 1,
        autotune_interval_hours: float = 24,
        max_window_days: int = 3,
        autotune_cfg: Optional[AutotuneConfig] = None,
        autosens_cfg: Optional[AutosensConfig] = None,
        prep_cfg: Optional[AutotunePrepConfig] = None,
        # ── ablation flags ──────────────────────────────────────────────────
        enable_autotune: bool = True,
        enable_autosens: bool = True,
        enable_sick_detection: bool = True,
    ):
        self.target = target
        self.warmup_duration = timedelta(days=warmup_days)
        self.autotune_interval = timedelta(hours=autotune_interval_hours)
        self.max_window_days = max_window_days
        self.autotune_cfg = autotune_cfg or AutotuneConfig()
        self.autosens_cfg = autosens_cfg or AutosensConfig()
        self.prep_cfg = prep_cfg or AutotunePrepConfig()

        self.enable_autotune = enable_autotune
        self.enable_autosens = enable_autosens
        self.enable_sick_detection = enable_sick_detection

        self.patients: Dict[str, AdaptiveState] = {}

    def initialize_patient(
        self,
        name: str,
        datetime: object,
        isf_pump: float,
        cr_pump: float,
        basal_pump: float,
    ):
        """Initialize adaptive state for a new patient."""
        self.patients[name] = AdaptiveState(
            isf=isf_pump,
            isf_pump=isf_pump,
            cr=cr_pump,
            cr_pump=cr_pump,
            basal=basal_pump,
            basal_pump=basal_pump,
            sim_start=datetime,
            last_autotune=datetime,
            autosens_buffer=AutosensBuffer(self.autosens_cfg),
            incremental_engine=IncrementalAutotunePrep(self.prep_cfg),
        )

    def record_delivery(self, name: str, datetime: object, basal_uhr: float, bolus: float):
        """
        Record actual delivered insulin into df_history.
        Must be called AFTER loop recommendation, from the controller.
        Powers sick detection — df_history basal/bolus columns.
        """
        if name not in self.patients:
            return
        state = self.patients[name]
        ts = pd.to_datetime(datetime, utc=True)
        state.df_history.loc[ts, "basal"] = float(basal_uhr)
        state.df_history.loc[ts, "bolus"] = float(bolus)

    # ──────────────────────────────────────────────────────────────────────
    #   Sick detection
    # ──────────────────────────────────────────────────────────────────────

    def _update_sick_flag(self, state: AdaptiveState, current_time):
        """
        Illness detection via sustained hyperglycemia duration.

        Meal spikes: BG > 180 for 1-3 hours, then returns to range.
        Illness:     BG > 180 for 6+ hours continuously.

        Also uses autotune parameter drift as a backup signal (fires after
        autotune has had a chance to adapt to illness state).
        """
        if not self.enable_sick_detection:
            return

        elapsed = current_time - state.sim_start
        if elapsed < self.warmup_duration:
            state._resistance_streak = 0
            return

        HYPER_THRESHOLD = 180
        HYPER_STREAK_TRIGGER = 72   # 72 × 5min = 6 hours

        # --- Primary: sustained hyperglycemia streak ---
        hyper_signal = False
        if "CGM" in state.df_history.columns and len(state.df_history) >= HYPER_STREAK_TRIGGER:
            recent_cgm = state.df_history["CGM"].iloc[-HYPER_STREAK_TRIGGER:]
            consecutive_hyper = 0
            for val in reversed(recent_cgm.values):
                if val > HYPER_THRESHOLD:
                    consecutive_hyper += 1
                else:
                    break
            state._resistance_streak = consecutive_hyper
            hyper_signal = consecutive_hyper >= HYPER_STREAK_TRIGGER

            if consecutive_hyper > 0 and consecutive_hyper % 12 == 0:
                logger.debug(
                    "Hyper streak: %d steps (%.0f min) BG>%d sick=%s",
                    consecutive_hyper, consecutive_hyper * 5,
                    HYPER_THRESHOLD, state.sick,
                )

        # --- Backup: autotune parameter drift (only if autotune has run) ---
        autotune_has_run = (
            abs(state.isf   - state.isf_pump)  > 0.01 or
            abs(state.basal - state.basal_pump) > 0.001
        )
        parameter_drift = False
        if autotune_has_run:
            basal_drift = state.basal / state.basal_pump
            isf_drift   = state.isf_pump / state.isf
            parameter_drift = (basal_drift > 1.05) or (isf_drift > 1.05)

        # ── DETECTION (only when not already sick) ──────────────────────
        if not state.sick:
            trigger = hyper_signal or parameter_drift
            if trigger:
                state.sick = True
                state.sick_start = current_time
                logger.warning(
                    "SICK DETECTED at %s: hyper_streak=%d steps (%.0f min) "
                    "parameter_drift=%s autotune_ran=%s",
                    current_time,
                    state._resistance_streak,
                    state._resistance_streak * 5,
                    parameter_drift, autotune_has_run,
                )
            return

        # ── RECOVERY (only when already sick) ───────────────────────────
        RECOVERY_STEPS = 24  # 2 hours
        bg_recovered = False
        if "CGM" in state.df_history.columns and len(state.df_history) >= RECOVERY_STEPS:
            recent_cgm = state.df_history["CGM"].iloc[-RECOVERY_STEPS:]
            bg_recovered = float((recent_cgm <= HYPER_THRESHOLD).mean()) > 0.75

        if bg_recovered:
            if state.sick_start and (current_time - state.sick_start) > timedelta(hours=6):
                state.sick = False
                state.sick_start = None

                if autotune_has_run:
                    alpha     = 0.7
                    old_isf   = state.isf
                    old_basal = state.basal
                    state.isf   = alpha * state.isf_pump   + (1 - alpha) * state.isf
                    state.basal = alpha * state.basal_pump + (1 - alpha) * state.basal
                    logger.info(
                        "SICK CLEARED at %s — snapped: ISF %.3f→%.3f  Basal %.4f→%.4f",
                        current_time, old_isf, state.isf, old_basal, state.basal,
                    )
                else:
                    logger.info(
                        "SICK CLEARED at %s — no snap (autotune never ran)",
                        current_time,
                    )

    # ──────────────────────────────────────────────────────────────────────
    #   Main step
    # ──────────────────────────────────────────────────────────────────────

    def manage_step(
        self,
        name: str,
        datetime: object,
        glucose: float,
        json_input: dict,
    ) -> Tuple[float, float, float, float, float]:
        """
        Execute one adaptation step.
        Returns: (effective_isf, effective_basal, effective_cr, target_min, target_max)
        """
        if name not in self.patients:
            raise ValueError(f"Patient {name} not initialized.")

        state = self.patients[name]

        #self._append(state, datetime, glucose, json_input)
        self._append(state, datetime, glucose, json_input)  

        self.rebuild_buckets(state)

        point = self._make_autosens_point_from_buckets(state, glucose)

        if self.enable_autosens and point:
            for p in point:
                state.autosens_buffer.push(p)
            self._update_autosens(state, datetime)

        if self.enable_sick_detection:
            self._update_sick_flag(state, datetime)

        if self.enable_autotune and not state.sick:
            self._maybe_autotune(state, datetime)

        ts = pd.to_datetime(datetime, utc=True)
        if ts.hour == 0 and ts.minute == 0:
            logger.warning(
                "DAILY SUMMARY %s: BG=%.1f autosens=%.3f sick=%s isf=%.3f basal=%.4f",
                datetime, glucose, state.autosens_ratio, state.sick,
                state.isf, state.basal,
            )

        return self._get_effective_values(state)

    # ──────────────────────────────────────────────────────────────────────
    #   Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _append(self, state, datetime, glucose, json_input):
        ts = pd.to_datetime(datetime, utc=True)
        state.df_history.loc[ts, "CGM"] = glucose

        if len(state.json_history) < len(state.df_history):
            state.json_history.append(json_input)
        else:
            state.json_history[-1] = json_input

        max_rows = self.max_window_days * 288
        if len(state.df_history) > max_rows:
            state.df_history  = state.df_history.iloc[-max_rows:]
            state.json_history = state.json_history[-max_rows:]

    def _make_autosens_point_from_buckets(self, state, glucose):
        buckets = state.prepared_buckets
        if not buckets:
            return None

        points = (
            buckets.get("ISFGlucoseData", []) +
            buckets.get("basalGlucoseData", [])
        )
        if points is None or len(points) < 5:
            return None

        non_meal = []
        for row in points:
            if row.get("COB", 0) > 0:
                continue
            non_meal.append(AutosensPoint(
                deviation=float(row.get("deviation", 0)),
                bgi=float(row.get("BGI", 0)),
                iob=float(row.get("IOB", 0)),
                cob=float(row.get("COB", 0)),
                glucose=float(row.get("glucose", 100)),
            ))
        return non_meal

    def _update_autosens(self, state, current_time):
        if len(state.df_history) < 5:
            return

        elapsed = pd.to_datetime(current_time, utc=True) - pd.to_datetime(state.sim_start, utc=True)
        if elapsed < self.warmup_duration:
            state.autosens_ratio = 1.0
            return

        result = compute_autosens(
            state.autosens_buffer,
            self.autosens_cfg,
            max_basal=state.basal_pump,
            isf=state.isf,
        )
        raw_ratio = result["ratio"]

        if not state.sick:
            state.autosens_ratio = raw_ratio
            if abs(raw_ratio - 1.0) > 0.01:
                logger.debug(
                    "Autosens NORMAL: ratio=%.3f (>1=resistant, <1=sensitive)",
                    raw_ratio,
                )
            return

        # Sick mode: smooth but never allow less insulin than baseline
        damping  = 0.2
        smoothed = damping * raw_ratio + (1 - damping) * state.autosens_ratio
        smoothed = max(smoothed, 1.0)
        smoothed = min(smoothed, self.autosens_cfg.autosens_max)
        logger.debug("Autosens SICK: raw=%.3f → smoothed=%.3f", raw_ratio, smoothed)
        state.autosens_ratio = smoothed

    def rebuild_buckets(self, state: AdaptiveState):
        prep_cfg = AutotunePrepConfig(
            basal_rate=state.basal,
            isf=state.isf,
            carb_ratio=state.cr,
            min_5m_carbimpact=self.prep_cfg.min_5m_carbimpact,
            categorize_uam_as_basal=self.prep_cfg.categorize_uam_as_basal,
        )
        categorized = categorized_buckets(
            state.df_history,
            loop_algorithm_input=self._merge_json_inputs(state.json_history),
            cfg=prep_cfg,
            json_history=state.json_history,
            incremental_engine=state.incremental_engine,
        )
        state.prepared_buckets = categorized

    def _maybe_autotune(self, state, datetime):
        MIN_DATA_BEFORE_AUTOTUNE = timedelta(days=7)
        if datetime - state.sim_start < MIN_DATA_BEFORE_AUTOTUNE:
            return
        if datetime - state.last_autotune < self.autotune_interval:
            return
        if not state.prepared_buckets:
            return
        if state.sick:
            return

        result = run_autotune(
            prepared_buckets=[state.prepared_buckets],
            current_values={"ISF": state.isf, "Basal": state.basal},
            pump_values={"ISF": state.isf_pump, "Basal": state.basal_pump},
            cfg=self.autotune_cfg,
        )
        logger.warning(
            "AUTOTUNE FIRED: ISF %.3f→%.3f, Basal %.4f→%.4f, sick=%s",
            state.isf, result["ISF"]["newValue"],
            state.basal, result["Basal"]["newValue"],
            state.sick,
        )
        state.isf   = result["ISF"]["newValue"]
        state.basal = result["Basal"]["newValue"]
        state.isf_history.append((datetime, state.isf))
        state.basal_history.append((datetime, state.basal))
        state.last_autotune = datetime

    def _get_effective_values(self, state):
        ratio = state.autosens_ratio

        # Safety net: lock ratio during warmup
        if ratio != 1.0 and self.enable_autosens:
            elapsed = None
            if state.sim_start is not None and not state.df_history.empty:
                elapsed = state.df_history.index[-1] - pd.to_datetime(state.sim_start, utc=True)
            if elapsed is not None and elapsed < self.warmup_duration:
                ratio = 1.0

        # If autosens disabled, always 1.0
        if not self.enable_autosens:
            ratio = 1.0

        if state.sick:
            ratio = max(ratio, 1.0)

        isf   = apply_autosens_to_isf(state.isf, ratio)
        basal = apply_autosens_to_basal(state.basal, ratio)

        target_min, target_max = apply_autosens_to_target(
            target_min=self.target - 10,
            target_max=self.target + 10,
            autosens_ratio=ratio,
            sensitivity_raises_target=True,
            resistance_lowers_target=False,
        )
        return isf, basal, state.cr, target_min, target_max

    def _merge_json_inputs(self, json_history: list) -> dict:
        if not json_history:
            return None
        merged = dict(json_history[-1])
        merged['doses'] = list(merged.get('doses', []) or [])
        seen_starts = {d.get('startDate') for d in merged['doses']}
        for json_input in reversed(json_history[:-1]):
            for dose in (json_input.get('doses') or []):
                start = dose.get('startDate')
                if start and start not in seen_starts:
                    merged['doses'].append(dose)
                    seen_starts.add(start)
        merged['doses'].sort(key=lambda d: d.get('startDate', ''))
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

    # ──────────────────────────────────────────────────────────────────────
    #   Public API
    # ──────────────────────────────────────────────────────────────────────

    def get_isf_history(self, patient_name: str) -> list:
        if patient_name not in self.patients:
            raise KeyError(f"Patient '{patient_name}' not found")
        return self.patients[patient_name].isf_history

    def get_current_isf(self, patient_name: str) -> float:
        if patient_name not in self.patients:
            raise KeyError(f"Patient '{patient_name}' not found")
        return self.patients[patient_name].isf

    def get_current_effective_isf(self, patient_name: str) -> float:
        """Effective ISF = autotune_isf / autosens_ratio."""
        if patient_name not in self.patients:
            raise KeyError(f"Patient '{patient_name}' not found")
        state = self.patients[patient_name]
        ratio = state.autosens_ratio if self.enable_autosens else 1.0
        if state.sick:
            ratio = max(ratio, 1.0)
        return round(state.isf / ratio, 3)

    def get_current_pump_isf(self, patient_name: str) -> float:
        if patient_name not in self.patients:
            raise KeyError(f"Patient '{patient_name}' not found")
        return self.patients[patient_name].isf_pump

    def get_current_autosens_ratio(self, patient_name: str) -> float:
        if patient_name not in self.patients:
            raise KeyError(f"Patient '{patient_name}' not found")
        return self.patients[patient_name].autosens_ratio

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
        return self.patients[patient_name].basal

    def get_autosens_log(self, patient_name: str) -> list:
        if patient_name not in self.patients:
            raise KeyError(f"Patient '{patient_name}' not found")
        return self.patients[patient_name].autosens_log

    def reset(self):
        self.patients = {}