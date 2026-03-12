from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import pandas as pd

from loop_to_python_adaptive.loop_oref_mapping import prepare_isf_glucose_data

"""
This module categorises bucketed CGM data points into:
  - CSFGlucoseData  (meal / carb-absorption)
  - UAMGlucoseData  (unannounced meals)
  - ISFGlucoseData  (insulin-sensitivity periods)
  - basalGlucoseData (basal-rate periods)

Uses loop_to_python_api.api.get_active_insulin for IOB calculations,
mirroring oref0's getIOB call.
"""
@dataclass(frozen=True)
class AutotunePrepConfig:
    """
    Configuration for preparing inputs to autotune_isf.tune_isf_like_oref0.
    """
    action_duration_minutes: int = 300    # DIA assumption 5h for rapid-acting insulin ------------------------------NEED VERIFICATION DUAL HORMONE?
    peak_activity_minutes: int = 75       # typical peak activity time for rapid-acting insulin; oref0 uses 75m 
    history_hours: int = 24               # how much historical data to use for autotune
    step_minutes: int = 5                 # CGM typically reports every 5 min

    cgm_col: str = "CGM"
    bgi_col: str = "BGI"

    # --- oref0-like categorization knobs (simplified) ---
    # classify points as CSF for X minutes after any announced carb entry
    meal_exclusion_minutes: int = 240
    # classify as UAM if deviation > threshold (mg/dL per 5 min)
    uam_deviation_threshold: float = 6.0
    # avoid using near-zero BGI points for ISF (prevents ratio explosions)
    min_bgi_abs_for_isf: float = 0.5


def _parse_loop_ts(s: str) -> datetime:
    # your fixtures use '...Z'; datetime.fromisoformat doesn't parse 'Z' in py<3.11 reliably
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _categorize_points_oref0_like(
    points: list[dict[str, Any]],
    *,
    loop_algorithm_input: dict,
    cfg: AutotunePrepConfig,
) -> dict[str, list[dict[str, Any]]]:
    """
    Simplified oref0-like categorization:
      - CSF: within meal_exclusion_minutes of announced carbs
      - UAM: deviation > uam_deviation_threshold
      - ISF: remaining points with |BGI| >= min_bgi_abs_for_isf
      - basal: everything else
    """
    carb_entries = loop_algorithm_input.get("carbEntries", []) or []
    carb_times = []
    for c in carb_entries:
        if "date" in c and float(c.get("grams", 0) or 0) > 0:
            carb_times.append(_parse_loop_ts(c["date"]))
    carb_times.sort()

    def in_meal_window(t: datetime) -> bool:
        if not carb_times:
            return False
        # find latest carb time <= t
        # (linear scan is fine for small fixtures; can optimize later)
        last = None
        for ct in carb_times:
            if ct <= t:
                last = ct
            else:
                break
        if last is None:
            return False
        return t <= last + timedelta(minutes=cfg.meal_exclusion_minutes)

    csf: list[dict[str, Any]] = []
    uam: list[dict[str, Any]] = []
    isf: list[dict[str, Any]] = []
    basal: list[dict[str, Any]] = []

    for p in points:
        t = _parse_loop_ts(p["date"]) if isinstance(p["date"], str) else p["date"]
        bgi = float(p["BGI"])
        dev = float(p["deviation"])

        if in_meal_window(t):
            csf.append(p)
        elif dev > cfg.uam_deviation_threshold:
            uam.append(p)
        elif abs(bgi) >= cfg.min_bgi_abs_for_isf:
            isf.append(p)
        else:
            basal.append(p)

    return {
        "ISFGlucoseData": isf,
        "CSFGlucoseData": csf,
        "UAMGlucoseData": uam,
        "basalGlucoseData": basal,
    }


def prepare_for_autotune_isf(
    df: pd.DataFrame,
    *,
    loop_algorithm_input: dict,
) -> dict[str, Any]:
    cfg = AutotunePrepConfig()

    # mapping layer
    df2, all_points = prepare_isf_glucose_data(
        df,
        loop_algorithm_input=loop_algorithm_input,
        cgm_col=cfg.cgm_col,
        bgi_col=cfg.bgi_col,
    )

    buckets = _categorize_points_oref0_like(all_points, loop_algorithm_input=loop_algorithm_input, cfg=cfg)

    return {
        "df": df2,
        "isf_glucose_data": buckets["ISFGlucoseData"],
        **buckets,
    }