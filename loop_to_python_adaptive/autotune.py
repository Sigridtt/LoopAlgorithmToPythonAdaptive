# loop_to_python_adaptive/autotune.py
"""
Generic parameter autotune core following oref0 lib/autotune/index.js.

This module provides the common tuning algorithm for any parameter (ISF, CR, Basal).
Parameter-specific extractors and setters are passed in via config objects.

oref0 tuning steps (generic):
  1. ratio(i) = 1 + deviation(i) / BGI(i)
  2. fullNewValue = current * median(ratios)
  3. adjustedValue = adjustmentFraction * fullNewValue + (1-adjustmentFraction) * pump_value
     cap to [pump_value/autosens_max, pump_value/autosens_min]
  4. newValue = 0.8 * current + 0.2 * adjustedValue
     cap to same bounds
  5. If fewer than min_points → leave unchanged
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional
import numpy as np
import copy




# ═══════════════════════════════════════════════════════════════════════════
#   GENERIC CONFIG & CORE TUNING
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AutotuneConfig:
    """
    Generic autotune config for any parameter.
    
    adjustment_fraction : Blend toward pump value (1.0 = full adjustment) for ISF and basal
    cr_adjustment_fraction : Blend toward pump value for CR
    autosens_max/min   : Safety caps as multiples of pump value
    min_points         : Minimum data points required before tuning
    min_bgi_abs        : Skip points where |BGI| is too small
    """
    min_points: int = 10
    adjustment_fraction: float = 1.0
    cr_adjustment_fraction: float = 0.5 
    autosens_max: float = 1.2
    autosens_min: float = 0.7
    min_bgi_abs: float = 1e-6



# Tune ISF OR basal
def tune_parameter(
    *,
    param_name: str,
    current_value: float,
    glucose_data: list[dict[str, Any]],
    pump_value: float,
    cfg: AutotuneConfig,
) -> dict[str, Any]:
    """
    Parameter tuning, following oref0 algorithm.

    Parameters
    ----------
    param_name     : Human-readable name for logging ("ISF", "Basal")
    current_value  : Current parameter value (what we're tuning from)
    glucose_data   : List of dicts with "deviation" and "BGI" keys
    pump_value     : Original pump value (safety anchor)
    cfg            : AutotuneConfig

    Returns
    -------
    dict with keys:
        newValue       : Parameter value to use next iteration
        p50_ratio      : Median of per-point ratios
        n_points       : Number of usable data points
        reason         : Status string
    """

    # Step 1: Compute per-point ratios
    ratios = []
    for p in glucose_data:
        bgi = float(p.get("BGI", 0))
        if abs(bgi) < cfg.min_bgi_abs:
            continue
        dev = float(p.get("deviation", 0))
        r = 1.0 + dev / bgi
        if not np.isfinite(r):
            continue
        ratios.append(r)

    # Step 2: Check minimum data
    if len(ratios) < cfg.min_points:
        return {
            "newValue": current_value,
            "p50_ratio": None,
            "n_points": len(ratios),
            "reason": f"Only {len(ratios)} {param_name} points (<{cfg.min_points}); {param_name} unchanged.",
        }

    # Step 3: Compute median ratio and full new value
    p50_ratio = float(np.median(ratios))
    full_new_value = round(current_value * p50_ratio, 3)



    # Step 4: Blend toward pump_value
    max_value = pump_value / cfg.autosens_min
    min_value = pump_value / cfg.autosens_max

    if full_new_value < 0:
        adjusted_value = current_value
    else:
        adjusted_value = (
            cfg.adjustment_fraction * full_new_value
            + (1.0 - cfg.adjustment_fraction) * pump_value
        )

    # Cap adjusted value
    adjusted_value = max(min_value, min(max_value, adjusted_value))

    # Step 5: Slow 20% update
    new_value = 0.8 * current_value + 0.2 * adjusted_value

    # Cap final value
    new_value = max(min_value, min(max_value, new_value))

    # Round
    new_value = round(new_value, 3)
    adjusted_value = round(adjusted_value, 3)
    p50_ratio = round(p50_ratio, 3)

    print(
        f"[{param_name}] p50_ratio={p50_ratio} | "
        f"Old={current_value:.3f} fullNew={full_new_value:.3f} "
        f"adjusted={adjusted_value:.3f} new={new_value:.3f}"
    )

    return {
        "newValue": new_value,
        "p50_ratio": p50_ratio,
        "n_points": len(ratios),
        "reason": "OK",
    }

def tune_cr(
        *,
        current_cr: float,
        pump_cr: float,
        cr_data: list[dict[str, Any]],
        isf: float, 
        cfg: AutotuneConfig
) -> dict[str,Any]:
    """
    Parameter tuning for CR, following oref0 algorithm.

    Parameters
    ----------
    current_cr  : Current parameter value (what we're tuning from)
    pump_cr     : Original pump value (safety anchor)
    cr_data   : List of dicts with "deviation" and "BGI" keys
    isf       : Current ISF
    cfg            : AutotuneConfig

    Returns
    -------
    dict with keys:
        newValue       : Parameter value to use next iteration
        p50_ratio      : Median of per-point ratios
        n_points       : Number of usable data points
        reason         : Status string
    """
    ratios = []

    for d in cr_data:
        carbs = float(d.get("CRCarbs", 0))
        if carbs <= 0:
            continue
        bg0 = float(d.get("CRInitialBG", 0))
        bg1 = float(d.get("CREndBG", 0))

        iob0 = float(d.get("CRInitialIOB", 0))
        iob1 = float(d.get("CREndIOB", 0))

        delta_bg = bg1 - bg0
        delta_iob = iob0 - iob1

        insulin_used = delta_iob + (delta_bg / isf)

        if insulin_used <= 0:
            continue

        implied_cr = carbs / insulin_used

        if not np.isfinite(implied_cr):
            continue

        ratio = implied_cr / current_cr

        # safety clamp
        if ratio < 0.5 or ratio > 1.5:
            continue

        ratios.append(ratio)

    if len(ratios) < cfg.min_points:
        return {
            "newValue": current_cr,
            "reason": "Not enough CR data",
            "n_points": len(ratios),
        }

    p50 = float(np.median(ratios))

    full_new = current_cr * p50

    min_val = pump_cr / cfg.autosens_max
    max_val = pump_cr / cfg.autosens_min

    adjusted = (
        cfg.cr_adjustment_fraction * full_new +
        (1 - cfg.cr_adjustment_fraction) * pump_cr
    )

    adjusted = max(min_val, min(max_val, adjusted))

    new_value = 0.8 * current_cr + 0.2 * adjusted
    new_value = max(min_val, min(max_val, new_value))

    return {
        "newValue": round(new_value, 3),
        "p50_ratio": round(p50, 3),
        "n_points": len(ratios),
        "reason": "OK",
    }
    



def run_autotune(
    *,
    prepared_buckets: list[dict],
    current_values: dict[str, float],
    pump_values: dict[str, float],
    cfg: AutotuneConfig = AutotuneConfig(),
) -> dict[str, Any]:
    """
    Expected input:

    current_values = {
        "ISF": float,
        "Basal": float,
        "CR": float,
    }

    pump_values = same structure

    prepared_buckets = list of outputs from prep module
    """

    # collect all data across buckets
    isf_points = []
    basal_points = []
    csf_points = []
    cr_data = []

    for bucket in prepared_buckets:
        isf_points.extend(bucket.get("ISFGlucoseData", []))
        basal_points.extend(bucket.get("basalGlucoseData", []))
        csf_points.extend(bucket.get("CSFGlucoseData", []))
        cr_data.extend(bucket.get("CRData", []))

    # ── ISF ─────────────────────────────────
    isf_result = tune_parameter(
        param_name="ISF",
        current_value=current_values["ISF"],
        pump_value=pump_values["ISF"],
        glucose_data=isf_points,
        cfg=cfg,
    )

    # ── BASAL ───────────────────────────────
    basal_result = tune_parameter(
        param_name="Basal",
        current_value=current_values["Basal"],
        pump_value=pump_values["Basal"],
        glucose_data=basal_points,
        cfg=cfg,
    )
    """
    # ── CR (uses CRData, not CSF!) ──────────
    cr_result = tune_cr(
        current_cr=current_values["CR"],
        pump_cr=pump_values["CR"],
        cr_data=cr_data,
        isf=current_values["ISF"],  # important!
        cfg=cfg,
    )
    """
    return {
        "ISF": isf_result,
        "Basal": basal_result,
        #"CR": cr_result,
    }