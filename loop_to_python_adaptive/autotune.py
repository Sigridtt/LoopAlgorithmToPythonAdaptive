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

from loop_to_python_adaptive.autotune_prep import (
    AutotunePrepConfig,
    prepare_for_autotune_isf,
)


# ═══════════════════════════════════════════════════════════════════════════
#   GENERIC CONFIG & CORE TUNING
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AutotuneConfig:
    """
    Generic autotune config for any parameter.
    
    adjustment_fraction : Blend toward pump value (1.0 = full adjustment)
    autosens_max/min   : Safety caps as multiples of pump value
    min_points         : Minimum data points required before tuning
    min_bgi_abs        : Skip points where |BGI| is too small
    """
    min_points: int = 10
    adjustment_fraction: float = 1.0
    autosens_max: float = 1.2
    autosens_min: float = 0.7
    min_bgi_abs: float = 1e-6


@dataclass(frozen=True)
class ParameterTuner:
    """
    Specifies how to extract/set a specific parameter from/to loop_algorithm_input.
    """
    name: str                              # "ISF", "CR", "Basal"
    extract_pump: Callable[[dict], float]  # Extracts pump (baseline) value
    extract_current: Callable[[dict], float]  # Extracts current value
    update_profile: Callable[[dict, float], dict]  # Sets new value and returns updated dict
    data_category: str                     # Which category to use from autotune_prep
                                          # "ISFGlucoseData", "basalGlucoseData", "CSFGlucoseData"


def tune_parameter(
    *,
    param_name: str,
    current_value: float,
    glucose_data: list[dict[str, Any]],
    pump_value: float,
    cfg: AutotuneConfig = AutotuneConfig(),
) -> dict[str, Any]:
    """
    Generic parameter tuning, following oref0 algorithm.

    Parameters
    ----------
    param_name     : Human-readable name for logging ("ISF", "CR", "Basal")
    current_value  : Current parameter value (what we're tuning from)
    glucose_data   : List of dicts with "deviation" and "BGI" keys
    pump_value     : Original pump value (safety anchor)
    cfg            : AutotuneConfig

    Returns
    -------
    dict with keys:
        newValue       : Parameter value to use next iteration
        fullNewValue   : Raw value implied by data
        adjustedValue  : After blend and cap
        p50_ratio      : Median of per-point ratios
        n_points       : Number of usable data points
        reason         : Status string
    """

    # Step 1: Compute per-point ratios
    ratios: list[float] = []
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
            "fullNewValue": None,
            "adjustedValue": None,
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
        "fullNewValue": full_new_value,
        "adjustedValue": adjusted_value,
        "p50_ratio": p50_ratio,
        "n_points": len(ratios),
        "reason": "OK",
    }


# ═══════════════════════════════════════════════════════════════════════════
#   EXTRACTION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def extract_pump_isf(loop_algorithm_input: dict) -> float:
    """Extract pump ISF from loop_algorithm_input."""
    sensitivity = loop_algorithm_input.get("sensitivity", [])
    if not sensitivity:
        raise ValueError("loop_algorithm_input has no 'sensitivity' key")
    return float(sensitivity[0]["value"])


def extract_pump_basal(loop_algorithm_input: dict) -> float:
    """Extract pump basal rate from loop_algorithm_input."""
    basal = loop_algorithm_input.get("basal", [])
    if not basal:
        raise ValueError("loop_algorithm_input has no 'basal' key")
    return float(basal[0]["value"])


def extract_pump_cr(loop_algorithm_input: dict) -> float:
    """Extract pump carb ratio from loop_algorithm_input."""
    carb_ratio = loop_algorithm_input.get("carbRatio", [])
    if not carb_ratio:
        raise ValueError("loop_algorithm_input has no 'carbRatio' key")
    return float(carb_ratio[0]["value"])


# ═══════════════════════════════════════════════════════════════════════════
#   PROFILE UPDATE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def update_profile_isf(
    loop_algorithm_input: dict,
    new_isf: float,
) -> dict:
    """Update ISF in profile by scaling the sensitivity schedule."""
    updated = copy.deepcopy(loop_algorithm_input)
    old_isf = extract_pump_isf(updated)
    if old_isf == 0:
        return updated
    ratio = new_isf / old_isf
    for entry in updated.get("sensitivity", []):
        entry["value"] = round(float(entry["value"]) * ratio, 3)
    return updated


def update_profile_basal(
    loop_algorithm_input: dict,
    new_basal: float,
) -> dict:
    """Update basal in profile by scaling the basal schedule."""
    updated = copy.deepcopy(loop_algorithm_input)
    old_basal = extract_pump_basal(updated)
    if old_basal == 0:
        return updated
    ratio = new_basal / old_basal
    for entry in updated.get("basal", []):
        entry["value"] = round(float(entry["value"]) * ratio, 3)
    return updated


def update_profile_cr(
    loop_algorithm_input: dict,
    new_cr: float,
) -> dict:
    """Update carb ratio in profile by scaling the carbRatio schedule."""
    updated = copy.deepcopy(loop_algorithm_input)
    old_cr = extract_pump_cr(updated)
    if old_cr == 0:
        return updated
    ratio = new_cr / old_cr
    for entry in updated.get("carbRatio", []):
        entry["value"] = round(float(entry["value"]) * ratio, 3)
    return updated


# ═══════════════════════════════════════════════════════════════════════════
#   PARAMETER TUNER DEFINITIONS (factory functions)
# ═══════════════════════════════════════════════════════════════════════════

def get_isf_tuner() -> ParameterTuner:
    """Create a tuner for ISF."""
    return ParameterTuner(
        name="ISF",
        extract_pump=extract_pump_isf,
        extract_current=extract_pump_isf,  # On first call, current = pump
        update_profile=update_profile_isf,
        data_category="ISFGlucoseData",
    )


def get_basal_tuner() -> ParameterTuner:
    """Create a tuner for basal rate."""
    return ParameterTuner(
        name="Basal",
        extract_pump=extract_pump_basal,
        extract_current=extract_pump_basal,
        update_profile=update_profile_basal,
        data_category="basalGlucoseData",
    )


def get_cr_tuner() -> ParameterTuner:
    """Create a tuner for carb ratio."""
    return ParameterTuner(
        name="CR",
        extract_pump=extract_pump_cr,
        extract_current=extract_pump_cr,
        update_profile=update_profile_cr,
        data_category="CSFGlucoseData",  # CR uses CSF data from autotune_prep
    )


# ═══════════════════════════════════════════════════════════════════════════
#   GENERIC MULTI-ITERATION RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_autotune_parameter_iterations(
    df_windows: list,
    *,
    loop_algorithm_inputs: list[dict],
    tuner: ParameterTuner,
    pump_value: float | None = None,
    current_value: float | None = None,
    n_iterations: int = 1,
    cfg: AutotuneConfig = AutotuneConfig(),
    json_history_list: list[list[dict]] | None = None,
) -> dict[str, Any]:
    """
    Run parameter autotune for n_iterations.

    Parameters
    ----------
    df_windows             : List of DataFrames (one per day)
    loop_algorithm_inputs  : Matching list of Loop JSON dicts
    tuner                  : ParameterTuner specifying the parameter
    pump_value             : Original pump parameter value (anchor)
    current_value          : Current parameter value (what we're tuning from)
    n_iterations           : Number of passes
    cfg                    : AutotuneConfig
    json_history_list      : BGI history for each window

    Returns
    -------
    dict with keys:
        finalValue      : Parameter value after all iterations
        value_history   : List of values after each iteration
        last_result     : Full result from final tune_parameter() call
    """

    # Extract pump value if not provided
    if pump_value is None:
        pump_value = tuner.extract_pump(loop_algorithm_inputs[0])

    if current_value is None:
        current_value = pump_value

    value_history: list[float] = []
    last_result: dict[str, Any] = {}

    for iteration in range(n_iterations):
        print(
            f"\n=== Autotune {tuner.name} iteration {iteration+1}/{n_iterations} "
            f"(current={current_value:.3f}) ==="
        )

        all_points: list[dict[str, Any]] = []

        # Prepare config for this iteration
        # Need basal, isf, cr for prep — use either pump or current
        prep_cfg = AutotunePrepConfig(
            basal_rate=extract_pump_basal(loop_algorithm_inputs[0]),
            isf=extract_pump_isf(loop_algorithm_inputs[0]),
            carb_ratio=extract_pump_cr(loop_algorithm_inputs[0]),
        )

        for i, (df_window, loop_input) in enumerate(
            zip(df_windows, loop_algorithm_inputs)
        ):
            print(f"  Window {i+1}/{len(df_windows)}...", end=" ", flush=True)

            window_json_history = (
                json_history_list[i]
                if json_history_list and i < len(json_history_list)
                else None
            )

            # Prepare all categories from this window
            result = prepare_for_autotune_isf(
                df_window,
                loop_algorithm_input=loop_input,
                cfg=prep_cfg,
                json_history=window_json_history,
            )

            # Extract the category relevant to this parameter
            window_points = result.get(tuner.data_category, [])
            all_points.extend(window_points)

            print(f"{len(window_points)} {tuner.name} points")

        print(f"  Total {tuner.name} points: {len(all_points)}")

        # Tune this parameter
        last_result = tune_parameter(
            param_name=tuner.name,
            current_value=current_value,
            glucose_data=all_points,
            pump_value=pump_value,
            cfg=cfg,
        )

        current_value = last_result["newValue"]
        value_history.append(current_value)

    return {
        "finalValue": current_value,
        "value_history": value_history,
        "last_result": last_result,
    }


# ═══════════════════════════════════════════════════════════════════════════
#   CONVENIENCE WRAPPERS FOR ISF, BASAL, CR
# ═══════════════════════════════════════════════════════════════════════════

def run_autotune_isf_iterations(
    df_windows: list,
    *,
    loop_algorithm_inputs: list[dict],
    pump_isf: float | None = None,
    isf_current: float | None = None,
    n_iterations: int = 1,
    cfg: AutotuneConfig = AutotuneConfig(),
    json_history_list: list[list[dict]] | None = None,
) -> dict[str, Any]:
    """
    Tune ISF (convenience wrapper around run_autotune_parameter_iterations).
    Returns: finalISF, isf_history, last_result
    """
    result = run_autotune_parameter_iterations(
        df_windows,
        loop_algorithm_inputs=loop_algorithm_inputs,
        tuner=get_isf_tuner(),
        pump_value=pump_isf,
        current_value=isf_current,
        n_iterations=n_iterations,
        cfg=cfg,
        json_history_list=json_history_list,
    )
    return {
        "finalISF": result["finalValue"],
        "isf_history": result["value_history"],
        "last_result": result["last_result"],
    }


def run_autotune_basal_iterations(
    df_windows: list,
    *,
    loop_algorithm_inputs: list[dict],
    pump_basal: float | None = None,
    basal_current: float | None = None,
    n_iterations: int = 1,
    cfg: AutotuneConfig = AutotuneConfig(),
    json_history_list: list[list[dict]] | None = None,
) -> dict[str, Any]:
    """
    Tune basal rate (convenience wrapper).
    Returns: finalBasal, basal_history, last_result
    """
    result = run_autotune_parameter_iterations(
        df_windows,
        loop_algorithm_inputs=loop_algorithm_inputs,
        tuner=get_basal_tuner(),
        pump_value=pump_basal,
        current_value=basal_current,
        n_iterations=n_iterations,
        cfg=cfg,
        json_history_list=json_history_list,
    )
    return {
        "finalBasal": result["finalValue"],
        "basal_history": result["value_history"],
        "last_result": result["last_result"],
    }


def run_autotune_cr_iterations(
    df_windows: list,
    *,
    loop_algorithm_inputs: list[dict],
    pump_cr: float | None = None,
    cr_current: float | None = None,
    n_iterations: int = 1,
    cfg: AutotuneConfig = AutotuneConfig(),
    json_history_list: list[list[dict]] | None = None,
) -> dict[str, Any]:
    """
    Tune carb ratio (convenience wrapper).
    Returns: finalCR, cr_history, last_result
    """
    result = run_autotune_parameter_iterations(
        df_windows,
        loop_algorithm_inputs=loop_algorithm_inputs,
        tuner=get_cr_tuner(),
        pump_value=pump_cr,
        current_value=cr_current,
        n_iterations=n_iterations,
        cfg=cfg,
        json_history_list=json_history_list,
    )
    return {
        "finalCR": result["finalValue"],
        "cr_history": result["value_history"],
        "last_result": result["last_result"],
    }