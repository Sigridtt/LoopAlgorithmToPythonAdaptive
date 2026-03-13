"""
This module implements the ISF tuning portion of openaps/oref0 lib/autotune/index.js. 
The new ISF is returned and should be fed back in as
`isf_current` on the next iteration.

oref0 ISF tuning steps (from index.js):
  1. ratio(i) = 1 + deviation(i) / BGI(i)                   
     # for each ISF-categorised point
  2. fullNewISF  = isf_current * median(ratios)
  3. adjustedISF = adjustmentFraction * fullNewISF + (1 - adjustmentFraction) * pump_isf
     then cap adjustedISF to [pump_isf / autosens_max, pump_isf / autosens_min]
     # Since adjustmentFraction is set to 1 by default this step most often does nothing, but it is in oref0 so we include it
  4. newISF = 0.8 * isf_current + 0.2 * adjustedISF
     then cap newISF to same bounds
  5. If fewer than min_points ISF data points → leave ISF unchanged
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class AutotuneISFConfig:
    """
    Mirrors the oref0 profile fields used in index.js for ISF tuning.

    adjustment_fraction      : autotune_isf_adjustmentFraction in oref0.
                               Blends fullNewISF toward pump_isf.
                               1.0 = full adjustment (oref0 default),
                               0.0 = no adjustment from pump ISF.
    autosens_max / min       : safety caps as multiples of pump_isf.
                               oref0 defaults: max=1.2, min=0.7.
    min_points               : require at least this many ISF data points
                               before tuning (oref0: 10).
    min_bgi_abs              : skip points where |BGI| is too small to
                               divide by safely.
    """
    min_points: int = 10              # oref0 default
    adjustment_fraction: float = 1    # oref0 default
    autosens_max: float = 1.2         # oref0 default
    autosens_min: float = 0.7         # oref0 default
    min_bgi_abs: float = 1e-6         # avoid divide by tiny numbers

#################
#    HELPERS    #
#################

def extract_pump_isf(loop_algorithm_input: dict) -> float:
    """
    Extract the ISF value from a loop_algorithm_input dict.

    loop_algorithm_input["sensitivity"] is a list of schedule entries:
        [{"startDate": ..., "endDate": ..., "value": <float>}]

    For autotune we use the first entry as the pump's anchor ISF,
    matching how oref0 reads pumpISF from the pump profile.
    """
    sensitivity = loop_algorithm_input.get("sensitivity", [])
    if not sensitivity:
        raise ValueError(
            "loop_algorithm_input has no 'sensitivity' key. "
        )
    return float(sensitivity[0]["value"])


def extract_pump_basal(loop_algorithm_input: dict) -> float:
    """
    Extract the basal rate value from a loop_algorithm_input dict.

    loop_algorithm_input["basal"] is a list of schedule entries:
        [{"startDate": ..., "endDate": ..., "value": <float>}]

    For autotune we use the first entry as the pump's anchor basal rate,
    matching how oref0 reads pumpBasal from the pump profile.
    """
    basal = loop_algorithm_input.get("basal", [])
    if not basal:
        raise ValueError(
            "loop_algorithm_input has no 'basal' key. "
        )
    return float(basal[0]["value"])

def extract_pump_cr(loop_algorithm_input: dict) -> float:
    """
    Extract the carbohydrate ratio (CR) value from a loop_algorithm_input dict.

    loop_algorithm_input["carbRatio"] is a list of schedule entries:
        [{"startDate": ..., "endDate": ..., "value": <float>}]

    For autotune we use the first entry as the pump's anchor carbohydrate ratio,
    matching how oref0 reads pumpCR from the pump profile.
    """
    carb_ratio = loop_algorithm_input.get("carbRatio", [])
    if not carb_ratio:
        raise ValueError(
            "loop_algorithm_input has no 'carbRatio' key. "
        )
    return float(carb_ratio[0]["value"])


def tune_isf(
    *,
    isf_current: float,
    isf_glucose_data: list[dict[str, Any]],
    pump_isf: float,
    cfg: AutotuneISFConfig = AutotuneISFConfig(),
) -> dict[str, Any]:
    """
    Tune ISF for one iteration, exactly following oref0 index.js.

    Parameters
    ----------
    isf_current      : ISF used this iteration (mg/dL per U). On the first
                       call this is the user's pump ISF. On later calls
                       pass in the previous iteration's `newISF`.
    isf_glucose_data : List of dicts with at least keys "deviation" and "BGI".
                       These are the ISF-categorised points from autotune_prep.
    pump_isf         : The user's original pump ISF, used as the safety anchor.
                       In oref0 this never changes across iterations.
    cfg              : AutotuneISFConfig.

    Returns
    -------
    dict with keys:
        newISF        : ISF to use next iteration.
        fullNewISF    : Raw ISF implied by the data (before blending/capping).
        adjustedISF   : After adjustmentFraction blend and first cap.
        p50_ratio     : Median of per-point ratios.
        n_points      : Number of usable ISF data points.
        reason        : Human-readable status string.
    """

    # Step 1: compute per-point ratios
 
    ratios: list[float] = []
    for p in isf_glucose_data:
        bgi = float(p["BGI"])
        if abs(bgi) < cfg.min_bgi_abs:
            continue
        dev = float(p["deviation"])
        r = 1.0 + dev / bgi
        if not np.isfinite(r):
            continue
        ratios.append(r)


    # Step 2: require minimum data
    # oref0: "leave ISF unchanged if fewer than 10 ISF data points"
  
    if len(ratios) < cfg.min_points:
        print(
            f"Only found {len(ratios)} ISF data points, "
            f"leaving ISF unchanged at {isf_current}"
        )
        return {
            "newISF":      isf_current,
            "fullNewISF":  None,
            "adjustedISF": None,
            "p50_ratio":   None,
            "n_points":    len(ratios),
            "reason":      f"Only {len(ratios)} ISF points (<{cfg.min_points}); ISF unchanged.",
        }

  
    # Step 3: fullNewISF = isf_current * median(ratios)

    p50_ratio    = float(np.median(ratios))
    full_new_isf = round(isf_current * p50_ratio, 3)


    # Step 4: adjustedISF = blend fullNewISF toward pump_isf
    # oref0: adjustedISF = adjustmentFraction*fullNewISF + (1 - adjustmentFraction)*pumpISF
    # Then cap to [pump_isf/autosens_max, pump_isf/autosens_min]
   
    # low autosens ratio = high ISF  → maxISF = pumpISF / autosens_min
    # high autosens ratio = low ISF  → minISF = pumpISF / autosens_max
    max_isf = pump_isf / cfg.autosens_min
    min_isf = pump_isf / cfg.autosens_max

    if full_new_isf < 0:
        # oref0: "if fullNewISF < 0, adjustedISF = ISF" (leave unchanged)
        adjusted_isf = isf_current
    else:
        adjusted_isf = (
            cfg.adjustment_fraction * full_new_isf
            + (1.0 - cfg.adjustment_fraction) * pump_isf
        )

    # first cap
    if adjusted_isf > max_isf:
        print(
            f"Limiting adjusted ISF of {adjusted_isf:.2f} to {max_isf:.2f} "
            f"(pump ISF {pump_isf} / autosens_min {cfg.autosens_min})"
        )
        adjusted_isf = max_isf
    elif adjusted_isf < min_isf:
        print(
            f"Limiting adjusted ISF of {adjusted_isf:.2f} to {min_isf:.2f} "
            f"(pump ISF {pump_isf} / autosens_max {cfg.autosens_max})"
        )
        adjusted_isf = min_isf

    
    # Step 5: slow 20% update toward adjustedISF
    
    new_isf = 0.8 * isf_current + 0.2 * adjusted_isf

    # second cap (same bounds)
    if new_isf > max_isf:
        print(
            f"Limiting ISF of {new_isf:.2f} to {max_isf:.2f} "
            f"(pump ISF {pump_isf} / autosens_min {cfg.autosens_min})"
        )
        new_isf = max_isf
    elif new_isf < min_isf:
        print(
            f"Limiting ISF of {new_isf:.2f} to {min_isf:.2f} "
            f"(pump ISF {pump_isf} / autosens_max {cfg.autosens_max})"
        )
        new_isf = min_isf

    new_isf      = round(new_isf, 3)
    adjusted_isf = round(adjusted_isf, 3)
    p50_ratio    = round(p50_ratio, 3)

    print(
        f"p50_ratio: {p50_ratio}  "
        f"Old ISF: {isf_current}  fullNewISF: {full_new_isf}  "
        f"adjustedISF: {adjusted_isf}  newISF: {new_isf}"
    )

    return {
        "newISF":      new_isf,
        "fullNewISF":  full_new_isf,
        "adjustedISF": adjusted_isf,
        "p50_ratio":   p50_ratio,
        "n_points":    len(ratios),
        "reason":      "OK",
    }


###############
#    ENTRY    #
###############


def run_autotune_isf_iterations(
    df_windows: list,                   # list of pd.DataFrames, one per day
    *,
    loop_algorithm_inputs: list[dict],  # one per window, aligned with df_windows
    n_iterations: int = 1,              # number of passes
    cfg: AutotuneISFConfig = AutotuneISFConfig(),
) -> dict[str, Any]:
    """
    Run ISF autotune for `n_iterations` passes over the data windows.

    On each iteration the tuned ISF from the previous pass is fed back in
    as `isf_current` for the next pass — exactly as oref0 re-runs daily.

    Parameters
    ----------
    df_windows           : List of DataFrames (e.g. one per 24h window).
    loop_algorithm_inputs: Matching list of LoopAlgorithm JSON input dicts,
                           one per window (used for BGI prediction and carb entries).
    pump_isf             : User's pump ISF — fixed anchor for safety caps.
    pump_basal           : User's pump basal rate (U/hr).
    pump_cr              : User's pump carb ratio (g/U).
    n_iterations         : How many full passes over the windows to run.
    cfg                  : AutotuneISFConfig.

    Returns
    -------
    dict with keys:
        finalISF     : The ISF after all iterations.
        isf_history  : List of ISF values after each iteration (length = n_iterations).
        last_result  : Full result dict from the final tune_isf() call.
    """
    from loop_to_python_adaptive.autotune_prep import (
        AutotunePrepConfig,
        prepare_for_autotune_isf,
    )
    pump_isf = extract_pump_isf(loop_algorithm_inputs[0])
    pump_basal = extract_pump_basal(loop_algorithm_inputs[0])
    pump_cr = extract_pump_cr(loop_algorithm_inputs[0])

    isf_current = pump_isf
    isf_history: list[float] = []
    last_result: dict[str, Any] = {}

    for iteration in range(n_iterations):
        print(f"\n=== Autotune ISF iteration {iteration + 1}/{n_iterations} "
              f"(current ISF: {isf_current:.3f}) ===")

        # Accumulate ISF glucose data across all windows for this iteration
        all_isf_points: list[dict[str, Any]] = []

        prep_cfg = AutotunePrepConfig(
            basal_rate=pump_basal,
            isf=isf_current,        # ← updated each iteration
            carb_ratio=pump_cr,
        )

        for i, (df_window, loop_input) in enumerate(
            zip(df_windows, loop_algorithm_inputs)
        ):
            print(f"  Window {i + 1}/{len(df_windows)} ...", end=" ", flush=True)
            result = prepare_for_autotune_isf(
                df_window,
                loop_algorithm_input=loop_input,
                cfg=prep_cfg,
            )
            window_isf_points = result["ISFGlucoseData"]
            all_isf_points.extend(window_isf_points)
            print(f"{len(window_isf_points)} ISF points")

        print(f"  Total ISF points this iteration: {len(all_isf_points)}")

        # Tune ISF on the accumulated points
        last_result = tune_isf(
            isf_current=isf_current,
            isf_glucose_data=all_isf_points,
            pump_isf=pump_isf,
            cfg=cfg,
        )

        isf_current = last_result["newISF"]
        isf_history.append(isf_current)

    return {
        "finalISF":   isf_current,
        "isf_history": isf_history,
        "last_result": last_result,
    }

def update_profile_isf(
    loop_algorithm_input: dict,
    new_isf_scalar: float,
) -> dict:
    """
    Return a new loop_algorithm_input dict with the ISF updated to new_isf_scalar.

    Called after run_autotune_isf_iterations() to apply the tuned ISF
    back into the profile for the next simulation epoch.

    The sensitivity schedule shape is preserved — all entries are scaled
    by the ratio new_isf / current_pump_isf. This mirrors oref0 behaviour:
    the overall ISF level shifts but the diurnal shape stays intact.

    Parameters
    ----------
    loop_algorithm_input : The current loop_algorithm_input dict.
                           Must contain a "sensitivity" key.
    new_isf_scalar       : The finalISF from run_autotune_isf_iterations().

    Returns
    -------
    A new dict — the original is never mutated.
    """
    import copy
    old_isf = extract_pump_isf(loop_algorithm_input)
    if old_isf == 0:
        return loop_algorithm_input

    ratio = new_isf_scalar / old_isf

    updated = copy.deepcopy(loop_algorithm_input)
    for entry in updated["sensitivity"]:
        entry["value"] = round(float(entry["value"]) * ratio, 3)

    return updated