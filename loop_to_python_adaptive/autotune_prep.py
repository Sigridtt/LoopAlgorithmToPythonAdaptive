"""
TO DO:
Get true therapy settings 
"""
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
"""


@dataclass(frozen=True)
class AutotunePrepConfig:
    """
    Therapy settings and oref0 categorization knobs.

    basal_rate   : U/hr  — currentBasal used for UAM/absorbing thresholds
    isf          : mg/dL per U  (sens in oref0)
    carb_ratio   : g per U
    min_5m_carbimpact: 8 // mg/dL per 5m (8 mg/dL/5m corresponds to 24g/hr at a CSF of 4 mg/dL/g (x/5*60/4)) (lib/profile/index.js line 35 in oref0)
    categorize_uam_as_basal : mirrors oref0 --categorize-uam-as-basal flag
    """
    # Therapy settings
    basal_rate:   float = 1.0  # U/hr — MUST be set to user's actual basal rate;
                               # oref0 reads this from pump profile schedule, no universal default
    isf:          float = 50.0 # mg/dL per U — MUST be set to user's actual ISF;
                               # oref0 uses time-based schedule lookup, no universal default
                               # (50 matches oref0's example profile)
    carb_ratio:   float = 10.0 # g per U — MUST be set to user's actual CR;
                               # oref0 uses time-based schedule lookup, no universal default
                               # (10 matches oref0's example profile)

    # Carb absorption model
    min_5m_carbimpact: float = 8.0  # mg/dL per 5m (8 mg/dL/5m corresponds to 24g/hr at a CSF of 4 mg/dL/g (x/5*60/4)) (lib/profile/index.js line 35 in oref0)

    # UAM post-processing flag
    categorize_uam_as_basal: bool = False

    # Column / field names
    cgm_col:       str = "CGM"
    bgi_col:       str = "BGI"
    avg_delta_col: str = "avgDelta"
    deviation_col: str = "deviation"
    iob_col:       str = "IOB"
    cob_col:       str = "COB"


# Timestamp helper
def _to_utc(val: Any) -> datetime:
    """Coerce ISO string, pd.Timestamp, or datetime → UTC-aware datetime."""
    if isinstance(val, pd.Timestamp):
        dt = val.to_pydatetime()
    elif isinstance(val, str):
        s = val[:-1] + "+00:00" if val.endswith("Z") else val
        dt = datetime.fromisoformat(s)
    elif isinstance(val, datetime):
        dt = val
    else:
        raise TypeError(f"Cannot parse timestamp: {val!r}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


####################
#    CATEGORIZE    #
####################

def _categorize_points(
    points: list[dict],
    *,
    loop_algorithm_input: dict,
    cfg: AutotunePrepConfig,
) -> dict[str, list[dict]]:
    """
    Comparable to oref0's categorize.js 
    Points must be sorted oldest → newest (ascending date).
    Each point dict is expected to have at minimum:
        date, glucose, BGI, avgDelta, deviation, IOB   (all from loop_oref_mapping)
    """
    # --- Build carb treatment list (newest-first; pop() consumes oldest) ---
    carb_entries = loop_algorithm_input.get("carbEntries", []) or []
    treatments: list[dict] = []
    for c in carb_entries:
        grams = float(c.get("grams", 0) or 0)
        if grams >= 1 and "date" in c:
            treatments.append({"carbs": grams, "_ts": _to_utc(c["date"])})
    treatments.sort(key=lambda t: t["_ts"], reverse=True)  # newest first

    # Drop treatments older than the oldest point
    if points:
        oldest_ts = _to_utc(points[0]["date"])
        treatments = [t for t in treatments if t["_ts"] >= oldest_ts]

    # --- oref0 state ---
    calculating_cr  = False
    absorbing       = 0
    uam             = 0
    meal_cob        = 0.0
    meal_carbs      = 0.0
    cr_carbs        = 0.0
    datum_type      = ""

    cr_initial_iob:       float            = 0.0
    cr_initial_bg:        float            = 0.0
    cr_initial_carb_time: Optional[datetime] = None

    CSFGlucoseData:   list[dict] = []
    ISFGlucoseData:   list[dict] = []
    basalGlucoseData: list[dict] = []
    UAMGlucoseData:   list[dict] = []
    CRData:           list[dict] = []

    isf = cfg.isf
    current_basal = cfg.basal_rate
    basal_bgi = round(current_basal * isf / 60 * 5, 2)  # mg/dL/5min

    # --- Main loop: oldest → newest ---
    for raw_point in points:
        point  = dict(raw_point)
        ts     = _to_utc(point["date"])
        bg_ms  = int(ts.timestamp() * 1000)

        # Consume carb treatments that occurred before this BG timestamp
        my_carbs = 0.0
        if treatments:
            oldest = treatments[-1]
            if int(oldest["_ts"].timestamp() * 1000) < bg_ms:
                if oldest["carbs"] >= 1:
                    meal_cob   += oldest["carbs"]
                    meal_carbs += oldest["carbs"]
                    my_carbs    = oldest["carbs"]
                treatments.pop()

        # Retrieve pre-computed values from point dict
        bgi       = float(point["BGI"])
        avg_delta = float(point["avgDelta"])
        deviation = float(point["deviation"])
        iob_val   = float(point.get("IOB", 0.0))

        # BG < 80: zero out positive deviations (oref0 behaviour)
        glucose = point.get("glucose")
        if glucose is not None and glucose < 80 and deviation > 0:
            deviation = 0.0
            point["deviation"] = deviation

        # COB decay
        if meal_cob > 0:
            ci       = max(deviation, cfg.min_5m_carbimpact)
            absorbed = ci * cfg.carb_ratio / isf
            meal_cob = max(0.0, meal_cob - absorbed)

        # CR tracking (mirrors oref0; feeds CRData but doesn't affect categories)
        if meal_cob > 0 or calculating_cr:
            cr_carbs += my_carbs
            if not calculating_cr:
                cr_initial_iob       = iob_val
                cr_initial_bg        = glucose or 0.0
                cr_initial_carb_time = ts

            if meal_cob > 0:
                calculating_cr = True
            elif iob_val > current_basal / 2:
                calculating_cr = True
            else:
                cr_end_time    = ts
                cr_elapsed_min = round((cr_end_time - cr_initial_carb_time).total_seconds() / 60)
                if cr_elapsed_min >= 60:
                    CRData.append({
                        "CRInitialIOB":      cr_initial_iob,
                        "CRInitialBG":       cr_initial_bg,
                        "CRInitialCarbTime": cr_initial_carb_time,
                        "CREndIOB":          iob_val,
                        "CREndBG":           glucose or 0.0,
                        "CREndTime":         cr_end_time,
                        "CRCarbs":           cr_carbs,
                    })
                cr_carbs       = 0.0
                calculating_cr = False

        ##########################################
        # Categorization: CSF → UAM → basal/ISF  #
        ##########################################
        if meal_cob > 0 or absorbing or meal_carbs > 0:
            # --- CSF ---
            if iob_val < current_basal / 2:
                absorbing = 0
            elif deviation > 0:
                absorbing = 1
            else:
                absorbing = 0

            if not absorbing and not meal_cob:
                meal_carbs = 0.0

            if datum_type != "csf":
                point["mealAbsorption"] = "start"
            datum_type = "csf"
            point["mealCarbs"] = meal_carbs
            CSFGlucoseData.append(point)

        else:
            if datum_type == "csf" and CSFGlucoseData:
                CSFGlucoseData[-1]["mealAbsorption"] = "end"

            # --- UAM ---
            if iob_val > 2 * current_basal or deviation > 6 or uam:
                uam = 1 if deviation > 0 else 0

                if datum_type != "uam":
                    point["uamAbsorption"] = "start"
                datum_type = "uam"
                UAMGlucoseData.append(point)

            else:
                # --- Basal vs ISF ---
                # basalBGI > -4*BGI → basal insulin activity dominates
                if basal_bgi > -4 * bgi:
                    datum_type = "basal"
                    basalGlucoseData.append(point)
                else:
                    # Unexplained rise → basal, not ISF
                    if avg_delta > 0 and avg_delta > -2 * bgi:
                        datum_type = "basal"
                        basalGlucoseData.append(point)
                    else:
                        datum_type = "ISF"
                        ISFGlucoseData.append(point)


    # Post-processing: UAM redistribution 

    csf_len   = len(CSFGlucoseData)
    isf_len   = len(ISFGlucoseData)
    uam_len   = len(UAMGlucoseData)
    basal_len = len(basalGlucoseData)

    if cfg.categorize_uam_as_basal:
        basalGlucoseData = basalGlucoseData + UAMGlucoseData
        UAMGlucoseData   = []

    elif csf_len > 12:
        # ≥ 1h of carb absorption → assume all meals announced
        basalGlucoseData = basalGlucoseData + UAMGlucoseData
        UAMGlucoseData   = []

    else:
        if 2 * basal_len < uam_len:
            basalGlucoseData = basalGlucoseData + UAMGlucoseData
            basalGlucoseData.sort(key=lambda d: float(d["deviation"]))
            basalGlucoseData = basalGlucoseData[: len(basalGlucoseData) // 2]

        if 2 * isf_len < uam_len and isf_len < 10:
            ISFGlucoseData = ISFGlucoseData + UAMGlucoseData
            ISFGlucoseData.sort(key=lambda d: float(d["deviation"]))
            ISFGlucoseData = ISFGlucoseData[: len(ISFGlucoseData) // 2]

    # Re-measure after UAM redistribution
    basal_len = len(basalGlucoseData)
    isf_len   = len(ISFGlucoseData)

    # Too many CSF relative to basal+ISF → promote CSF → ISF
    if 4 * basal_len + isf_len < csf_len and isf_len < 10:
        ISFGlucoseData = ISFGlucoseData + CSFGlucoseData
        CSFGlucoseData = []

    return {
        "CRData":           CRData,
        "CSFGlucoseData":   CSFGlucoseData,
        "ISFGlucoseData":   ISFGlucoseData,
        "basalGlucoseData": basalGlucoseData,
        "UAMGlucoseData":   UAMGlucoseData,
    }




def prepare_for_autotune_isf(
    df: pd.DataFrame,
    *,
    loop_algorithm_input: dict,
    cfg: AutotunePrepConfig = AutotunePrepConfig(),
) -> dict[str, Any]:
    """
    Main entry point.

    1. Calls loop_oref_mapping.prepare_isf_glucose_data to enrich df with
       BGI, avgDelta, deviation, and IOB, and to build the point list.
    2. Runs the oref0-faithful stateful categorizer over those points.
    """
    

    df2, all_points = prepare_isf_glucose_data(
        df,
        loop_algorithm_input=loop_algorithm_input,
        basal=cfg.basal_rate,
        isf=cfg.isf,
        cr=cfg.carb_ratio,
        cgm_col=cfg.cgm_col,
        bgi_col=cfg.bgi_col,
        avg_delta_col=cfg.avg_delta_col,
        deviation_col=cfg.deviation_col,
        include_iob=True,
        include_cob=False,
        iob_col=cfg.iob_col,
        cob_col=cfg.cob_col,
    )

    
    # Sort ascending (oldest → newest) for the stateful loop.
    all_points.sort(key=lambda p: p["date"])

    buckets = _categorize_points(
        all_points,
        loop_algorithm_input=loop_algorithm_input,
        cfg=cfg,
    )

    return {
        "df": df2,
        "isf_glucose_data": buckets["ISFGlucoseData"],
        **buckets,
    }