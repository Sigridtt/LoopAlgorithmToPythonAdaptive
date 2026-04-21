"""
This module maps loop data to fit  oref0 autotune format. 
It takes predictions from Loop Algorithm and computes BGI equivalent
the dataframe returned has bgi, deviation, avgDelta

"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import logging

import pandas as pd

import loop_to_python_adaptive.api as api
import loop_to_python_api.helpers as helpers

from loop_to_python_api.api import get_prediction_values_and_dates, get_active_insulin, get_active_carbs, insulin_percent_effect_remaining
   
AlignMode = Literal["ffill", "nearest", "strict"]
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#   Insulin model parameters
#
#   Source: LoopKit ExponentialInsulinModel defaults
#   These match the values used internally by loop_to_python_api when
#   insulin_type is passed to get_active_insulin / get_prediction_values_and_dates.
#   loop_to_python_api does not expose these parameters — they are used
#   inside the Swift layer only. They are mirrored here so we can call
#   insulin_percent_effect_remaining with the correct values for each type.
#
#   Reference: ExponentialInsulinModel.swift (https://github.com/tidepool-org/LoopAlgorithm/blob/main/Sources/LoopAlgorithm/Insulin/ExponentialInsulinModel.swift)
#   action_duration and peak_activity_time in minutes, delay in minutes.
# ---------------------------------------------------------------------------

INSULIN_MODEL_PARAMS: dict[str, dict] = {
    "novolog":   {"action_duration": 360, "peak_activity_time": 75, "delay": 10},
    "humalog":   {"action_duration": 360, "peak_activity_time": 75, "delay": 10},
    "apidra":    {"action_duration": 360, "peak_activity_time": 75, "delay": 10},
    "fiasp":     {"action_duration": 360, "peak_activity_time": 55, "delay": 10},
    "lyumjev":   {"action_duration": 360, "peak_activity_time": 45, "delay": 10},
    "afrezza":   {"action_duration": 300, "peak_activity_time": 29, "delay": 10},
}

#Timezone fix
def _to_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize df.index to tz-aware UTC."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must have a DatetimeIndex.")
    out = df.copy()
    out.index = out.index.tz_localize("UTC") if out.index.tz is None else out.index.tz_convert("UTC")
    return out

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


##########################
#   GENERATE BGI SERIES  #
##########################
def _get_model_params(insulin_type: str) -> dict:
    """Return model params for the given insulin type, defaulting to novolog."""
    return INSULIN_MODEL_PARAMS.get(
        insulin_type.lower(),
        INSULIN_MODEL_PARAMS["novolog"],
    )

def insulin_activity_at(
    mins_ago: float,
    action_duration: float,
    peak_activity_time: float,
    delay: float,
    dt: float = 0.5,
) -> float:
    """
    Instantaneous insulin activity (fraction of dose / min) at `mins_ago`
    minutes after delivery.

    Computed as the central-difference numerical derivative of
    percentEffectRemaining, using step dt minutes.

    activity = -d/dt[percentEffectRemaining(t)]

    dt=0.5 min is small enough for accuracy on the exponential curve
    and large enough to avoid floating-point noise?

    Returns U_fraction/min — multiply by dose (U) to get U/min.
    """
    per_before = insulin_percent_effect_remaining(
        mins_ago - dt, action_duration, peak_activity_time, delay
    )
    per_after = insulin_percent_effect_remaining(
        mins_ago + dt, action_duration, peak_activity_time, delay
    )
    # Negative because percentEffectRemaining decreases over time
    return -(per_after - per_before) / (2 * dt)


'''                                  
@dataclass(frozen=True)
class BGIConfig:
    action_duration_minutes: int
    peak_activity_minutes: int
    step_minutes: int = 5
    history_hours: int = 16
    min_effect_fraction: float = 0.0


def generate_bgi_series_from_insulin_prediction(loop_algorithm_input: dict) -> pd.Series:
    
    #Calls LoopAlgorithm via loop_to_python_api to get insulin-only prediction values+dates,
    #then returns BGI(t) = pred(t+5m) - pred(t).
    
    values, dates = get_prediction_values_and_dates(loop_algorithm_input)

    p_idx = pd.to_datetime(dates, utc=True)
    pred = pd.Series(values, index=p_idx, dtype="float64").sort_index()

    # BGI(t) = pred(t+5m) - pred(t), assigned to time t
    # Typically negative during insulin action because predicted glucose is descending.
    bgi = pred.shift(-1) - pred
    return bgi
'''
def generate_bgi_series_from_predictions(
    df: pd.DataFrame,
    *,
    json_history: list[dict],
) -> pd.Series:
    """
    Compute BGI(t) = pred(t+5m) - pred(t) using Loop's own predictions.

    Each json snapshot in json_history was built at a specific prediction_start.
    We call get_prediction_values_and_dates for each snapshot and extract the
    BGI at the prediction_start timestamp only — this gives us one reliable
    BGI value per snapshot that is internally consistent with what Loop computed.

    The resulting sparse series is then reindexed onto df's index using
    forward-fill, which is appropriate because the Loop prediction is valid
    for the 5-min window starting at prediction_start.

    This is the correct BGI method because:
      - It uses exactly the same insulin model and dose history that Loop used
      - It avoids the mismatch between our Python activity model and Loop's
        internal Swift model
      - It is consistent with oref0's intent (BGI = expected BG change per 5min
        from insulin alone)
    """
    # In generate_bgi_series_from_predictions, add this as the first line after `out = _to_utc_index(df)`:
    
    out = _to_utc_index(df)
    if not json_history:
        return pd.Series(float("nan"), index=out.index, dtype="float64")

    bgi_points: dict[pd.Timestamp, float] = {}

    for json_input in json_history:
        try:
            values, dates = get_prediction_values_and_dates(json_input)
        except Exception:
            continue

        if not values or not dates:
            continue

        p_idx = pd.to_datetime(dates, utc=True)
        pred  = pd.Series(values, index=p_idx, dtype="float64").sort_index()

        # BGI at each prediction point = pred(t+5m) - pred(t)
        bgi_series = pred.shift(-1) - pred

        # Extract only the value at the prediction_start (first point),
        # which is the BGI that was valid when this snapshot was taken
        if len(bgi_series) >= 1:
            ts  = bgi_series.index[0]
            val = bgi_series.iloc[0]
            if not pd.isna(val):
                bgi_points[ts] = val

    if not bgi_points:
        # Fallback: return NaN series
        return pd.Series(float("nan"), index=out.index, dtype="float64")

    sparse = pd.Series(bgi_points, dtype="float64").sort_index()

    # Forward-fill onto df index so every CGM row gets a BGI value
    combined = sparse.reindex(
        sparse.index.union(out.index)
    ).ffill().reindex(out.index)

    return combined




'''
def generate_bgi_series_from_activity(
    df: pd.DataFrame,
    *,
    loop_algorithm_input: dict,
    isf: float,
    insulin_type: str = "novolog",
) -> pd.Series:
    """
    Compute BGI(t) as oref0's categorize.js:

        BGI = -iob.activity * sens * 5

    where iob.activity = sum of activityContrib across all doses (U/min).

    Key properties vs alternatives:
      - No future CGM data needed (pure insulin model, analytical)
      - No nonlinearity error (exact derivative, not finite IOB difference)
      - Covers the full CGM history window (not limited to prediction horizon)
      - insulin_type resolved from loop_algorithm_input["insulinType"] if present

    """
    resolved_type = loop_algorithm_input.get("insulinType", insulin_type).lower()
    params = _get_model_params(resolved_type)

    action_duration    = params["action_duration"]
    peak_activity_time = params["peak_activity_time"]
    delay              = params["delay"]

    doses = loop_algorithm_input.get("doses", []) or []

    out = _to_utc_index(df)
    bgis: list[float] = []

    for ts in out.index:
        total_activity = 0.0  # U/min

        for dose in doses:
            dose_type = dose.get("type", "")
            if dose_type not in ("bolus", "basal"):
                continue

            volume = float(dose.get("volume", 0) or 0)
            if volume <= 0:
                continue

            dose_time = pd.to_datetime(dose["startDate"], utc=True)
            mins_ago  = (ts - dose_time).total_seconds() / 60.0

            # Only doses within the insulin action window
            if mins_ago < 0 or mins_ago > action_duration + delay:
                continue

            activity = insulin_activity_at(
                mins_ago, action_duration, peak_activity_time, delay
            )  # fraction/min
            total_activity += activity * volume  # U/min

        # BGI = -activity * ISF * 5  →  mg/dL per 5 min
        bgi = -total_activity * isf * 5
        bgis.append(round(bgi, 3))

    return pd.Series(bgis, index=out.index, dtype="float64")

'''

def add_bgi_to_history_df(
    df: pd.DataFrame,
    isf: float,
    bgi_col: str = "BGI",
    align: AlignMode = "ffill",
    loop_algorithm_input: dict | None = None,
    json_history: list[dict] | None = None,
) -> pd.DataFrame:
    """
    Adds a BGI column to the given history dataframe by generating a BGI series from predictions.
    """
    out = _to_utc_index(df)

    if loop_algorithm_input is None:
        loop_algorithm_input = api.get_loop_algorithm_input()
    insulin_type = loop_algorithm_input.get("insulinType", "novolog")

    raw_bgi = generate_bgi_series_from_predictions(
            out,
            json_history=json_history,)
    out[bgi_col] = pd.to_numeric(raw_bgi, errors="coerce")

    #if (out[bgi_col] > 0).sum() > (out[bgi_col] < 0).sum():
        #logger.warning("BGI mostly positive — possible sign mismatch with Loop predictions")
  

    # Aligning timestamps needed for bgi from predictions as they are "in the future"
    # if align == "ffill":
    #     out[bgi_col] = bgi_pred.reindex(out.index, method="ffill")
    # elif align == "nearest":
    #     out[bgi_col] = bgi_pred.reindex(out.index, method="nearest")
    # elif align == "strict":
    #     out[bgi_col] = bgi_pred.reindex(out.index)
    # else:
    #     raise ValueError("align must be one of: 'ffill', 'nearest', 'strict'.")
    
    #bgi using activity model
    # out[bgi_col] = generate_bgi_series_from_activity(
    #     out,
    #     loop_algorithm_input=loop_algorithm_input,
    #     isf=isf,
    #     insulin_type=insulin_type,
    # )
    return out


##############################
#   IOB (INSULIN ON BOARD)   #
##############################

def add_iob_to_history_df(
    df: pd.DataFrame,
    *,
    loop_algorithm_input: dict,
    basal: float,
    isf: float,
    cr: float,
    iob_col: str = "IOB",
    insulin_type: str = "novolog",
    lookback: int = 72,
) -> pd.DataFrame:
    """
    Adds an IOB (insulin on board, in U) column to the history dataframe.

    Each row's IOB is computed from the `lookback` preceding rows using
    get_active_insulin from loop_to_python_api. The insulin_type is read
    from loop_algorithm_input if present, otherwise falls back to the
    `insulin_type` parameter.

    :param lookback: Number of rows (5-min intervals) to include in each IOB
                     calculation. Default 72 = 6 hours.
    """

    resolved_insulin_type = loop_algorithm_input.get("insulinType", insulin_type)

    out = _to_utc_index(df)
    if "basal" not in out.columns:
        out["basal"] = basal
    if "bolus" not in out.columns:
        out["bolus"] = float("nan")

    iobs: list[float] = []
    for i, ts in enumerate(out.index):
        start_i   = max(0, i - lookback + 1)
        sub       = out.iloc[start_i : i + 1]
        json_data = helpers.get_json_loop_prediction_input_from_df(
            sub, basal, isf, cr, ts, insulin_type=resolved_insulin_type
        )
        iobs.append(float(get_active_insulin(json_data)))

    out[iob_col] = iobs
    return out


#############################
#   COB (carbs on board)    #
#############################

def add_cob_to_history_df(
    df: pd.DataFrame,
    *,
    loop_algorithm_input: dict,
    basal: float,
    isf: float,
    cr: float,
    cob_col: str = "COB",
    insulin_type: str = "novolog",
    lookback: int = 72,
) -> pd.DataFrame:
    """
    Adds a COB (carbs on board, in g) column to the history dataframe.

    Each row's COB is computed from the `lookback` preceding rows using
    get_active_carbs from loop_to_python_api.

    :param lookback: Number of rows (5-min intervals) to look back. Default 72 = 6 hours.
  
    """

    resolved_insulin_type = loop_algorithm_input.get("insulinType", insulin_type)

    out = _to_utc_index(df)
    if "basal" not in out.columns:
        out["basal"] = basal
    if "bolus" not in out.columns:
        out["bolus"] = float("nan")

    cobs: list[float] = []
    for i, ts in enumerate(out.index):
        start_i   = max(0, i - lookback + 1)
        sub       = out.iloc[start_i : i + 1]
        json_data = helpers.get_json_loop_prediction_input_from_df(
            sub, basal, isf, cr, ts, insulin_type=resolved_insulin_type
        )
        cobs.append(float(get_active_carbs(json_data)))

    out[cob_col] = cobs
    return out

##################
#    avgDelta    #
##################
def add_avg_delta_to_history_df(
    df: pd.DataFrame,
    *,
    cgm_col: str = "CGM",
    avg_delta_col: str = "avgDelta",
    window_points: int = 4,
) -> pd.DataFrame:
    """
    Adds avgDelta as a recent-past slope estimate.

    oref0 computes avgDelta over the last 4 CGM datapoints (i.e., ~15 minutes of history), 
    so default window_points=4.

    Units: mg/dL per 5 minutes.
    """
    out = _to_utc_index(df)
    if cgm_col not in out.columns:
        raise ValueError(f"df missing {cgm_col!r}")

    cgm = pd.to_numeric(out[cgm_col], errors="coerce")

    # avgDelta(t) = mean of the last 4 per-5m deltas ending at time t (oref0-style)
    out[avg_delta_col] = cgm.diff().rolling(window_points, min_periods=window_points).mean()

    return out


###################
#    deviation    #
###################
def add_deviation_to_history_df(
    df: pd.DataFrame,
    *,
    avg_delta_col: str = "avgDelta",
    bgi_col: str = "BGI",
    deviation_col: str = "deviation",
) -> pd.DataFrame:
    """
    Adds deviation = avgDelta - BGI as a column.
    deviation > 0 : BG rising more than insulin predicts (carbs / UAM)
    deviation < 0 : BG falling more than insulin predicts (ISF too weak)
    deviation ≈ 0 : insulin model explains BG movement well
    """
    out = _to_utc_index(df)
    if avg_delta_col not in out.columns:
        raise ValueError(f"df missing {avg_delta_col!r}; compute avgDelta first")
    if bgi_col not in out.columns:
        raise ValueError(f"df missing {bgi_col!r}; compute BGI first")

    out[deviation_col] = pd.to_numeric(out[avg_delta_col], errors="coerce") - pd.to_numeric(out[bgi_col], errors="coerce")
    return out



###########################
#    BUILDING FOR PREP    #
###########################


def build_isf_glucose_data_from_df(
    df: pd.DataFrame,
    *,
    cgm_col: str = "CGM",
    bgi_col: str = "BGI",
    avg_delta_col: str = "avgDelta",
    deviation_col: str = "deviation",
    iob_col: str = "IOB",        # passed through if present
    cob_col: str = "COB",        # passed through if present
) -> list[dict]:
    """
    Converts the enriched df into a list of point dicts for autotune_prep.

    Points where avgDelta, BGI, or deviation are NaN are skipped —
    these are the warmup rows at the start of the window where
    avgDelta needs 4 prior points to be valid.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must have a DatetimeIndex.")
    for col in [cgm_col, bgi_col, avg_delta_col, deviation_col]:
        if col not in df.columns:
            raise ValueError(f"df missing {col!r}")

    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")

    avg_delta_s = pd.to_numeric(df[avg_delta_col], errors="coerce")
    bgi_s = pd.to_numeric(df[bgi_col], errors="coerce")
    dev_s = pd.to_numeric(df[deviation_col], errors="coerce")
    cgm_s = pd.to_numeric(df[cgm_col], errors="coerce")
    iob_s = pd.to_numeric(df[iob_col], errors="coerce") if iob_col in df.columns else None
    cob_s = pd.to_numeric(df[cob_col], errors="coerce") if cob_col in df.columns else None

    pts: list[dict] = []
    for i in range(len(df)):
        if pd.isna(avg_delta_s.iat[i]) or pd.isna(bgi_s.iat[i]) or pd.isna(dev_s.iat[i]):
            continue

        pt: dict = {
            "date":     idx[i].isoformat(),
            "glucose":  float(cgm_s.iat[i]) if not pd.isna(cgm_s.iat[i]) else None,
            "avgDelta": float(avg_delta_s.iat[i]),
            "BGI":      float(bgi_s.iat[i]),
            "deviation":float(dev_s.iat[i]),
        }
        if iob_s is not None and not pd.isna(iob_s.iat[i]):
            pt["IOB"] = float(iob_s.iat[i])
        if cob_s is not None and not pd.isna(cob_s.iat[i]):
            pt["COB"] = float(cob_s.iat[i])

        pts.append(pt)

    return pts


def prepare_isf_glucose_data(
    df: pd.DataFrame,
    *,
    loop_algorithm_input: dict,
    basal: float,
    isf: float,
    cr: float,
    cgm_col: str = "CGM",
    bgi_col: str = "BGI",
    avg_delta_col: str = "avgDelta",      
    deviation_col: str = "deviation",     
    include_iob: bool = True,   
    include_cob: bool = False,  # <-- (COB not needed by categorizer, but useful for debugging)
    iob_col: str = "IOB",
    cob_col: str = "COB",
    json_history: list[dict] | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Add BGI/avgDelta/deviation to df and prepare points for autotune_prep.

    Call order and reasoning:
      1. BGI       — analytical, from insulin model, no other columns needed
      2. avgDelta  — from CGM only 
      3. deviation — requires BGI and avgDelta
      4. IOB       — independent; needed by categoriser later, not by BGI
      5. COB       — optional, for debugging only

    """
    df2 = add_bgi_to_history_df(df, isf, bgi_col=bgi_col, align="ffill",loop_algorithm_input=loop_algorithm_input,json_history=json_history,)
    df2 = add_avg_delta_to_history_df(df2, cgm_col=cgm_col, avg_delta_col="avgDelta", window_points=4)
    df2 = add_deviation_to_history_df(df2, avg_delta_col="avgDelta", bgi_col=bgi_col, deviation_col="deviation")

    if include_iob:
        df2 = add_iob_to_history_df(df2, loop_algorithm_input=loop_algorithm_input,
                                    basal=basal, isf=isf, cr=cr, iob_col=iob_col)
    if include_cob:
        df2 = add_cob_to_history_df(df2, loop_algorithm_input=loop_algorithm_input,
                                    basal=basal, isf=isf, cr=cr, cob_col=cob_col)

    pts = build_isf_glucose_data_from_df(
        df2,
        cgm_col=cgm_col,
        bgi_col=bgi_col,
        avg_delta_col= avg_delta_col,
        deviation_col=deviation_col,
        iob_col=iob_col,  
        cob_col=cob_col,   
    )
    return df2, pts