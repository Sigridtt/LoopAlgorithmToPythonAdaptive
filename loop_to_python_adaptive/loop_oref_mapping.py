from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

import pandas as pd

import loop_to_python_adaptive.api as api
import loop_to_python_api.helpers as helpers

from loop_to_python_api.api import get_prediction_values_and_dates, get_active_insulin, get_active_carbs
   
AlignMode = Literal["ffill", "nearest", "strict"]

"""
This module maps loop data to fit  oref0 autotune format. 
It takes predictions from Loop Algorithm and computes BGI equivalent
the dataframe returned has bgi, deviation, avgDelta
"""

#Timezone fix
def _to_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize df.index to tz-aware UTC."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must have a DatetimeIndex.")
    out = df.copy()
    out.index = out.index.tz_localize("UTC") if out.index.tz is None else out.index.tz_convert("UTC")
    return out

##########################
#   GENERATE BGI SERIES  #
##########################
@dataclass(frozen=True)
class BGIConfig:
    action_duration_minutes: int
    peak_activity_minutes: int
    step_minutes: int = 5
    history_hours: int = 16
    min_effect_fraction: float = 0.0


def generate_bgi_series_from_insulin_prediction(loop_algorithm_input: dict) -> pd.Series:
    """
    Calls LoopAlgorithm via loop_to_python_api to get insulin-only prediction values+dates,
    then returns BGI(t) = pred(t+5m) - pred(t).
    """
    values, dates = get_prediction_values_and_dates(loop_algorithm_input)

    p_idx = pd.to_datetime(dates, utc=True)
    pred = pd.Series(values, index=p_idx, dtype="float64").sort_index()

    # BGI(t) = pred(t+5m) - pred(t), assigned to time t
    # Typically negative during insulin action because predicted glucose is descending.
    bgi = pred.shift(-1) - pred
    return bgi


def add_bgi_to_history_df(
    df: pd.DataFrame,
    bgi_col: str = "BGI",
    align: AlignMode = "ffill",
    loop_algorithm_input: dict | None = None,
) -> pd.DataFrame:
    """
    Adds a BGI column to the given history dataframe by generating a BGI series from predictions.
    """
    out = _to_utc_index(df)

    if loop_algorithm_input is None:
        loop_algorithm_input = api.get_loop_algorithm_input()

    bgi_pred = generate_bgi_series_from_insulin_prediction(loop_algorithm_input)

    # Aligning timestamps
    if align == "ffill":
        out[bgi_col] = bgi_pred.reindex(out.index, method="ffill")
    elif align == "nearest":
        out[bgi_col] = bgi_pred.reindex(out.index, method="nearest")
    elif align == "strict":
        out[bgi_col] = bgi_pred.reindex(out.index)
    else:
        raise ValueError("align must be one of: 'ffill', 'nearest', 'strict'.")

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

    :param df: DatetimeIndex dataframe with at least 'basal' and 'bolus' columns.
               If missing, basal is filled from the `basal` parameter and bolus
               is set to NaN.
    :param loop_algorithm_input: Full LoopAlgorithm JSON input dict. Used to
                                 read insulinType if present.
    :param basal: Basal rate (U/hr) — used if df has no 'basal' column.
    :param isf: Insulin sensitivity factor (mg/dL per U).
    :param cr: Carbohydrate ratio (g per U).
    :param iob_col: Name of the output column. Default: 'IOB'.
    :param insulin_type: Insulin model to use. Default: 'novolog'.
    :param lookback: Number of rows (5-min intervals) to include in each IOB
                     calculation. Default 72 = 6 hours.
    :return: Copy of df with the IOB column added.
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

    :param df: DatetimeIndex dataframe. Should contain a 'carbs' column with
               meal entries (NaN between meals). If missing, COB will always
               be 0.
    :param loop_algorithm_input: Full LoopAlgorithm JSON input dict. Used to
                                 read insulinType if present.
    :param basal: Basal rate (U/hr).
    :param isf: Insulin sensitivity factor (mg/dL per U).
    :param cr: Carbohydrate ratio (g per U).
    :param cob_col: Name of the output column. Default: 'COB'.
    :param insulin_type: Insulin model to use. Default: 'novolog'.
    :param lookback: Number of rows (5-min intervals) to look back. Default 72 = 6 hours.
    :return: Copy of df with the COB column added.
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
    so we default window_points=4.

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
    """
    out = _to_utc_index(df)
    if avg_delta_col not in out.columns:
        raise ValueError(f"df missing {avg_delta_col!r}; compute avgDelta first")
    if bgi_col not in out.columns:
        raise ValueError(f"df missing {bgi_col!r}; compute BGI first")

    out[deviation_col] = pd.to_numeric(out[avg_delta_col], errors="coerce") - pd.to_numeric(out[bgi_col], errors="coerce")
    return out



###########################
#    PREP For PIPELINE    #
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
    Builds list of dicts with keys "date", "avgDelta", "BGI", "deviation" for ISF tuning.
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
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Add BGI/avgDelta/deviation to df and prepare points for autotune_prep.
    """
    df2 = add_bgi_to_history_df(df, loop_algorithm_input=loop_algorithm_input, bgi_col=bgi_col, align="ffill",)
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