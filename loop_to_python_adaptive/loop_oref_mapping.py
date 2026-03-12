from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

import pandas as pd

import loop_to_python_adaptive.api as api
from loop_to_python_api.api import get_prediction_values_and_dates

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


#############################
#   COB (carbs on board)    #
#############################
# Option A: not implemented yet.
# NOTE: loop_to_python_api currently exposes single-value COB ("now"), not a series.
# To make COB a series, we either:
#   - implement a simple absorption model in Python, or
#   - call Swift per timestamp (slow), or
#   - extend the Swift API to return a COB time series.
#
# def add_cob_to_history_df(...): ...


##############################
#   IOB (INSULIN ON BOARD)   #
##############################
# Option A: not implemented yet (same reasons as COB).
#
# def add_iob_to_history_df(...): ...


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




def build_isf_glucose_data_from_df(
    df: pd.DataFrame,
    *,
    cgm_col: str = "CGM",
    bgi_col: str = "BGI",
    avg_delta_col: str = "avgDelta",
    deviation_col: str = "deviation",
) -> list[dict]:
    """
    Builds list of dicts with keys "date", "avgDelta", "BGI", "deviation" for ISF tuning.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must have a DatetimeIndex.")
    if cgm_col not in df.columns:
        raise ValueError(f"df missing {cgm_col!r}")
    if bgi_col not in df.columns:
        raise ValueError(f"df missing {bgi_col!r}; call add_bgi_to_history_df first")
    if avg_delta_col not in df.columns:
        raise ValueError(f"df missing {avg_delta_col!r}; call add_avg_delta_to_history_df first")
    if deviation_col not in df.columns:
        raise ValueError(f"df missing {deviation_col!r}; call add_deviation_to_history_df first")

    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")

    avg_delta_s = pd.to_numeric(df[avg_delta_col], errors="coerce")
    bgi_s = pd.to_numeric(df[bgi_col], errors="coerce")
    dev_s = pd.to_numeric(df[deviation_col], errors="coerce")

    pts: list[dict] = []
    for i in range(len(df)):
        if pd.isna(avg_delta_s.iat[i]) or pd.isna(bgi_s.iat[i]) or pd.isna(dev_s.iat[i]):
            continue

        ts = idx[i]
        pts.append(
            {
                "date": ts.isoformat(),
                "avgDelta": float(avg_delta_s.iat[i]),
                "BGI": float(bgi_s.iat[i]),
                "deviation": float(dev_s.iat[i]),
            }
        )

    return pts


def prepare_isf_glucose_data(
    df: pd.DataFrame,
    *,
    loop_algorithm_input: dict,
    cgm_col: str = "CGM",
    bgi_col: str = "BGI",
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Add BGI/avgDelta/deviation to df and prepare points for autotune_prep.
    """
    df2 = add_bgi_to_history_df(df, loop_algorithm_input=loop_algorithm_input, bgi_col=bgi_col, align="ffill",)
    df2 = add_avg_delta_to_history_df(df2, cgm_col=cgm_col, avg_delta_col="avgDelta", window_points=4)
    df2 = add_deviation_to_history_df(df2, avg_delta_col="avgDelta", bgi_col=bgi_col, deviation_col="deviation")

    pts = build_isf_glucose_data_from_df(
        df2,
        cgm_col=cgm_col,
        bgi_col=bgi_col,
        avg_delta_col="avgDelta",
        deviation_col="deviation",
    )
    return df2, pts