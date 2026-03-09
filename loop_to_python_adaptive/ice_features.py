from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from . import api


@dataclass(frozen=True)
class ICEFeatureConfig:
    insulin_type: str = "novolog"
    batch_size: int = 300
    overlap: int = 72  # 72 * 5 min = 6 hours; matches upstream default


def add_ice_to_history_df(
    df: pd.DataFrame,
    *,
    basal_scheduled_u_per_hr: float,
    isf_mgdl_per_u: float,
    cr_g_per_u: float,
    config: Optional[ICEFeatureConfig] = None,
) -> pd.DataFrame:
    """
    Compute insulin counteraction effect (ICE) time series using the Loop wrapper
    and add it to a copy of df as df['ice'].

    Expected df format (5-min index recommended):
      - DatetimeIndex
      - columns required by upstream:
          'CGM' (mg/dL)
          'basal' (U/hr)   (delivered basal rate per step)
          'bolus' (U)      (bolus delivered at that timestamp; 0 otherwise)
      - optional:
          'carbs' (g) (not needed for ICE computation itself)

    Returns:
      A copy of df with 'ice' column (float). Some leading rows may be NaN due
      to overlap/warmup requirements.
    """
    cfg = config or ICEFeatureConfig()

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must have a DatetimeIndex (timestamps).")
    required = {"CGM", "basal", "bolus"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df missing required columns: {sorted(missing)}")

    # Upstream helper mutates the df in-place, so operate on a copy.
    out = df.copy()

    # This is provided by loop_to_python_api.api (re-exported via our api.py)
    # It writes an 'ice' column aligned to timestamps.
    out = api.add_insulin_counteraction_effect_to_df(
        out,
        basal_scheduled_u_per_hr,
        isf_mgdl_per_u,
        cr_g_per_u,
        insulin_type=cfg.insulin_type,
        batch_size=cfg.batch_size,
        overlap=cfg.overlap,
    )
    return out


def ensure_ice_column(
    df: pd.DataFrame,
    *,
    basal_scheduled_u_per_hr: float,
    isf_mgdl_per_u: float,
    cr_g_per_u: float,
    config: Optional[ICEFeatureConfig] = None,
) -> pd.DataFrame:
    """
    Convenience: if df already has non-null 'ice' values, return df unchanged,
    otherwise compute ICE and return df with 'ice'.
    """
    if "ice" in df.columns and df["ice"].notna().any():
        return df
    return add_ice_to_history_df(
        df,
        basal_scheduled_u_per_hr=basal_scheduled_u_per_hr,
        isf_mgdl_per_u=isf_mgdl_per_u,
        cr_g_per_u=cr_g_per_u,
        config=config,
    )
