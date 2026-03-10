from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from loop_to_python_adaptive.loop_bgi import BGIConfig, prepare_isf_glucose_data


@dataclass(frozen=True)
class AutotunePrepConfig:
    """
    Configuration for preparing inputs to autotune_isf.tune_isf_like_oref0.
    """
    action_duration_minutes: int = 360
    peak_activity_minutes: int = 75
    delay_minutes: int = 10
    history_hours: int = 16
    step_minutes: int = 5

    cgm_col: str = "CGM"
    bgi_col: str = "BGI"


def prepare_for_autotune_isf(
    df: pd.DataFrame,
    *,
    loop_algorithm_input: dict,
    config: Optional[AutotunePrepConfig] = None,
) -> dict[str, Any]:
    """
    Prepare the exact inputs autotune_isf.tune_isf_like_oref0 needs.

    Returns:
      {
        "df": df_with_BGI,
        "isf_glucose_data": list[{"date","avgDelta","BGI","deviation"}...]
      }
    """
    cfg = config or AutotunePrepConfig()

    bgi_cfg = BGIConfig(
        action_duration_minutes=cfg.action_duration_minutes,
        peak_activity_minutes=cfg.peak_activity_minutes,
        delay_minutes=cfg.delay_minutes,
        step_minutes=cfg.step_minutes,
        history_hours=cfg.history_hours,
    )

    df2, isf_glucose_data = prepare_isf_glucose_data(
        df,
        loop_algorithm_input=loop_algorithm_input,
        config=bgi_cfg,
        cgm_col=cfg.cgm_col,
        bgi_col=cfg.bgi_col,
    )

    return {"df": df2, "isf_glucose_data": isf_glucose_data}