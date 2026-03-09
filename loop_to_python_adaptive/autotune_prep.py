from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class AutotuneISFConfig:
    min_points: int = 10
    adjustment_fraction: float = 0.2  # oref0 uses 20%
    autosens_max: float = 1.2         # oref0 default
    autosens_min: float = 0.7         # oref0 default
    min_bgi_abs: float = 1e-6         # avoid divide by tiny numbers
    ratio_clip_low: float = 0.3       # hard clip to avoid insane outliers
    ratio_clip_high: float = 3.0


def tune_isf_like_oref0(
    *,
    isf_current: float,
    isf_glucose_data: list[dict[str, Any]],
    pump_isf: Optional[float] = None,
    config: AutotuneISFConfig | None = None,
) -> dict[str, Any]:
    """
    oref0-like ISF tuning:
      ratio = 1 + deviation / BGI
      fullNewISF = ISF * median(ratio)
      newISF = (1-adjustment_fraction)*ISF + adjustment_fraction*adjustedISF
    """
    cfg = config or AutotuneISFConfig()

    ratios: list[float] = []
    for p in isf_glucose_data:
        dev = float(p["deviation"])
        bgi = float(p["BGI"])
        if abs(bgi) < cfg.min_bgi_abs:
            continue
        r = 1.0 + dev / bgi
        if not np.isfinite(r):
            continue
        r = float(np.clip(r, cfg.ratio_clip_low, cfg.ratio_clip_high))
        ratios.append(r)

    if len(ratios) < cfg.min_points:
        return {
            "newISF": float(isf_current),
            "p50_ratio": None,
            "fullNewISF": None,
            "adjustedISF": None,
            "n_points": len(ratios),
            "reason": f"Only {len(ratios)} usable ISF points (<{cfg.min_points}); leaving ISF unchanged.",
        }

    p50_ratio = float(np.median(ratios))
    full_new_isf = float(isf_current * p50_ratio)

    adjusted_isf = full_new_isf
    if pump_isf is not None and pump_isf > 0:
        # Match oref0 bounds:
        # low autosens ratio = high ISF => maxISF = pumpISF / autosens_min
        # high autosens ratio = low ISF => minISF = pumpISF / autosens_max
        max_isf = pump_isf / cfg.autosens_min
        min_isf = pump_isf / cfg.autosens_max
        adjusted_isf = float(np.clip(adjusted_isf, min_isf, max_isf))

    # Slow update like oref0
    new_isf = (1.0 - cfg.adjustment_fraction) * float(isf_current) + cfg.adjustment_fraction * float(adjusted_isf)

    return {
        "newISF": float(new_isf),
        "p50_ratio": p50_ratio,
        "fullNewISF": full_new_isf,
        "adjustedISF": float(adjusted_isf),
        "n_points": len(ratios),
        "reason": "OK",
    }