"""
autosens.py — Rolling insulin sensitivity ratio (autosens) for AdaptiveLoopController.

oref0 reference: lib/autosens.js
https://github.com/openaps/oref0/blob/master/lib/autosens.js

────────────────────────────────────────────────────────────────────────────────
RELATIONSHIP TO oref0
────────────────────────────────────────────────────────────────────────────────

In oref0, autosens runs every 5 minutes alongside the main loop. It looks at the
last 8 hours (and separately 24 hours) of CGM data and asks:

    "Has BG been moving more or less than insulin alone predicts?"

It answers this with a single number — the autosens ratio — computed as the
median of per-point sensitivity ratios:

    ratio(t) = deviation(t) / BGI(t)          [oref0: lib/autosens.js line ~120]

where:
    deviation = avgDelta - BGI
    BGI       = expected BG change per 5 min from insulin alone (negative during action)

Substituting:
    ratio(t) = (avgDelta - BGI) / BGI = avgDelta/BGI - 1

A ratio of 0.0  means BG moved exactly as insulin predicted  → sensitivity unchanged
A ratio of +0.2 means BG moved 20% more than predicted       → patient is more sensitive
A ratio of -0.2 means BG moved 20% less than predicted       → patient is more resistant

oref0 then computes:
    autosens_ratio = 1 + median(ratios)

and clips it to [autosens_min, autosens_max] = [0.7, 1.2] by default.

The ratio is applied temporarily to the current ISF and basal for the next dose:
    effective_isf   = pump_isf   / autosens_ratio   (higher ratio → lower ISF → more insulin)
    effective_basal = pump_basal * autosens_ratio

Note the asymmetry:
    - ISF is DIVIDED   by the ratio (more sensitive → lower ISF → smaller correction bolus)
    - Basal is multiplied by the ratio (more sensitive → higher basal to cover background need)

This is different from autotune:
    - autotune  permanently updates the pump ISF over days
    - autosens  temporarily scales the current effective ISF every 5 minutes

────────────────────────────────────────────────────────────────────────────────
KEY DESIGN DECISIONS vs oref0
────────────────────────────────────────────────────────────────────────────────

1. oref0 excludes points where COB > 0 or UAM is active from the autosens
   calculation, to avoid meal noise contaminating the ratio. We mirror this
   by accepting a list of deviation/BGI pairs that have already been filtered
   by the caller (AdaptiveLoopController) to exclude meal periods.
   The caller uses a simple heuristic: skip points where COB > 0 or
   |deviation| > deviation_threshold (default 6 mg/dL/5min, matching oref0).

2. oref0 computes autosens over both 8h and 24h windows and takes the more
   conservative (closer to 1.0) of the two. We do the same.

3. oref0 skips the autosens update if fewer than min_points valid ratios are
   available. We use min_points=10 as the default, matching oref0.

4. The autosens ratio is NOT applied to the pump ISF directly — it scales
   the autotune ISF (which may already differ from pump ISF). This gives a
   two-layer architecture:
       effective_isf = autotune_isf / autosens_ratio
   where autotune_isf is the daily-updated baseline and autosens_ratio is
   the short-term correction.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#   Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AutosensConfig:
    """
    Parameters controlling the autosens computation.

    Corresponds to oref0 profile fields:
        autosens_max     → profile.autosens.max   (default 1.2)
        autosens_min     → profile.autosens.min   (default 0.7)
        window_8h_points → 8 * 60 / 5 = 96 CGM points
        window_24h_points→ 24 * 60 / 5 = 288 CGM points
        min_points       → minimum valid ratios needed (oref0 uses ~10)
        deviation_threshold → exclude points where |deviation| > this value
                              (oref0 uses 6 mg/dL/5min as the UAM threshold)
        min_bgi_abs      → skip points where |BGI| is too small to divide by
    """
    autosens_max: float        = 1.2    # oref0 default: profile.autosens.max
    autosens_min: float        = 0.7    # oref0 default: profile.autosens.min
    window_8h_points: int      = 96     # 8h × 60min / 5min
    window_24h_points: int     = 288    # 24h × 60min / 5min
    min_points: int            = 10     # minimum valid ratios before updating
    deviation_threshold: float = 6.0   # mg/dL/5min — exclude UAM/meal spikes
    min_bgi_abs: float         = 1e-6  # avoid division by near-zero BGI


# ─────────────────────────────────────────────────────────────────────────────
#   Per-point data record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AutosensPoint:
    """
    One CGM data point as seen by autosens.

    oref0 autosens.js collects these fields for each 5-min bucket:
        deviation : avgDelta - BGI   (mg/dL per 5 min)
        bgi       : expected BG change from insulin (mg/dL per 5 min, usually negative)
        cob       : carbs on board at this time (g)  — used to exclude meal periods
        glucose   : raw CGM value (mg/dL)            — used for the BG<80 rule

    All of these are already computed by loop_oref_mapping.py and stored
    in the per-step json/log by AdaptiveLoopController, so no extra work
    is needed to populate AutosensPoint.
    """
    deviation: float
    bgi:       float
    cob:       float  = 0.0    # 0 if not tracked
    glucose:   float  = 100.0  # fallback if not available


# ─────────────────────────────────────────────────────────────────────────────
#   Rolling buffer
# ─────────────────────────────────────────────────────────────────────────────

class AutosensBuffer:
    """
    A fixed-length rolling buffer of AutosensPoints.

    oref0 autosens.js maintains the last 24 hours of data (288 points at
    5-min resolution) and computes autosens over two sub-windows: 8h and 24h.

    We use a deque with maxlen=window_24h_points so old data is automatically
    discarded. The caller (AdaptiveLoopController) appends one point per
    5-min CGM step via push().

    oref0 reference: lib/autosens.js — the outer loop over glucose history
    """

    def __init__(self, cfg: AutosensConfig):
        self.cfg = cfg
        # maxlen ensures oldest points are dropped automatically
        self._buffer: deque[AutosensPoint] = deque(maxlen=cfg.window_24h_points)

    def push(self, point: AutosensPoint) -> None:
        """
        Append one new CGM step to the buffer.

        Called every 5 minutes by AdaptiveLoopController._loop_policy(),
        mirroring oref0's per-step autosens data collection.
        """
        self._buffer.append(point)

    def points_8h(self) -> list[AutosensPoint]:
        """Return the most recent 8 hours of points (up to 96)."""
        n = min(len(self._buffer), self.cfg.window_8h_points)
        return list(self._buffer)[-n:]

    def points_24h(self) -> list[AutosensPoint]:
        """Return up to 24 hours of points (up to 288)."""
        return list(self._buffer)


# ─────────────────────────────────────────────────────────────────────────────
#   Core ratio computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_ratio_from_points(
    points: list[AutosensPoint],
    cfg: AutosensConfig,
) -> Optional[float]:
    """
    Compute the autosens ratio from a list of AutosensPoints.

    oref0 lib/autosens.js (line ~115):

        ratio = deviation / BGI

    A positive ratio means BG rose more than insulin predicted (more sensitive).
    A negative ratio means BG fell less than predicted (more resistant).

    oref0 then computes:
        autosens_ratio = 1 + median(ratios)

    and clips to [autosens_min, autosens_max].

    oref0 excludes points where:
        - COB > 0                   (carbs actively absorbing — meal noise)
        - |deviation| > 6          (UAM spike — unannounced meal)
        - glucose < 80 and deviation > 0  (low-BG rule: positive deviations
                                           at low BG are suppressed, matching
                                           autotune_prep's same rule)
        - |BGI| is too small to divide by reliably

    Returns None if fewer than min_points valid ratios are available,
    which tells the caller to leave the ratio unchanged (oref0 behaviour).
    """
    ratios: list[float] = []

    for p in points:
        # Exclude meal / UAM periods — same logic as oref0 autosens.js
        if p.cob > 0:
            continue
        if abs(p.deviation) > cfg.deviation_threshold:
            continue

        # Low-BG rule: oref0 zeroes positive deviations below 80 mg/dL
        # (lib/autosens.js mirrors autotune's categorize.js rule)
        deviation = p.deviation
        if p.glucose < 80 and deviation > 0:
            deviation = 0.0

        # Skip near-zero BGI to avoid numerical instability
        if abs(p.bgi) < cfg.min_bgi_abs:
            continue

        ratio = deviation / p.bgi
        if not np.isfinite(ratio):
            continue

        ratios.append(ratio)

    if len(ratios) < cfg.min_points:
        return None  # not enough data — leave ratio unchanged

    # oref0: autosens_ratio = 1 + median(ratios)
    return 1.0 + float(np.median(ratios))


# ─────────────────────────────────────────────────────────────────────────────
#   Main compute function
# ─────────────────────────────────────────────────────────────────────────────

def compute_autosens(
    buffer: AutosensBuffer,
    cfg:    AutosensConfig,
) -> dict:
    """
    Compute the autosens ratio from the rolling buffer.

    oref0 lib/autosens.js computes two candidate ratios:
        ratio_8h  — from the last 8 hours of data
        ratio_24h — from the last 24 hours of data

    and returns the one that is CLOSER TO 1.0 (i.e. more conservative).
    This prevents a single unusual period from driving a large correction.

    oref0 reference (lib/autosens.js line ~200):
        "use the smaller of the two autosens values"

    Both are clipped to [autosens_min, autosens_max] before comparison.

    Returns a dict with:
        ratio        : the final autosens ratio to apply  (1.0 = no change)
        ratio_8h     : raw 8h candidate (before conservatism selection)
        ratio_24h    : raw 24h candidate (before conservatism selection)
        n_points_8h  : number of valid ratio points in 8h window
        n_points_24h : number of valid ratio points in 24h window
        reason       : human-readable status string
    """
    ratio_8h  = _compute_ratio_from_points(buffer.points_8h(),  cfg)
    ratio_24h = _compute_ratio_from_points(buffer.points_24h(), cfg)

    def _clip(r: float) -> float:
        """Clip ratio to [autosens_min, autosens_max]."""
        return max(cfg.autosens_min, min(cfg.autosens_max, r))

    # Count valid points for diagnostics
    n_8h  = len([p for p in buffer.points_8h()
                 if p.cob == 0 and abs(p.deviation) <= cfg.deviation_threshold
                 and abs(p.bgi) >= cfg.min_bgi_abs])
    n_24h = len([p for p in buffer.points_24h()
                 if p.cob == 0 and abs(p.deviation) <= cfg.deviation_threshold
                 and abs(p.bgi) >= cfg.min_bgi_abs])

    # Neither window has enough data
    if ratio_8h is None and ratio_24h is None:
        return {
            "ratio":        1.0,
            "ratio_8h":     None,
            "ratio_24h":    None,
            "n_points_8h":  n_8h,
            "n_points_24h": n_24h,
            "reason":       f"Insufficient data (8h={n_8h}, 24h={n_24h} points); ratio=1.0",
        }

    # Clip available ratios
    clipped_8h  = _clip(ratio_8h)  if ratio_8h  is not None else None
    clipped_24h = _clip(ratio_24h) if ratio_24h is not None else None

    # oref0: pick the more conservative (closer to 1.0) of the two
    # If only one is available, use that one
    if clipped_8h is None:
        final_ratio = clipped_24h
        reason = f"Only 24h window valid; ratio={final_ratio:.3f}"
    elif clipped_24h is None:
        final_ratio = clipped_8h
        reason = f"Only 8h window valid; ratio={final_ratio:.3f}"
    else:
        # Both available — pick the one closer to 1.0
        if abs(clipped_8h - 1.0) < abs(clipped_24h - 1.0):
            final_ratio = clipped_8h
            reason = f"8h more conservative; ratio={final_ratio:.3f}"
        else:
            final_ratio = clipped_24h
            reason = f"24h more conservative; ratio={final_ratio:.3f}"

    return {
        "ratio":        round(final_ratio, 4),
        "ratio_8h":     round(clipped_8h,  4) if clipped_8h  is not None else None,
        "ratio_24h":    round(clipped_24h, 4) if clipped_24h is not None else None,
        "n_points_8h":  n_8h,
        "n_points_24h": n_24h,
        "reason":       reason,
    }


# ─────────────────────────────────────────────────────────────────────────────
#   Convenience: apply ratio to therapy settings
# ─────────────────────────────────────────────────────────────────────────────

def apply_autosens_to_isf(autotune_isf: float, autosens_ratio: float) -> float:
    """
    Return the effective ISF after applying the autosens ratio.

    oref0 lib/autosens.js (and determine-basal):
        effective_isf = autotune_isf / autosens_ratio

    A ratio > 1.0 means the patient is currently MORE sensitive than baseline
    → divide by a number > 1 → ISF goes DOWN → corrections are larger.

    A ratio < 1.0 means the patient is currently LESS sensitive (more resistant)
    → divide by a number < 1 → ISF goes UP → corrections are smaller.

    This is the ONLY place the autosens ratio is applied to ISF.
    The autotune_isf (daily baseline) is never modified by autosens.
    """
    if autosens_ratio <= 0:
        return autotune_isf
    return round(autotune_isf / autosens_ratio, 3)


def apply_autosens_to_basal(pump_basal: float, autosens_ratio: float) -> float:
    """
    Return the effective basal rate after applying the autosens ratio.

    oref0:
        effective_basal = pump_basal * autosens_ratio

    A ratio > 1.0 → multiply → basal goes UP (more insulin for sensitive patient).
    A ratio < 1.0 → multiply → basal goes DOWN (less insulin for resistant patient).

    Note the asymmetry with ISF:
        ISF   is DIVIDED   by the ratio
        basal is MULTIPLIED by the ratio
    Both corrections push in the same clinical direction (more sensitive → more insulin).
    """
    return round(pump_basal * autosens_ratio, 4)

def apply_autosens_to_target(
    target_min: float,
    target_max: float,
    autosens_ratio: float,
    sensitivity_raises_target: bool = True,
    resistance_lowers_target: bool = False,
) -> Tuple[float, float]:
    """
    Adjust BG target based on autosens ratio (oref0 behavior). To avoid unstable feedback loops
    
    Inverts the sensitivity ratio to adjust target away from the danger zone:
      - High sensitivity (ratio < 1) → raise target to avoid hypos
      - High resistance (ratio > 1) → lower target to tighten control
    
    The formula: new_target = (old_target - 60) / ratio + 60
    This keeps the scale from 60 (hypo threshold) symmetric.
    
    Parameters
    ----------
    target_min, target_max : float
        Current target bounds (mg/dL)
    autosens_ratio : float
        Autosens sensitivity ratio (0.7 to 1.2)
    sensitivity_raises_target : bool
        If True and ratio < 1 (sensitive), raise target
    resistance_lowers_target : bool
        If True and ratio > 1 (resistant), lower target
    
    Returns
    -------
    Tuple[float, float]
        (new_min, new_max) adjusted targets
    """
    new_min = target_min
    new_max = target_max
    
    # Raise target if sensitive
    if sensitivity_raises_target and autosens_ratio < 1.0:
        new_min = round((target_min - 60) / autosens_ratio) + 60
        new_max = round((target_max - 60) / autosens_ratio) + 60
    
    # Lower target if resistant
    elif resistance_lowers_target and autosens_ratio > 1.0:
        new_min = round((target_min - 60) / autosens_ratio) + 60
        new_max = round((target_max - 60) / autosens_ratio) + 60
    
    return new_min, new_max