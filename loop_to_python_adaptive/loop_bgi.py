from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable

import numpy as np
import pandas as pd

import loop_to_python_adaptive.api as api

FIVE_MINUTES = 5


# -------------------------
# Time + schedule utilities
# -------------------------

def parse_loop_timestamp(iso8601: str) -> datetime:
    """Parse Loop ISO8601 timestamps like '2023-10-17T20:59:03Z' into tz-aware datetime."""
    return datetime.fromisoformat(iso8601.replace("Z", "+00:00"))


def _lookup_schedule_value(at_time: datetime, schedule: list[dict]) -> float:
    """
    Look up schedule value at time from Loop schedule segments:
      [{"startDate","endDate","value"}...]
    Includes closest-prior fallback.
    """
    # exact segment match
    for seg in schedule:
        start = parse_loop_timestamp(seg["startDate"])
        end = parse_loop_timestamp(seg["endDate"])
        if start <= at_time < end:
            return float(seg["value"])

    # closest prior segment
    closest = None
    closest_start = None
    for seg in schedule:
        start = parse_loop_timestamp(seg["startDate"])
        if start <= at_time and (closest_start is None or start > closest_start):
            closest = seg
            closest_start = start

    if closest is None:
        raise ValueError("No schedule segment covers or precedes at_time")

    return float(closest["value"])


def lookup_isf_mgdl_per_u(at_time: datetime, sensitivity_schedule: list[dict]) -> float:
    """ISF (mg/dL/U) at time."""
    return _lookup_schedule_value(at_time, sensitivity_schedule)


def lookup_basal_rate_u_per_hour(at_time: datetime, basal_schedule: list[dict]) -> float:
    """Basal rate (U/hr) at time."""
    return _lookup_schedule_value(at_time, basal_schedule)


def convert_basal_rate_to_microbolus_units(basal_rate_u_per_hour: float, step_minutes: int = FIVE_MINUTES) -> float:
    """Convert U/hr to U delivered over one timestep."""
    return basal_rate_u_per_hour * (step_minutes / 60.0)


# -------------------------
# Dose event extraction
# -------------------------

def extract_bolus_events(loop_algorithm_input: dict) -> list[tuple[datetime, float]]:
    """
    Extract bolus events from LoopAlgorithmInput doses[]:
      {"type":"bolus","startDate":...,"volume":...}
    Returns list[(time, units)].
    """
    out: list[tuple[datetime, float]] = []
    for dose in loop_algorithm_input.get("doses", []):
        if dose.get("type") != "bolus":
            continue
        t = parse_loop_timestamp(dose["startDate"])
        u = float(dose["volume"])
        if u > 0:
            out.append((t, u))
    return out


def build_microbasal_events(
    loop_algorithm_input: dict,
    *,
    start_time: datetime,
    end_time: datetime,
    step_minutes: int = FIVE_MINUTES,
) -> list[tuple[datetime, float]]:
    """
    Discretize scheduled basal into micro-doses: one event every step_minutes.
    Returns list[(time, units_over_step)].
    """
    basal_schedule = loop_algorithm_input["basal"]
    events: list[tuple[datetime, float]] = []

    t = start_time
    while t < end_time:
        rate = lookup_basal_rate_u_per_hour(t, basal_schedule)  # U/hr
        u = convert_basal_rate_to_microbolus_units(rate, step_minutes=step_minutes)
        if u > 0:
            events.append((t, u))
        t += timedelta(minutes=step_minutes)

    return events


# -------------------------
# BGI computation
# -------------------------

@dataclass(frozen=True)
class BGIConfig:
    action_duration_minutes: int
    peak_activity_minutes: int
    delay_minutes: int
    step_minutes: int = FIVE_MINUTES
    history_hours: int = 16

    # numeric safety
    min_effect_fraction: float = 0.0  # tune later if needed


def _batch_percent_remaining(
    fn: Callable[[float, float, float, float], float],
    ages_minutes: np.ndarray,
    *,
    action_duration_minutes: int,
    peak_activity_minutes: int,
    delay_minutes: int,
) -> np.ndarray:
    """
    Batch wrapper around insulin_percent_effect_remaining (scalar API).
    """
    return np.fromiter(
        (
            fn(
                float(m),
                float(action_duration_minutes),
                float(peak_activity_minutes),
                float(delay_minutes),
            )
            for m in ages_minutes
        ),
        dtype=float,
        count=len(ages_minutes),
    )


def add_bgi_to_history_df(
    df: pd.DataFrame,
    *,
    loop_algorithm_input: dict,
    insulin_percent_effect_remaining: Callable[[float, float, float, float], float] | None = None,
    config: BGIConfig,
    cgm_col: str = "CGM",
    bgi_col: str = "BGI",
) -> pd.DataFrame:
    """
    Add BGI (mg/dL per step_minutes) aligned to df.index.

    BGI(t) computed oref0-style:
      units_effect_next_step = sum(dose_units * (R(age) - R(age+step)))
      BGI = - units_effect_next_step * ISF(t)

    Upstream call policy:
      - If insulin_percent_effect_remaining is not supplied, this function will
        use loop_to_python_adaptive.api.insulin_percent_effect_remaining internally.
      - This keeps upstream usage contained inside loop_bgi.py.
    """
    if insulin_percent_effect_remaining is None:
        insulin_percent_effect_remaining = api.insulin_percent_effect_remaining

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must have a DatetimeIndex.")
    if cgm_col not in df.columns:
        raise ValueError(f"df missing required column {cgm_col!r}.")

    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")

    out = df.copy()
    out.index = idx
    out[bgi_col] = np.nan

    # Build event window: include enough history so older doses are negligible
    end_time = idx.max().to_pydatetime()
    start_time = (idx.min() - timedelta(hours=config.history_hours)).to_pydatetime()

    bolus = extract_bolus_events(loop_algorithm_input)
    microbasal = build_microbasal_events(
        loop_algorithm_input,
        start_time=start_time,
        end_time=end_time,
        step_minutes=config.step_minutes,
    )
    events = sorted(bolus + microbasal, key=lambda x: x[0])

    if not events:
        out[bgi_col] = 0.0
        return out

    event_times = np.array([t for (t, _) in events], dtype="datetime64[ns]")
    event_units = np.array([u for (_, u) in events], dtype=float)

    left = 0
    right = 0
    action_td = np.timedelta64(config.action_duration_minutes, "m")
    step = float(config.step_minutes)

    # main loop
    for t in out.index.to_pydatetime():
        t64 = np.datetime64(t)

        # include events <= t
        while right < len(event_times) and event_times[right] <= t64:
            right += 1

        # exclude events < t - action_duration
        cutoff = t64 - action_td
        while left < right and event_times[left] < cutoff:
            left += 1

        if left >= right:
            out.at[pd.Timestamp(t, tz="UTC"), bgi_col] = 0.0
            continue

        active_times = event_times[left:right]
        active_units = event_units[left:right]
        ages = ((t64 - active_times) / np.timedelta64(1, "m")).astype(float)

        r_now = _batch_percent_remaining(
            insulin_percent_effect_remaining,
            ages,
            action_duration_minutes=config.action_duration_minutes,
            peak_activity_minutes=config.peak_activity_minutes,
            delay_minutes=config.delay_minutes,
        )
        r_later = _batch_percent_remaining(
            insulin_percent_effect_remaining,
            ages + step,
            action_duration_minutes=config.action_duration_minutes,
            peak_activity_minutes=config.peak_activity_minutes,
            delay_minutes=config.delay_minutes,
        )

        frac = np.maximum(config.min_effect_fraction, r_now - r_later)
        units_effect_next = float(np.sum(active_units * frac))

        sens = lookup_isf_mgdl_per_u(t, loop_algorithm_input["sensitivity"])
        out.at[pd.Timestamp(t, tz="UTC"), bgi_col] = -units_effect_next * sens

    return out


def build_isf_glucose_data_from_df(
    df: pd.DataFrame,
    *,
    cgm_col: str = "CGM",
    bgi_col: str = "BGI",
) -> list[dict]:
    """
    Build ISF tuning points for tune_isf_like_oref0().

    Align BGI as "effect over next timestep":
      avgDelta[i] = CGM[i] - CGM[i-1]
      deviation[i] = avgDelta[i] - BGI[i-1]

    Returns list of dicts with keys: date, avgDelta, BGI, deviation.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must have a DatetimeIndex.")
    if cgm_col not in df.columns:
        raise ValueError(f"df missing {cgm_col!r}")
    if bgi_col not in df.columns:
        raise ValueError(f"df missing {bgi_col!r}; call add_bgi_to_history_df first")

    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")

    cgm = pd.to_numeric(df[cgm_col], errors="coerce")
    bgi = pd.to_numeric(df[bgi_col], errors="coerce")

    pts: list[dict] = []
    for i in range(1, len(df)):
        if pd.isna(cgm.iat[i]) or pd.isna(cgm.iat[i - 1]):
            continue
        if pd.isna(bgi.iat[i - 1]):
            continue

        avg_delta = float(cgm.iat[i] - cgm.iat[i - 1])  # mg/dL per step
        bgi_step = float(bgi.iat[i - 1])                # mg/dL per step
        deviation = float(avg_delta - bgi_step)

        ts = idx[i]
        pts.append({"date": ts.isoformat(), "avgDelta": avg_delta, "BGI": bgi_step, "deviation": deviation})

    return pts


def prepare_isf_glucose_data(
    df: pd.DataFrame,
    *,
    loop_algorithm_input: dict,
    config: BGIConfig,
    cgm_col: str = "CGM",
    bgi_col: str = "BGI",
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Convenience wrapper for autotune_prep:
      - compute df_with_BGI
      - build isf_glucose_data list

    Returns:
      (df_with_BGI, isf_glucose_data)
    """
    df2 = add_bgi_to_history_df(
        df,
        loop_algorithm_input=loop_algorithm_input,
        config=config,
        cgm_col=cgm_col,
        bgi_col=bgi_col,
    )
    pts = build_isf_glucose_data_from_df(df2, cgm_col=cgm_col, bgi_col=bgi_col)
    return df2, pts