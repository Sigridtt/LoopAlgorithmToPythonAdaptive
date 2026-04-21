"""
Step 1.1 — Verify the core pipeline:

    df (from fixture)
        → prepare_for_autotune_isf   (autotune_prep)
        → run_autotune_isf_iterations (autotune_isf)
        → newISF
        → update_profile_isf         (autotune_isf)
        → updated loop_algorithm_input

No SimGlucose. Just the data pipeline.

Run with:
    pytest tests/test_pipeline.py -v -s
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from loop_to_python_adaptive.autotune import (
    run_autotune_isf_iterations,
    update_profile_isf,
    extract_pump_isf,
    extract_pump_basal,
    extract_pump_cr,
)
from loop_to_python_adaptive.loop_oref_mapping import add_bgi_to_history_df



def find_repo_root(start: Path) -> Path:
    p = start
    while True:
        if (p / "loop_to_python_adaptive").exists():
            return p
        if p.parent == p:
            raise RuntimeError("Could not find repo root.")
        p = p.parent


@pytest.fixture
def loop_input() -> dict:
    repo_root = find_repo_root(Path(__file__).resolve())
    file = repo_root / "tests" / "test_files" / "loop_algorithm_input.json"
    assert file.exists(), f"Missing fixture: {file}"
    return json.loads(file.read_text(encoding="utf-8"))


@pytest.fixture
def df_window(loop_input) -> pd.DataFrame:
    """
    Build a CGM DataFrame from the fixture's glucoseHistory,
    with basal and bolus columns extracted from doses.
    """
    # ── CGM ──────────────────────────────────────────────────────────────
    glucose = loop_input["glucoseHistory"]
    idx = pd.to_datetime([g["date"] for g in glucose], utc=True)
    df = pd.DataFrame(
        {"CGM": [float(g["value"]) for g in glucose]},
        index=idx,
    ).sort_index()

    # ── Basal and bolus from doses ────────────────────────────────────────
    # Default to pump basal everywhere; overwrite with actual dose records
    pump_basal = extract_pump_basal(loop_input)
    df["basal"] = pump_basal / 60   # U/hr → U/min to match SimGlucose convention
    df["bolus"] = 0.0

    doses = loop_input.get("doses", [])
    for dose in doses:
        dose_time = pd.to_datetime(dose["startDate"], utc=True)
        dose_type = dose.get("type", "")
        volume    = float(dose.get("volume", 0) or 0)

        # Find the closest CGM timestamp
        if len(df) == 0:
            continue
        closest_idx = df.index.get_indexer([dose_time], method="nearest")[0]

        if dose_type == "basal":
            df.iloc[closest_idx, df.columns.get_loc("basal")] = volume / 60
        elif dose_type == "bolus":
            df.iloc[closest_idx, df.columns.get_loc("bolus")] = volume

    return df

# ── Test 1: extractors work ───────────────────────────────────────────────────

def test_extract_pump_settings(loop_input):
    """pump_isf, pump_basal, pump_cr can be extracted from the fixture."""
    isf   = extract_pump_isf(loop_input)
    basal = extract_pump_basal(loop_input)
    cr    = extract_pump_cr(loop_input)

    assert isf   > 0, f"pump_isf={isf}"
    assert basal > 0, f"pump_basal={basal}"
    assert cr    > 0, f"pump_cr={cr}"
    print(f"\npump_isf={isf}, pump_basal={basal}, pump_cr={cr}")

# ── Test 2: diagnose categorisation ──────────────────────────────────────────

def test_pipeline_categorisation(loop_input, df_window):
    """
    Diagnostic: show how many points land in each bucket.
    Helps understand why ISF points may be 0.
    Run with: pytest tests/test_pipeline.py::test_pipeline_categorisation -v -s
    """
    from loop_to_python_adaptive.autotune_prep import (
        AutotunePrepConfig,
        prepare_for_autotune_isf,
    )
    from loop_to_python_adaptive.autotune import extract_pump_basal, extract_pump_cr

    cfg = AutotunePrepConfig(
        basal_rate=extract_pump_basal(loop_input),
        isf=extract_pump_isf(loop_input),
        carb_ratio=extract_pump_cr(loop_input),
    )

    result = prepare_for_autotune_isf(
        df_window,
        loop_algorithm_input=loop_input,
        cfg=cfg,
    )

    print(f"\nCGM window: {df_window.index[0]} → {df_window.index[-1]}")
    print(f"Total points:  {len(df_window)}")
    print(f"ISF points:    {len(result['ISFGlucoseData'])}")
    print(f"CSF points:    {len(result['CSFGlucoseData'])}")
    print(f"UAM points:    {len(result['UAMGlucoseData'])}")
    print(f"Basal points:  {len(result['basalGlucoseData'])}")
    df_bgi = add_bgi_to_history_df(df_window, isf=extract_pump_isf(loop_input), loop_algorithm_input=loop_input)
    print(f"BGI NaN count: {df_bgi['BGI'].isna().sum()} / {len(df_bgi)}")
    print(f"BGI sample:\n{df_bgi['BGI'].head(10)}")

# ── Test 3: full pipeline — ISF changes ──────────────────────────────────────

def test_pipeline_isf_changes(loop_input, df_window):
    """
    Core Step 1.1 test:
      df → run_autotune_isf_iterations → newISF
    Done when: isf_before != isf_after
    """
    isf_before = extract_pump_isf(loop_input)
    print(f"\nISF before: {isf_before}")

    result = run_autotune_isf_iterations(
        [df_window],
        loop_algorithm_inputs=[loop_input],
        n_iterations=1,
    )

    isf_after = result["finalISF"]
    last      = result["last_result"]

    print(f"ISF after:  {isf_after}")
    print(f"n_points:   {last['n_points']}")
    print(f"reason:     {last['reason']}")
    print(f"p50_ratio:  {last['p50_ratio']}")

    assert last["reason"] == "OK", (
        f"Autotune did not run — reason: {last['reason']}. "
        f"Only {last['n_points']} ISF points found. "
        "Check the fixture has enough basal-only periods."
    )
    assert isf_before != isf_after, (
        f"ISF unchanged at {isf_before}. "
        "Autotune ran but p50_ratio=1.0 — check deviation/BGI values."
    )
    assert isf_after > 0
    assert isf_after < 500


# ── Test 4: update_profile_isf applies newISF to the dict ────────────────────

def test_update_profile_isf(loop_input, df_window):
    """
    After autotune, update_profile_isf must return a dict whose
    extract_pump_isf equals the finalISF from autotune.
    """
    result = run_autotune_isf_iterations(
        [df_window],
        loop_algorithm_inputs=[loop_input],
        n_iterations=1,
    )

    new_isf     = result["finalISF"]
    new_profile = update_profile_isf(loop_input, new_isf)

    # The updated dict must have the new ISF
    assert extract_pump_isf(new_profile) == pytest.approx(new_isf, rel=1e-3), (
        f"Expected ISF {new_isf} in updated profile, "
        f"got {extract_pump_isf(new_profile)}"
    )

    # Original must be unchanged
    original_isf = extract_pump_isf(loop_input)
    assert original_isf != new_isf or result["last_result"]["reason"] != "OK", \
        "Original loop_input was mutated — deepcopy missing in update_profile_isf"

    print(f"\nISF in updated profile: {extract_pump_isf(new_profile):.3f}")


# ── Test 5: multiple iterations stay within safety bounds ────────────────────

def test_pipeline_multiple_iterations_stable(loop_input, df_window):
    """
    Running 3 iterations should not crash and ISF must stay within
    oref0's autosens bounds: [pump_isf/1.2, pump_isf/0.7]
    """
    pump_isf = extract_pump_isf(loop_input)

    result = run_autotune_isf_iterations(
        [df_window],
        loop_algorithm_inputs=[loop_input],
        n_iterations=3,
    )

    history = result["isf_history"]
    assert len(history) == 3, f"Expected 3 history entries, got {len(history)}"

    for i, isf in enumerate(history):
        assert isf > 0, f"Iteration {i+1}: ISF={isf} not positive"
        assert isf >= pump_isf / 1.2 * 0.99, f"Iteration {i+1}: ISF {isf} below min bound"
        assert isf <= pump_isf / 0.7 * 1.01, f"Iteration {i+1}: ISF {isf} above max bound"

    print(f"\nISF history over 3 iterations: {[round(x,3) for x in history]}")