"""
Step 1.1 — Verify the core pipeline:

    df (from fixture)
        → prepare_for_autotune_isf   (autotune_prep)
        → run_autotune_isf_iterations (autotune_isf)
        → newISF

No SimGlucose. No AdaptiveLoopController. Just the data pipeline.

Run with:
    pytest tests/test_pipeline.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from loop_to_python_adaptive.autotune_isf import (
    run_autotune_isf_iterations,
    extract_pump_isf,
    extract_pump_basal,
    extract_pump_cr,
)
from loop_to_python_adaptive.autotune_prep import AutotunePrepConfig


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
    """Build a CGM DataFrame from the fixture's glucoseHistory."""
    glucose = loop_input["glucoseHistory"]
    idx = pd.to_datetime([g["date"] for g in glucose], utc=True)
    df = pd.DataFrame(
        {"CGM": [float(g["value"]) for g in glucose]},
        index=idx,
    ).sort_index()
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


# ── Test 2: full pipeline runs and ISF changes ────────────────────────────────

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

    isf_after  = result["finalISF"]
    last       = result["last_result"]

    print(f"ISF after:  {isf_after}")
    print(f"n_points:   {last['n_points']}")
    print(f"reason:     {last['reason']}")
    print(f"p50_ratio:  {last['p50_ratio']}")

    # Must have run successfully
    assert last["reason"] == "OK", (
        f"Autotune did not run — reason: {last['reason']}. "
        f"Only {last['n_points']} ISF points found in fixture. "
        "Check that the fixture has enough basal-only periods."
    )

    # ISF must have changed
    assert isf_before != isf_after, (
        f"ISF unchanged at {isf_before}. "
        "Autotune ran but produced no change — check p50_ratio above."
    )

    # newISF must be finite and positive
    assert isf_after > 0
    assert isf_after < 500  # sanity upper bound


# ── Test 3: multiple iterations converge ─────────────────────────────────────

def test_pipeline_multiple_iterations_stable(loop_input, df_window):
    """
    Running 3 iterations should not crash and ISF should stay within bounds.
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
        assert isf > 0, f"Iteration {i+1}: ISF={isf} is not positive"
        # Must stay within autosens bounds (default 0.7–1.2 × pump_isf)
        assert isf >= pump_isf * 0.7 * 0.99, f"ISF {isf} too low vs pump {pump_isf}"
        assert isf <= pump_isf / 0.7 * 1.01, f"ISF {isf} too high vs pump {pump_isf}"

    print(f"\nISF history over 3 iterations: {history}")