from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from loop_to_python_adaptive.autotune_prep import AutotunePrepConfig, prepare_for_autotune_isf
from loop_to_python_adaptive.autotune_isf import tune_isf_like_oref0


def find_repo_root(start: Path) -> Path:
    p = start
    while True:
        if (p / "loop_to_python_adaptive").exists():
            return p
        if p.parent == p:
            raise RuntimeError("Could not find repo root (no loop_to_python_adaptive directory found).")
        p = p.parent


def test_autotune_isf_runs_on_loop_algorithm_input_fixture():
    repo_root = find_repo_root(Path(__file__).resolve())
    file = repo_root / "tests" / "test_files" / "loop_algorithm_input.json"
    assert file.exists(), f"Missing file: {file}"

    loop_input = json.loads(file.read_text(encoding="utf-8"))

    glucose = loop_input["glucoseHistory"]
    assert len(glucose) > 20

    idx = pd.to_datetime([g["date"] for g in glucose], utc=True)
    df = pd.DataFrame({"CGM": [float(g["value"]) for g in glucose]}, index=idx).sort_index()

    # Key change: fixture basal schedule starts ~6h before first glucose point
    prep = prepare_for_autotune_isf(
        df,
        loop_algorithm_input=loop_input,
        config=AutotunePrepConfig(history_hours=6),
    )

    res = tune_isf_like_oref0(
        isf_current=50.0,
        pump_isf=None,
        isf_glucose_data=prep["isf_glucose_data"],
    )
    print("Autotune ISF result:", res)
    print(len(prep["ISFGlucoseData"]), len(prep["CSFGlucoseData"]), len(prep["UAMGlucoseData"]), len(prep["basalGlucoseData"]))

    assert len(prep["ISFGlucoseData"]) > 0
    assert len(prep["CSFGlucoseData"]) >= 0
    assert len(prep["UAMGlucoseData"]) >= 0
    assert res["reason"] == "OK", res
    assert res["n_points"] >= 10, res
    assert np.isfinite(res["newISF"]), res
    assert abs(res["newISF"] - 80.0) > 0.01, res