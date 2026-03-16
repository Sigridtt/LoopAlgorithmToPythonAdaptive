import json
from pathlib import Path
from loop_to_python_api.api import get_active_insulin
from loop_to_python_api.helpers import get_json_loop_prediction_input_from_df
import pandas as pd

# Use your fixture
loop_input = json.loads(Path("tests/test_files/loop_algorithm_input.json").read_text())
glucose = loop_input["glucoseHistory"]
idx = pd.to_datetime([g["date"] for g in glucose], utc=True)
df = pd.DataFrame({"CGM": [float(g["value"]) for g in glucose]}, index=idx).sort_index()

# Build a small sub-window
sub = df.iloc[:5]
from loop_to_python_adaptive.autotune_isf import extract_pump_basal, extract_pump_isf, extract_pump_cr
json_data = get_json_loop_prediction_input_from_df(
    sub,
    extract_pump_basal(loop_input),
    extract_pump_isf(loop_input),
    extract_pump_cr(loop_input),
    sub.index[-1],
    insulin_type="novolog",
)
result = get_active_insulin(json_data)
print(type(result), result)