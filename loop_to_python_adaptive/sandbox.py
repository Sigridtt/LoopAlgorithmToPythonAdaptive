from pathlib import Path
import json

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parent.parent  # loop_to_python_adaptive/ -> repo root

FIXTURE = REPO_ROOT / "python_tests" / "test_files" / "loop_algorithm_input.json"
loop_input = json.loads(FIXTURE.read_text(encoding="utf-8"))
print("Loaded fixture:", FIXTURE)