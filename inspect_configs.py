"""
inspect_configs.py  –  run this once to find the real AutotuneConfig / AutosensConfig fields.

    cd C:\\...\\LoopAlgorithmToPythonAdaptive
    .venv\\Scripts\\activate
    python inspect_configs.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))                        # project root
sys.path.insert(0, str(HERE / "loop_to_python_adaptive"))  # package

import inspect

def show(cls):
    print(f"\n=== {cls.__module__}.{cls.__name__} ===")
    try:
        sig = inspect.signature(cls.__init__)
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            default = "" if param.default is inspect.Parameter.empty else f" = {param.default!r}"
            annotation = f": {param.annotation}" if param.annotation is not inspect.Parameter.empty else ""
            print(f"  {name}{annotation}{default}")
    except Exception as e:
        print(f"  ERROR: {e}")

try:
    from autotune import AutotuneConfig, run_autotune
    show(AutotuneConfig)
    print(f"  run_autotune signature: {inspect.signature(run_autotune)}")
except ImportError as e:
    print(f"autotune import failed: {e}")

try:
    from autosens import AutosensConfig, AutosensPoint, AutosensBuffer
    show(AutosensConfig)
    show(AutosensPoint)
except ImportError as e:
    print(f"autosens import failed: {e}")