"""
conftest.py
C:\\...\\LoopAlgorithmToPythonAdaptive\\conftest.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

_HERE         = Path(__file__).parent.resolve()
_PROJECT_ROOT = _HERE.parent.resolve()
_DEFAULT_OREF0_BIN = _PROJECT_ROOT / "oref0" / "bin"


def pytest_configure(config: "pytest.Config") -> None:
    # Add the flat package directory so 'from autotune import ...' works
    pkg = _HERE / "loop_to_python_adaptive"
    if pkg.exists() and str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))
        print(f"\n[conftest] sys.path += {pkg}")

    # Also add project root so 'from loop_to_python_adaptive.autotune import ...' works
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))


def pytest_addoption(parser: "pytest.Parser") -> None:
    parser.addoption(
        "--oref0-bin",
        default=str(_DEFAULT_OREF0_BIN),
        help=f"Path to oref0 bin/ directory (default: {_DEFAULT_OREF0_BIN})",
    )


@pytest.fixture(scope="session")
def node_executable() -> str:
    node = shutil.which("node") or shutil.which("node.exe")
    if node is None:
        pytest.skip("Node.js not found on PATH. Install from https://nodejs.org")
    try:
        version = subprocess.check_output([node, "--version"], text=True, timeout=10).strip()
        print(f"\n[conftest] Node.js: {node}  ({version})")
    except Exception as exc:
        pytest.skip(f"node --version failed: {exc}")
    return node


@pytest.fixture(scope="session")
def oref0_bin_dir(request: "pytest.FixtureRequest") -> Path:
    raw     = request.config.getoption("--oref0-bin")
    bin_dir = Path(raw).resolve()

    if not bin_dir.exists():
        pytest.skip(
            f"oref0 bin/ not found at {bin_dir}.\n"
            "Run:  npm install  inside the oref0 folder."
        )

    node_modules = bin_dir.parent / "node_modules"
    if not node_modules.exists():
        pytest.skip(
            f"oref0 node_modules missing ({node_modules}).\n"
            f"Run:  cd {bin_dir.parent}  &&  npm install"
        )

    print(f"\n[conftest] oref0 bin: {bin_dir}")
    return bin_dir


@pytest.fixture(scope="session")
def run_oref0_js(node_executable: str, oref0_bin_dir: Path):
    """
    Mirrors the bash test exactly:
        cd bash-unit-test-temp
        ../bin/oref0-autotune-core.js arg0.json arg1.json ...

    - Files are written into a temp directory
    - The script is called with RELATIVE filenames  (arg0.json, arg1.json …)
    - subprocess cwd = temp directory
    - NODE_PATH points at oref0/node_modules so require() still resolves

    This avoids the Windows absolute-path bug in oref0's require() calls.
    """
    oref0_root     = oref0_bin_dir.parent
    node_modules   = oref0_root / "node_modules"

    def _run(
        script_name: str,
        *json_args: Any,
        extra_flags: Optional[List[str]] = None,
        timeout: int = 60,
    ) -> Tuple[Dict[str, Any], str]:
        script = oref0_bin_dir / script_name
        if not script.exists():
            raise FileNotFoundError(
                f"oref0 script not found: {script}\n"
                f"Available: {[p.name for p in oref0_bin_dir.iterdir()]}"
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Write each arg as a numbered JSON file
            rel_args: List[str] = []
            for i, data in enumerate(json_args):
                fname = f"arg{i}.json"
                (tmp / fname).write_text(json.dumps(data), encoding="utf-8")
                rel_args.append(fname)          # ← RELATIVE, not absolute

            cmd = [node_executable, str(script)]
            if extra_flags:
                cmd += extra_flags
            cmd += rel_args

            # Inherit environment, add NODE_PATH so oref0 require() works
            env = os.environ.copy()
            existing_np = env.get("NODE_PATH", "")
            env["NODE_PATH"] = (
                str(node_modules) + os.pathsep + existing_np
                if existing_np else str(node_modules)
            )

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(tmp),               # ← cd into temp dir, like bash test
                env=env,
            )

            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            try:
                out_json = json.loads(stdout) if stdout else {}
            except json.JSONDecodeError:
                out_json = {"_raw_stdout": stdout}

            return out_json, stderr

    return _run