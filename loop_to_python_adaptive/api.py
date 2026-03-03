from __future__ import annotations

import importlib
from typing import Any

_upstream_api = importlib.import_module("loop_to_python_api.api")
__all__ = list(getattr(_upstream_api, "__all__", [n for n in dir(_upstream_api) if not n.startswith("_")]))


def __getattr__(name: str) -> Any:
    return getattr(_upstream_api, name)


def __dir__() -> list[str]:
    return sorted(set(__all__))