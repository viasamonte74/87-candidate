"""Replaceable miner-owned methods (role "candidate").

A method here is written from the public contract alone (``institutional_v02.contracts``,
``canonical``); tests/test_separation.py proves by import graph and at runtime that this
package never reaches an evaluator-only module. Methods register under (class_id, method_id)
and are bound to a class with ``classes.with_candidates``. No registered method is a chain
participant until a uid is assigned.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

Method = Callable[[dict[str, Any]], dict[str, Any]]
_REGISTRY: dict[str, dict[str, Method]] = {}


def register(class_id: str, method_id: str, fn: Method) -> None:
    methods = _REGISTRY.setdefault(class_id, {})
    if method_id in methods:
        raise ValueError(f"method {method_id!r} already registered for {class_id}")
    methods[method_id] = fn


def for_class(class_id: str) -> dict[str, Method]:
    return dict(_REGISTRY.get(class_id, {}))


from . import witness_ic, witness_type_c  # noqa: E402,F401  (register on import)
