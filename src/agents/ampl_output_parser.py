from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class SolveInfo:
    status: str
    objective: Optional[float]


_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_REFERENCE_OBJ_RE = re.compile(
    rf"REFERENCE_OBJECTIVE\s*(?:[:=]\s*)?({_NUMBER})", re.IGNORECASE
)
_OBJECTIVE_VALUE_RE = re.compile(
    rf"objective_value\s*[:=]\s*({_NUMBER})", re.IGNORECASE
)
_OBJ_RE = re.compile(rf"objective\s+({_NUMBER})", re.IGNORECASE)


def parse_solve_output(text: str) -> SolveInfo:
    t = text or ""
    status = "unknown"

    low = t.lower()
    if "optimal solution" in low or "optimal" in low:
        status = "optimal"
    if "infeasible" in low:
        status = "infeasible"
    if "unbounded" in low:
        status = "unbounded"

    obj = None
    # Prefer explicit machine-readable values over solver progress text. Solver
    # logs may contain an earlier rounded objective in scientific notation.
    m = _REFERENCE_OBJ_RE.search(t) or _OBJECTIVE_VALUE_RE.search(t) or _OBJ_RE.search(t)
    if m:
        try:
            obj = float(m.group(1))
        except Exception:
            obj = None

    return SolveInfo(status=status, objective=obj)
