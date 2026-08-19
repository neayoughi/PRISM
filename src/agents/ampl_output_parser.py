from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class SolveInfo:
    status: str
    objective: Optional[float]


_OBJ_RE = re.compile(r"objective\s+(-?\d+(?:\.\d+)?)", re.IGNORECASE)


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
    m = _OBJ_RE.search(t)
    if m:
        try:
            obj = float(m.group(1))
        except Exception:
            obj = None

    return SolveInfo(status=status, objective=obj)
