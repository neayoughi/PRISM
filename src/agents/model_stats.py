import re
from dataclasses import dataclass
from typing import Dict

@dataclass
class ModelStats:
    n_vars: int
    n_params: int
    n_sets: int
    n_objectives: int
    n_constraints_decl: int

def _strip_comments(text: str) -> str:
    # Remove everything after # on each line
    return re.sub(r"#.*", "", text)

def compute_model_stats(model_text: str) -> ModelStats:
    t = _strip_comments(model_text)

    # Simple AMPL-ish patterns
    var_pat = re.compile(r"(?m)^\s*var\s+([A-Za-z_]\w*)\b")
    param_pat = re.compile(r"(?m)^\s*param\s+([A-Za-z_]\w*)\b")
    set_pat = re.compile(r"(?m)^\s*set\s+([A-Za-z_]\w*)\b")

    # Objectives may be: minimize X: ... or maximize X: ...
    obj_pat = re.compile(r"(?m)^\s*(minimize|maximize)\s+([A-Za-z_]\w*)\s*:")

    # Constraint declarations often: subject to Name ... :
    con_pat = re.compile(r"(?m)^\s*subject\s+to\s+([A-Za-z_]\w*)\b")

    return ModelStats(
        n_vars=len(var_pat.findall(t)),
        n_params=len(param_pat.findall(t)),
        n_sets=len(set_pat.findall(t)),
        n_objectives=len(obj_pat.findall(t)),
        n_constraints_decl=len(con_pat.findall(t)),
    )

def stats_to_dict(s: ModelStats) -> Dict[str, int]:
    return {
        "n_sets": s.n_sets,
        "n_params": s.n_params,
        "n_vars": s.n_vars,
        "n_objectives": s.n_objectives,
        "n_constraints_decl": s.n_constraints_decl,
    }
