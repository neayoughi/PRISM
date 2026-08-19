from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class RunPaths:
    problem_id: str
    run_id: str
    gen_run_dir: str
    ref_solution_dir: str
    gen_model: str
    gen_data: str
    gen_output: str
    ref_model: str
    ref_data: str
    ref_output: str


_ATTEMPT_RE = re.compile(r"^model-debug_attempt_(\d+)\.mod$")


def _read_text(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _pick_generated_model_path(gen_run_dir: str) -> str:
    """
    Prefer the latest model-debug_attempt_#.mod when present.
    Fall back to model.mod.
    """
    best_n = -1
    best_path = ""

    try:
        names = os.listdir(gen_run_dir)
    except OSError:
        names = []

    for name in names:
        m = _ATTEMPT_RE.match(name)
        if not m:
            continue
        n = int(m.group(1))
        if n > best_n:
            best_n = n
            best_path = os.path.join(gen_run_dir, name)

    if best_path and os.path.exists(best_path):
        return best_path

    return os.path.join(gen_run_dir, "model.mod")


def load_run(gen_run_dir: str, ref_solution_dir: str, problem_id: str, run_id: str) -> RunPaths:
    gen_model_path = _pick_generated_model_path(gen_run_dir)
    gen_data_path = os.path.join(gen_run_dir, "data.dat")

    gen_output_path = os.path.join(gen_run_dir, "final_solution.txt")
    if not os.path.exists(gen_output_path):
        gen_output_path = os.path.join(gen_run_dir, "initial_solution.txt")

    ref_model_path = os.path.join(ref_solution_dir, "model.mod")
    ref_data_path = os.path.join(ref_solution_dir, "data.dat")

    ref_output_path = os.path.join(ref_solution_dir, "Terminal Saved Output.txt")
    if not os.path.exists(ref_output_path):
        ref_output_path = os.path.join(ref_solution_dir, "final_solution.txt")

    return RunPaths(
        problem_id=problem_id,
        run_id=run_id,
        gen_run_dir=gen_run_dir,
        ref_solution_dir=ref_solution_dir,
        gen_model=_read_text(gen_model_path),
        gen_data=_read_text(gen_data_path),
        gen_output=_read_text(gen_output_path),
        ref_model=_read_text(ref_model_path),
        ref_data=_read_text(ref_data_path),
        ref_output=_read_text(ref_output_path),
    )


def discover_runs(gen_root: str, ref_root: str) -> List[Tuple[str, str, str, str]]:
    """Return (problem_id, run_id, gen_run_dir, ref_solution_dir)."""
    out: List[Tuple[str, str, str, str]] = []
    if not os.path.isdir(gen_root):
        return out

    for problem_id in sorted(os.listdir(gen_root)):
        pdir = os.path.join(gen_root, problem_id)
        if not os.path.isdir(pdir):
            continue

        ref_solution_dir = os.path.join(ref_root, problem_id, "solution")
        if not os.path.isdir(ref_solution_dir):
            continue

        for run_id in sorted(os.listdir(pdir)):
            rdir = os.path.join(pdir, run_id)
            if not os.path.isdir(rdir):
                continue
            out.append((problem_id, run_id, rdir, ref_solution_dir))

    return out
