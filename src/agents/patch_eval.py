from __future__ import annotations

import json
import os
import re
import subprocess
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .ampl_output_parser import parse_solve_output
from .llm_agent import AgentConfig, call_classifier


_OBJVAL_RE = re.compile(r"objective_value\s*[:=]\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)


@dataclass
class PatchEvalResult:
    status: str
    reference_objective: Optional[float]
    patched_objective: Optional[float]
    objective_match: Optional[bool]
    abs_diff: Optional[float]
    rel_diff: Optional[float]
    patch_text: str
    solver_output: str
    attempts_used: int = 0
    max_attempts: int = 0
    attempt_history: List[Dict[str, Any]] = field(default_factory=list)
    patched_model_text: str = ""
    patched_data_text: str = ""



def _safe_read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except FileNotFoundError:
        return ""



def _find_optional_input(gen_run_dir: str, name: str) -> str:
    cand = [
        os.path.join(gen_run_dir, name),
        os.path.join(os.path.dirname(gen_run_dir), name),
        os.path.join(os.path.dirname(os.path.dirname(gen_run_dir)), name),
    ]
    for p in cand:
        if os.path.exists(p):
            return p
    return ""



def _parse_objective_from_solver_output(text: str) -> Optional[float]:
    if not text:
        return None
    m = _OBJVAL_RE.search(text)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    info = parse_solve_output(text)
    return info.objective



def _write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)



def regenerate_model_with_patch(
    cfg: AgentConfig,
    system_prompt: str,
    generated_model: str,
    generated_data: str,
    description_text: str,
    patch_type: str,
    patch_text: str,
    solver_feedback: str = "",
    attempt_number: int = 1,
) -> Dict[str, str]:
    """Return a dict with keys: model_mod (required), data_dat (optional)."""

    user_prompt = (
        "INPUTS\n"
        f"attempt_number: {attempt_number}\n"
        f"patch_type: {patch_type}\n\n"
        "patch_text:\n"
        f"{patch_text}\n\n"
        "description (optional):\n"
        f"{description_text}\n\n"
        "current_model_mod:\n"
        f"{generated_model}\n\n"
        "current_data_dat:\n"
        f"{generated_data}\n"
    )

    if solver_feedback:
        user_prompt += (
            "\nSOLVER_FEEDBACK_FROM_PREVIOUS_ATTEMPT\n"
            f"{solver_feedback}\n\n"
            "Use the solver feedback to refine the patched model while preserving the intended patch.\n"
        )

    raw = call_classifier(cfg, system_prompt, user_prompt)
    raw2 = raw.strip()
    raw2 = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", raw2)
    raw2 = re.sub(r"```\s*$", "", raw2)

    try:
        obj = json.loads(raw2)
    except Exception:
        m = re.search(r"\{.*\}", raw2, flags=re.DOTALL)
        if not m:
            return {"model_mod": raw2}
        obj = json.loads(m.group(0))

    model_mod = str(obj.get("model_mod", "") or obj.get("model", "") or "")
    data_dat = str(obj.get("data_dat", "") or obj.get("data", "") or "")

    model_mod = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", model_mod.strip())
    model_mod = re.sub(r"```\s*$", "", model_mod).strip()
    data_dat = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", data_dat.strip())
    data_dat = re.sub(r"```\s*$", "", data_dat).strip()

    if not model_mod:
        model_mod = raw2

    out: Dict[str, str] = {"model_mod": model_mod}
    if data_dat:
        out["data_dat"] = data_dat
    return out



def _remove_name_definition_from_data(data_text: str, name: str) -> str:
    lines = data_text.splitlines()
    out: List[str] = []
    skip = False
    pat = re.compile(rf"^\s*(set|param)\s+{re.escape(name)}\b")
    for ln in lines:
        if skip:
            if ";" in ln:
                skip = False
            continue
        if pat.search(ln):
            if ";" not in ln:
                skip = True
            continue
        out.append(ln)
    return "\n".join(out) + ("\n" if data_text.endswith("\n") else "")



def solve_with_amplpy(work_dir: str, model_path: str, data_path: str) -> Tuple[str, str]:
    """Run AMPL via amplpy with Gurobi. Return (solver_output_text, output_path)."""
    exec_path = os.path.join(work_dir, "execute_ampl_patch_eval.py")
    out_path = os.path.join(work_dir, "patch_eval_solution.txt")

    ampl_code = textwrap.dedent(
        f"""\
        #!/usr/bin/env python3
        from amplpy import AMPL

        ampl = AMPL()
        ampl.reset()
        ampl.set_option('reset_initial_guesses', True)
        ampl.set_option('send_statuses', False)
        ampl.set_option('solver', 'gurobi')

        ampl.read("./model_patched.mod")
        ampl.read_data("./data.dat")

        try:
            ampl.solve()
            solve_result = ""
            solve_message = ""
            try:
                solve_result = str(ampl.get_value("solve_result"))
            except Exception:
                pass
            try:
                solve_message = str(ampl.get_value("solve_message"))
            except Exception:
                pass
            objs = ampl.getObjectives()
            obj_val = None
            for name in objs:
                try:
                    obj_val = float(objs[name].value())
                    break
                except Exception:
                    pass
            if obj_val is not None:
                print(f"objective_value: {{obj_val}}")
            if solve_result:
                print(f"solve_result: {{solve_result}}")
            if solve_message:
                print(f"solve_message: {{solve_message}}")
            print("ampl_solve_completed")
        except Exception as e:
            import sys
            print("Error:", str(e), file=sys.stderr)
        """
    )
    _write_text(exec_path, ampl_code)

    def _run_once() -> Tuple[int, str, str]:
        p = subprocess.run(
            ["python3", os.path.basename(exec_path)],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        return p.returncode, p.stdout or "", p.stderr or ""

    try:
        rc, stdout, stderr = _run_once()
        if rc != 0:
            m = re.search(r"\n(\w+) was defined in the model\.", stderr)
            if not m:
                m = re.search(r"\b(\w+) was defined in the model\b", stderr)
            if m:
                name = m.group(1)
                dt = _safe_read(data_path)
                dt2 = _remove_name_definition_from_data(dt, name)
                if dt2 != dt:
                    _write_text(data_path, dt2)
                    rc, stdout, stderr = _run_once()

        text_out = stdout if rc == 0 else ("Error: " + (stderr or stdout))
    except subprocess.TimeoutExpired:
        text_out = "Error: AMPL execution timed out."
    except Exception as e:
        text_out = f"Error: AMPL execution failed: {e}"

    _write_text(out_path, text_out)
    return text_out, out_path



def _is_solver_error(text: str) -> bool:
    if not text:
        return True
    low = text.lower()
    if low.startswith("error:"):
        return True
    if "error:" in low:
        return True
    if "ampl execution timed out" in low:
        return True
    return False



def evaluate_patch(
    *,
    cfg: AgentConfig,
    system_prompt: str,
    gen_run_dir: str,
    out_dir: str,
    reference_output_text: str,
    generated_model_text: str,
    generated_data_text: str,
    patch_type: str,
    patch_text: str,
    max_refinement_attempts: int = 5,
    abs_tol: float = 1e-6,
    rel_tol: float = 1e-6,
) -> PatchEvalResult:
    ref_info = parse_solve_output(reference_output_text)
    ref_obj = ref_info.objective

    if ref_obj is None:
        return PatchEvalResult(
            status="missing_reference_objective",
            reference_objective=None,
            patched_objective=None,
            objective_match=None,
            abs_diff=None,
            rel_diff=None,
            patch_text=patch_text,
            solver_output="",
            attempts_used=0,
            max_attempts=max_refinement_attempts,
        )

    desc_path = _find_optional_input(gen_run_dir, "description.txt")
    description_text = _safe_read(desc_path)

    work_dir = os.path.join(out_dir, "patch_eval")
    os.makedirs(work_dir, exist_ok=True)

    current_model_text = generated_model_text
    current_data_text = generated_data_text
    solver_feedback = ""
    history: List[Dict[str, Any]] = []

    last_patched_model_text = ""
    last_patched_data_text = ""
    last_solver_output = ""
    last_patched_obj: Optional[float] = None

    for attempt_idx in range(1, max_refinement_attempts + 1):
        attempt_dir = os.path.join(work_dir, f"attempt_{attempt_idx}")
        os.makedirs(attempt_dir, exist_ok=True)

        patched_model = regenerate_model_with_patch(
            cfg,
            system_prompt,
            current_model_text,
            current_data_text,
            description_text,
            patch_type,
            patch_text,
            solver_feedback=solver_feedback,
            attempt_number=attempt_idx,
        )

        patched_model_path = os.path.join(attempt_dir, "model_patched.mod")
        data_path = os.path.join(attempt_dir, "data.dat")
        patched_model_text2 = patched_model.get("model_mod", "")
        patched_data_text2 = patched_model.get("data_dat", "") or current_data_text

        _write_text(patched_model_path, patched_model_text2)
        _write_text(data_path, patched_data_text2)

        sol_text, _ = solve_with_amplpy(attempt_dir, patched_model_path, data_path)
        patched_obj = _parse_objective_from_solver_output(sol_text)

        last_patched_model_text = patched_model_text2
        last_patched_data_text = patched_data_text2
        last_solver_output = sol_text
        last_patched_obj = patched_obj

        if _is_solver_error(sol_text):
            history.append(
                {
                    "attempt": attempt_idx,
                    "status": "solver_error",
                    "patched_objective": patched_obj,
                    "solver_output": sol_text,
                }
            )
            solver_feedback = sol_text
            current_model_text = patched_model_text2
            current_data_text = patched_data_text2
            continue

        if patched_obj is None:
            history.append(
                {
                    "attempt": attempt_idx,
                    "status": "solver_error",
                    "patched_objective": None,
                    "solver_output": sol_text,
                }
            )
            solver_feedback = sol_text or "Error: solver completed without a parseable objective value."
            current_model_text = patched_model_text2
            current_data_text = patched_data_text2
            continue

        abs_diff = abs(patched_obj - ref_obj)
        denom = abs(ref_obj) if ref_obj != 0 else 1.0
        rel_diff = abs_diff / denom
        ok = (abs_diff <= abs_tol) or (rel_diff <= rel_tol)

        history.append(
            {
                "attempt": attempt_idx,
                "status": "patch_works" if ok else "patch_not_work",
                "patched_objective": patched_obj,
                "abs_diff": abs_diff,
                "rel_diff": rel_diff,
                "solver_output": sol_text,
            }
        )

        return PatchEvalResult(
            status="patch_works" if ok else "patch_not_work",
            reference_objective=ref_obj,
            patched_objective=patched_obj,
            objective_match=ok,
            abs_diff=abs_diff,
            rel_diff=rel_diff,
            patch_text=patch_text,
            solver_output=sol_text,
            attempts_used=attempt_idx,
            max_attempts=max_refinement_attempts,
            attempt_history=history,
            patched_model_text=patched_model_text2,
            patched_data_text=patched_data_text2,
        )

    return PatchEvalResult(
        status="solver_error_max_attempts",
        reference_objective=ref_obj,
        patched_objective=last_patched_obj,
        objective_match=None,
        abs_diff=None,
        rel_diff=None,
        patch_text=patch_text,
        solver_output=last_solver_output,
        attempts_used=max_refinement_attempts,
        max_attempts=max_refinement_attempts,
        attempt_history=history,
        patched_model_text=last_patched_model_text,
        patched_data_text=last_patched_data_text,
    )



def result_to_dict(r: PatchEvalResult) -> Dict[str, Any]:
    sol = r.solver_output or ""
    if len(sol) > 4000:
        sol = sol[:4000] + "\n... (truncated)"
    return {
        "status": r.status,
        "reference_objective": r.reference_objective,
        "patched_objective": r.patched_objective,
        "objective_match": r.objective_match,
        "abs_diff": r.abs_diff,
        "rel_diff": r.rel_diff,
        "patch_text": r.patch_text,
        "solver_output": sol,
        "attempts_used": r.attempts_used,
        "max_attempts": r.max_attempts,
        "attempt_history": r.attempt_history,
        "patched_model_text": r.patched_model_text,
        "patched_data_text": r.patched_data_text,
    }
