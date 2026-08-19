from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, Tuple

from .fs_utils import load_run
from .model_stats import compute_model_stats, stats_to_dict
from .render import render_markdown
from .agentic_workflow import (
    AgenticOrchestrator,
    ErrorAnalyzerAgent,
    ExpertReviewAgent,
    PatchTesterAgent,
    PromptBundle,
    WorkflowConfig,
)



def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()



def _compute_stats_pair(
    ref_model_text: str, gen_model_text: str
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    ref_stats = stats_to_dict(compute_model_stats(ref_model_text))
    gen_stats = stats_to_dict(compute_model_stats(gen_model_text))
    keys = sorted(set(ref_stats.keys()) | set(gen_stats.keys()))
    diff = {k: int(gen_stats.get(k, 0)) - int(ref_stats.get(k, 0)) for k in keys}
    return ref_stats, gen_stats, diff



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_run_dir", required=True)
    ap.add_argument("--ref_solution_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--problem_id", default="")
    ap.add_argument("--run_id", default="")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max_patch_refinement_attempts", type=int, default=5)
    ap.add_argument("--max_analysis_cycles", type=int, default=5)

    args = ap.parse_args()

    problem_id = args.problem_id or os.path.basename(os.path.dirname(args.gen_run_dir.rstrip("/")))
    run_id = args.run_id or os.path.basename(args.gen_run_dir.rstrip("/"))

    os.makedirs(args.out_dir, exist_ok=True)

    run = load_run(args.gen_run_dir, args.ref_solution_dir, problem_id, run_id)

    prompts_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "prompts"))
    prompts = PromptBundle(
        system_prompt=_read(os.path.join(prompts_dir, "system.txt")),
        patch_regen_system=_read(os.path.join(prompts_dir, "patch_regen_system.txt")),
        expert_review_system=_read(os.path.join(prompts_dir, "expert_review_system.txt")),
        taxonomy=_read(os.path.join(prompts_dir, "taxonomy.txt")),
        schema_text=_read(os.path.join(prompts_dir, "output_schema.json")),
        task_template=_read(os.path.join(prompts_dir, "task_template.txt")),
    )

    ref_stats, gen_stats, diff_stats = _compute_stats_pair(run.ref_model, run.gen_model)
    with open(os.path.join(args.out_dir, "model_stats.json"), "w", encoding="utf-8") as f:
        json.dump({"reference": ref_stats, "generated": gen_stats, "diff": diff_stats}, f, indent=2)

    cfg = WorkflowConfig(
        model=args.model,
        max_patch_refinement_attempts=args.max_patch_refinement_attempts,
        max_analysis_cycles=args.max_analysis_cycles,
    )

    orchestrator = AgenticOrchestrator(
        error_analyzer=ErrorAnalyzerAgent(cfg, prompts),
        patch_tester=PatchTesterAgent(cfg, prompts),
        expert_reviewer=ExpertReviewAgent(cfg, prompts),
        cfg=cfg,
    )

    workflow_res = orchestrator.execute(
        run=run,
        gen_run_dir=args.gen_run_dir,
        out_dir=args.out_dir,
        ref_stats=ref_stats,
        gen_stats=gen_stats,
        diff_stats=diff_stats,
    )

    parsed = workflow_res.final_record
    initial_record = workflow_res.initial_record
    eval_dict = workflow_res.patch_eval
    workflow_trace = workflow_res.workflow_trace

    with open(os.path.join(args.out_dir, "initial_record.json"), "w", encoding="utf-8") as f:
        json.dump(initial_record, f, indent=2)

    with open(os.path.join(args.out_dir, "record.json"), "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2)

    with open(os.path.join(args.out_dir, "workflow_trace.json"), "w", encoding="utf-8") as f:
        json.dump(workflow_trace, f, indent=2)

    md = render_markdown(parsed)
    with open(os.path.join(args.out_dir, "record.md"), "w", encoding="utf-8") as f:
        f.write(md)

    csv_path = os.path.join(args.out_dir, "record.csv")
    status_ref = str(parsed.get("status", {}).get("reference", ""))
    status_gen = str(parsed.get("status", {}).get("generated", ""))
    obj_ref = parsed.get("objective", {}).get("reference", None)
    obj_gen = parsed.get("objective", {}).get("generated", None)
    rel_err = parsed.get("objective", {}).get("relative_error", None)
    primary = str(parsed.get("primary_label", ""))
    secondary = parsed.get("secondary_labels", []) or []
    patch_obj = parsed.get("minimal_patch", {}) or {}
    patch_type = str(patch_obj.get("patch_type", ""))
    patch_text = str(patch_obj.get("patch_text", ""))

    expert_review = parsed.get("expert_review", {}) or {}
    row = {
        "problem_id": problem_id,
        "run_id": run_id,
        "reference_status": status_ref,
        "generated_status": status_gen,
        "reference_objective": obj_ref,
        "generated_objective": obj_gen,
        "relative_error": rel_err,
        "primary_label": primary,
        "secondary_labels": ";".join([str(x) for x in secondary]),
        "patch_type": patch_type,
        "patch_text": patch_text,
        "workflow_status": (parsed.get("workflow", {}) or {}).get("status", ""),
        "analysis_cycles_used": (parsed.get("workflow", {}) or {}).get("analysis_cycles_used", 0),
        "patch_refinement_attempts_used": (parsed.get("workflow", {}) or {}).get("patch_refinement_attempts_used", 0),
        "expert_final_primary_label": expert_review.get("final_primary_label", ""),
        "expert_review_confidence": expert_review.get("confidence", None),
        "ref_n_sets": ref_stats.get("n_sets"),
        "ref_n_params": ref_stats.get("n_params"),
        "ref_n_vars": ref_stats.get("n_vars"),
        "ref_n_objectives": ref_stats.get("n_objectives"),
        "ref_n_constraints_decl": ref_stats.get("n_constraints_decl"),
        "gen_n_sets": gen_stats.get("n_sets"),
        "gen_n_params": gen_stats.get("n_params"),
        "gen_n_vars": gen_stats.get("n_vars"),
        "gen_n_objectives": gen_stats.get("n_objectives"),
        "gen_n_constraints_decl": gen_stats.get("n_constraints_decl"),
        "diff_n_sets": diff_stats.get("n_sets"),
        "diff_n_params": diff_stats.get("n_params"),
        "diff_n_vars": diff_stats.get("n_vars"),
        "diff_n_objectives": diff_stats.get("n_objectives"),
        "diff_n_constraints_decl": diff_stats.get("n_constraints_decl"),
    }

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)

    patch_eval_path = os.path.join(args.out_dir, "patch_eval.json")
    patch_eval_csv_path = os.path.join(args.out_dir, "patch_eval.csv")
    with open(patch_eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_dict, f, indent=2)

    eval_row = {
        "problem_id": problem_id,
        "run_id": run_id,
        "primary_label": primary,
        "patch_type": patch_type,
        "patch_text": patch_text,
        "reference_objective": eval_dict.get("reference_objective"),
        "patched_objective": eval_dict.get("patched_objective"),
        "objective_match": eval_dict.get("objective_match"),
        "abs_diff": eval_dict.get("abs_diff"),
        "rel_diff": eval_dict.get("rel_diff"),
        "status": eval_dict.get("status"),
        "solver_output": eval_dict.get("solver_output"),
        "attempts_used": eval_dict.get("attempts_used", 0),
        "max_attempts": eval_dict.get("max_attempts", 0),
    }
    with open(patch_eval_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(eval_row.keys()))
        w.writeheader()
        w.writerow(eval_row)


if __name__ == "__main__":
    main()
