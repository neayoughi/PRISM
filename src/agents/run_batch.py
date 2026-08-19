from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from typing import Tuple

from .fs_utils import discover_runs



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_root", required=True)
    ap.add_argument("--ref_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--max_patch_refinement_attempts", type=int, default=5)
    ap.add_argument("--max_analysis_cycles", type=int, default=5)
    ap.add_argument("--fail_fast", action="store_true", help="Stop on first nonzero exit code.")
    args = ap.parse_args()

    runs = discover_runs(args.gen_root, args.ref_root)
    os.makedirs(args.out_root, exist_ok=True)

    summary_csv = os.path.join(args.out_root, "batch_report.csv")
    patch_csv = os.path.join(args.out_root, "batch_patch_eval.csv")

    summary_fields = [
        "problem_id",
        "run_id",
        "reference_status",
        "generated_status",
        "reference_objective",
        "generated_objective",
        "relative_error",
        "primary_label",
        "secondary_labels",
        "patch_type",
        "patch_text",
        "workflow_status",
        "analysis_cycles_used",
        "patch_refinement_attempts_used",
        "expert_final_primary_label",
        "expert_review_confidence",
        "ref_n_sets",
        "ref_n_params",
        "ref_n_vars",
        "ref_n_objectives",
        "ref_n_constraints_decl",
        "gen_n_sets",
        "gen_n_params",
        "gen_n_vars",
        "gen_n_objectives",
        "gen_n_constraints_decl",
        "diff_n_sets",
        "diff_n_params",
        "diff_n_vars",
        "diff_n_objectives",
        "diff_n_constraints_decl",
    ]
    patch_fields = [
        "problem_id",
        "run_id",
        "primary_label",
        "patch_type",
        "patch_text",
        "reference_objective",
        "patched_objective",
        "objective_match",
        "abs_diff",
        "rel_diff",
        "status",
        "solver_output",
        "attempts_used",
        "max_attempts",
    ]

    ok = 0
    fail = 0
    failed: list[Tuple[str, str]] = []

    with open(summary_csv, "w", newline="", encoding="utf-8") as summary_fp, open(
        patch_csv, "w", newline="", encoding="utf-8"
    ) as patch_fp:
        summary_writer = csv.DictWriter(summary_fp, fieldnames=summary_fields)
        patch_writer = csv.DictWriter(patch_fp, fieldnames=patch_fields)
        summary_writer.writeheader()
        patch_writer.writeheader()

        for problem_id, run_id, gen_run_dir, ref_solution_dir in runs:
            out_dir = os.path.join(args.out_root, problem_id, run_id)
            os.makedirs(out_dir, exist_ok=True)

            cmd = [
                sys.executable,
                "-m",
                "agents.base_run_one",
                "--gen_run_dir",
                gen_run_dir,
                "--ref_solution_dir",
                ref_solution_dir,
                "--out_dir",
                out_dir,
                "--problem_id",
                problem_id,
                "--run_id",
                run_id,
                "--model",
                args.model,
                "--max_patch_refinement_attempts",
                str(args.max_patch_refinement_attempts),
                "--max_analysis_cycles",
                str(args.max_analysis_cycles),
            ]

            p = subprocess.run(cmd, check=False)
            if p.returncode == 0:
                ok += 1

                record_json = os.path.join(out_dir, "record.json")
                patch_json = os.path.join(out_dir, "patch_eval.json")
                rec = None
                try:
                    with open(record_json, "r", encoding="utf-8") as f:
                        rec = json.load(f)
                    ms = rec.get("model_stats", {})
                    ref_ms = ms.get("reference", {})
                    gen_ms = ms.get("generated", {})
                    diff_ms = ms.get("diff", {})
                    workflow = rec.get("workflow", {}) or {}
                    expert_review = rec.get("expert_review", {}) or {}
                    row = {
                        "problem_id": problem_id,
                        "run_id": run_id,
                        "reference_status": rec.get("status", {}).get("reference", ""),
                        "generated_status": rec.get("status", {}).get("generated", ""),
                        "reference_objective": rec.get("objective", {}).get("reference", None),
                        "generated_objective": rec.get("objective", {}).get("generated", None),
                        "relative_error": rec.get("objective", {}).get("relative_error", None),
                        "primary_label": rec.get("primary_label", ""),
                        "secondary_labels": ";".join([str(x) for x in (rec.get("secondary_labels", []) or [])]),
                        "patch_type": rec.get("minimal_patch", {}).get("patch_type", ""),
                        "patch_text": rec.get("minimal_patch", {}).get("patch_text", ""),
                        "workflow_status": workflow.get("status", ""),
                        "analysis_cycles_used": workflow.get("analysis_cycles_used", 0),
                        "patch_refinement_attempts_used": workflow.get("patch_refinement_attempts_used", 0),
                        "expert_final_primary_label": expert_review.get("final_primary_label", ""),
                        "expert_review_confidence": expert_review.get("confidence", None),
                        "ref_n_sets": ref_ms.get("n_sets"),
                        "ref_n_params": ref_ms.get("n_params"),
                        "ref_n_vars": ref_ms.get("n_vars"),
                        "ref_n_objectives": ref_ms.get("n_objectives"),
                        "ref_n_constraints_decl": ref_ms.get("n_constraints_decl"),
                        "gen_n_sets": gen_ms.get("n_sets"),
                        "gen_n_params": gen_ms.get("n_params"),
                        "gen_n_vars": gen_ms.get("n_vars"),
                        "gen_n_objectives": gen_ms.get("n_objectives"),
                        "gen_n_constraints_decl": gen_ms.get("n_constraints_decl"),
                        "diff_n_sets": diff_ms.get("n_sets"),
                        "diff_n_params": diff_ms.get("n_params"),
                        "diff_n_vars": diff_ms.get("n_vars"),
                        "diff_n_objectives": diff_ms.get("n_objectives"),
                        "diff_n_constraints_decl": diff_ms.get("n_constraints_decl"),
                    }
                    summary_writer.writerow(row)
                    summary_fp.flush()
                except Exception:
                    pass

                try:
                    with open(patch_json, "r", encoding="utf-8") as f:
                        pr = json.load(f)
                    prow = {
                        "problem_id": problem_id,
                        "run_id": run_id,
                        "primary_label": (rec or {}).get("primary_label", ""),
                        "patch_type": ((rec or {}).get("minimal_patch", {}) or {}).get("patch_type", ""),
                        "patch_text": ((rec or {}).get("minimal_patch", {}) or {}).get("patch_text", ""),
                        "reference_objective": pr.get("reference_objective", None),
                        "patched_objective": pr.get("patched_objective", None),
                        "objective_match": pr.get("objective_match", None),
                        "abs_diff": pr.get("abs_diff", None),
                        "rel_diff": pr.get("rel_diff", None),
                        "status": pr.get("status", ""),
                        "solver_output": pr.get("solver_output", ""),
                        "attempts_used": pr.get("attempts_used", 0),
                        "max_attempts": pr.get("max_attempts", 0),
                    }
                    patch_writer.writerow(prow)
                    patch_fp.flush()
                except Exception:
                    pass
            else:
                fail += 1
                failed.append((problem_id, run_id))
                if args.fail_fast:
                    break

    print(f"Batch finished. ok={ok} fail={fail} total={ok+fail}")
    if failed:
        print("Failed runs:")
        for pid, rid in failed:
            print(f"- {pid}/{rid}")


if __name__ == "__main__":
    main()
