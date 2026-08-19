from __future__ import annotations

from typing import Any, Dict, List



def _fmt_stats_block(title: str, stats: Dict[str, Any]) -> List[str]:
    keys = ["n_sets", "n_params", "n_vars", "n_objectives", "n_constraints_decl"]
    parts: List[str] = []
    parts.append(f"- {title}: " + ", ".join([f"{k}={stats.get(k)}" for k in keys]))
    return parts



def render_markdown(record: Dict[str, Any]) -> str:
    pid = record.get("problem_id", "")
    rid = record.get("run_id", "")
    primary = record.get("primary_label", "")
    secondary = record.get("secondary_labels", [])
    status = record.get("status", {})
    obj = record.get("objective", {})
    model_stats = record.get("model_stats", {})
    workflow = record.get("workflow", {}) or {}
    expert_review = record.get("expert_review", {}) or {}

    lines: List[str] = []
    lines.append("# Qualitative error record")
    lines.append("")
    lines.append(f"Problem: **{pid}**")
    lines.append(f"Run: **{rid}**")
    lines.append("")

    lines.append("## Workflow")
    lines.append(f"- Status: {workflow.get('status')}")
    lines.append(f"- Analysis cycles used: {workflow.get('analysis_cycles_used')}")
    lines.append(f"- Patch refinement attempts used: {workflow.get('patch_refinement_attempts_used')}")
    lines.append(f"- Expert review applied: {workflow.get('expert_review_applied')}")
    lines.append("")

    lines.append("## Status")
    lines.append(f"- Reference: {status.get('reference')}")
    lines.append(f"- Generated: {status.get('generated')}")
    lines.append("")

    lines.append("## Objective")
    lines.append(f"- Reference: {obj.get('reference')}")
    lines.append(f"- Generated: {obj.get('generated')}")
    lines.append(f"- Relative error: {obj.get('relative_error')}")
    lines.append("")

    lines.append("## Model statistics")
    ref_s = model_stats.get("reference", {})
    gen_s = model_stats.get("generated", {})
    diff_s = model_stats.get("diff", {})
    if ref_s and gen_s and diff_s:
        lines.extend(_fmt_stats_block("Reference", ref_s))
        lines.extend(_fmt_stats_block("Generated", gen_s))
        lines.extend(_fmt_stats_block("Diff (gen - ref)", diff_s))
    else:
        lines.append("(not available)")
    lines.append("")

    lines.append("## Final labels")
    lines.append(f"- Primary: **{primary}**")
    if secondary:
        lines.append("- Secondary: " + ", ".join([str(x) for x in secondary]))
    else:
        lines.append("- Secondary: (none)")
    lines.append("")

    lines.append("## Most likely root cause")
    lines.append(record.get("most_likely_root_cause", ""))
    lines.append("")

    lines.append("## Key differences")
    diffs = record.get("differences", [])
    if not diffs:
        lines.append("(none)")
    for d in diffs[:6]:
        lines.append(f"- **{d.get('category','')}**: {d.get('summary','')}")
        ev = d.get("evidence", [])
        for e in ev[:3]:
            lines.append(f"  - {e}")
        imp = d.get("impact")
        if imp:
            lines.append(f"  - Impact: {imp}")
    lines.append("")

    lines.append("## Minimal patch")
    mp = record.get("minimal_patch", {})
    lines.append(f"Patch type: {mp.get('patch_type','')}")
    lines.append("")
    lines.append("```")
    lines.append(mp.get("patch_text", ""))
    lines.append("```")
    lines.append("")

    if expert_review:
        lines.append("## Expert review")
        lines.append(f"- Final primary label: {expert_review.get('final_primary_label')}")
        sec = expert_review.get("final_secondary_labels", []) or []
        if sec:
            lines.append("- Final secondary labels: " + ", ".join([str(x) for x in sec]))
        else:
            lines.append("- Final secondary labels: (none)")
        lines.append(f"- Review summary: {expert_review.get('review_summary', '')}")
        lines.append(f"- Confidence: {expert_review.get('confidence')}")
        lines.append("")

    lines.append("## Confidence")
    lines.append(str(record.get("confidence", "")))
    lines.append("")

    return "\n".join(lines)
