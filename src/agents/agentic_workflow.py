from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .ampl_output_parser import parse_solve_output
from .patch_eval import evaluate_patch, result_to_dict
from .schema_utils import load_schema, validate_json
from .llm_agent import AgentConfig, call_classifier, parse_json_strict, repair_json


_EXPERT_SCHEMA_TEXT = json.dumps(
    {
        "type": "object",
        "required": [
            "final_primary_label",
            "final_secondary_labels",
            "final_most_likely_root_cause",
            "review_summary",
            "confidence",
        ],
        "properties": {
            "final_primary_label": {"type": "string"},
            "final_secondary_labels": {"type": "array", "items": {"type": "string"}},
            "final_most_likely_root_cause": {"type": "string"},
            "review_summary": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "additionalProperties": False,
    },
    indent=2,
)


def _sanitize_record(obj: Dict[str, Any]) -> Dict[str, Any]:
    if "$schema" in obj:
        obj = dict(obj)
        obj.pop("$schema", None)
    return obj


@dataclass
class WorkflowConfig:
    model: str
    max_patch_refinement_attempts: int = 5
    max_analysis_cycles: int = 5
    abs_tol: float = 1e-6
    rel_tol: float = 1e-6


@dataclass
class PromptBundle:
    system_prompt: str
    patch_regen_system: str
    expert_review_system: str
    taxonomy: str
    schema_text: str
    task_template: str


@dataclass
class WorkflowResult:
    final_record: Dict[str, Any]
    initial_record: Dict[str, Any]
    patch_eval: Dict[str, Any]
    workflow_trace: Dict[str, Any]


class ErrorAnalyzerAgent:
    def __init__(self, cfg: WorkflowConfig, prompts: PromptBundle):
        self.cfg = cfg
        self.prompts = prompts
        self.validator = load_schema(prompts.schema_text)
        self.agent_cfg = AgentConfig(model=cfg.model)

    def analyze(
        self,
        *,
        run: Any,
        ref_stats: Dict[str, int],
        gen_stats: Dict[str, int],
        diff_stats: Dict[str, int],
        analysis_round: int,
        previous_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        user_prompt = self._build_user_prompt(
            run=run,
            ref_stats=ref_stats,
            gen_stats=gen_stats,
            diff_stats=diff_stats,
            analysis_round=analysis_round,
            previous_context=previous_context,
        )

        raw = call_classifier(self.agent_cfg, self.prompts.system_prompt, user_prompt)
        try:
            parsed = parse_json_strict(raw)
            parsed = _sanitize_record(parsed)
            validate_json(parsed, self.validator)
        except Exception:
            fixed = repair_json(self.agent_cfg, self.prompts.system_prompt, raw, self.prompts.schema_text)
            parsed = parse_json_strict(fixed)
            parsed = _sanitize_record(parsed)
            validate_json(parsed, self.validator)

        parsed.setdefault("problem_id", run.problem_id)
        parsed.setdefault("run_id", run.run_id)
        parsed["model_stats"] = {
            "reference": ref_stats,
            "generated": gen_stats,
            "diff": diff_stats,
        }
        parsed["analysis_round"] = analysis_round
        if previous_context:
            parsed["reanalysis_reason"] = previous_context.get("reason", "")
        return parsed

    def _build_user_prompt(
        self,
        *,
        run: Any,
        ref_stats: Dict[str, int],
        gen_stats: Dict[str, int],
        diff_stats: Dict[str, int],
        analysis_round: int,
        previous_context: Optional[Dict[str, Any]],
    ) -> str:
        ref_info = parse_solve_output(run.ref_output)
        gen_info = parse_solve_output(run.gen_output)

        rel: Optional[float] = None
        if ref_info.objective is not None and gen_info.objective is not None and ref_info.objective != 0:
            rel = abs(gen_info.objective - ref_info.objective) / abs(ref_info.objective)

        model_stats_block = (
            f"reference: {ref_stats}\n"
            f"generated: {gen_stats}\n"
            f"diff_gen_minus_ref: {diff_stats}\n"
        )

        prompt = self.prompts.task_template.format(
            taxonomy=self.prompts.taxonomy,
            schema=self.prompts.schema_text,
            problem_id=run.problem_id,
            run_id=run.run_id,
            model_stats_block=model_stats_block,
            ref_model=run.ref_model,
            ref_data=run.ref_data,
            ref_output=run.ref_output,
            gen_model=run.gen_model,
            gen_data=run.gen_data,
            gen_output=run.gen_output,
        )

        hint = (
            "\n\nNUMERIC_HINT\n"
            f"reference_status: {ref_info.status}\n"
            f"generated_status: {gen_info.status}\n"
            f"reference_objective: {ref_info.objective}\n"
            f"generated_objective: {gen_info.objective}\n"
            f"relative_error: {rel}\n"
            f"analysis_round: {analysis_round}\n"
        )

        if not previous_context:
            return prompt + hint

        retry_block = {
            "reason": previous_context.get("reason"),
            "previous_analysis": previous_context.get("previous_analysis"),
            "patch_eval": previous_context.get("patch_eval"),
        }

        extra = (
            "\n\nREANALYSIS_CONTEXT\n"
            "The previous patch attempt did not solve the problem. Reconsider the labels and propose a better minimal patch.\n"
            "Prefer revising the root-cause diagnosis if the prior patch produced the wrong objective.\n"
            + json.dumps(retry_block, indent=2)
        )
        return prompt + hint + extra


class PatchTesterAgent:
    def __init__(self, cfg: WorkflowConfig, prompts: PromptBundle):
        self.cfg = cfg
        self.prompts = prompts
        self.agent_cfg = AgentConfig(model=cfg.model)

    def evaluate(
        self,
        *,
        gen_run_dir: str,
        out_dir: str,
        run: Any,
        analysis_record: Dict[str, Any],
        analysis_round: int,
    ) -> Dict[str, Any]:
        minimal_patch = analysis_record.get("minimal_patch", {}) or {}
        patch_type = str(minimal_patch.get("patch_type", ""))
        patch_text = str(minimal_patch.get("patch_text", ""))

        eval_res = evaluate_patch(
            cfg=self.agent_cfg,
            system_prompt=self.prompts.patch_regen_system,
            gen_run_dir=gen_run_dir,
            out_dir=os.path.join(out_dir, f"analysis_round_{analysis_round}"),
            reference_output_text=run.ref_output,
            generated_model_text=run.gen_model,
            generated_data_text=run.gen_data,
            patch_type=patch_type,
            patch_text=patch_text,
            max_refinement_attempts=self.cfg.max_patch_refinement_attempts,
            abs_tol=self.cfg.abs_tol,
            rel_tol=self.cfg.rel_tol,
        )
        return result_to_dict(eval_res)


class ExpertReviewAgent:
    def __init__(self, cfg: WorkflowConfig, prompts: PromptBundle):
        self.cfg = cfg
        self.prompts = prompts
        self.agent_cfg = AgentConfig(model=cfg.model)
        self.validator = load_schema(_EXPERT_SCHEMA_TEXT)

    def review(
        self,
        *,
        run: Any,
        current_analysis: Dict[str, Any],
        patch_eval: Dict[str, Any],
    ) -> Dict[str, Any]:
        user_prompt = self._build_prompt(run=run, current_analysis=current_analysis, patch_eval=patch_eval)
        raw = call_classifier(self.agent_cfg, self.prompts.expert_review_system, user_prompt)
        try:
            parsed = parse_json_strict(raw)
            validate_json(parsed, self.validator)
        except Exception:
            fixed = repair_json(self.agent_cfg, self.prompts.expert_review_system, raw, _EXPERT_SCHEMA_TEXT)
            parsed = parse_json_strict(fixed)
            validate_json(parsed, self.validator)
        return parsed

    def _build_prompt(self, *, run: Any, current_analysis: Dict[str, Any], patch_eval: Dict[str, Any]) -> str:
        return (
            "Return JSON only.\n\n"
            "Select the final labels that best explain the actual cause of the error after considering the successful patch result.\n"
            "You may keep or revise the analyzer labels, but the final labels must be consistent with the reference model, generated model, patch, and patched solve result.\n"
            "Use the taxonomy when possible.\n\n"
            "Output schema:\n"
            + _EXPERT_SCHEMA_TEXT
            + "\n\nTAXONOMY\n"
            + self.prompts.taxonomy
            + "\n\nREFERENCE_MODEL\n```ampl\n"
            + run.ref_model
            + "\n```\n\nREFERENCE_DATA\n```ampl\n"
            + run.ref_data
            + "\n```\n\nREFERENCE_SOLVER_OUTPUT\n```\n"
            + run.ref_output
            + "\n```\n\nGENERATED_MODEL\n```ampl\n"
            + run.gen_model
            + "\n```\n\nGENERATED_DATA\n```ampl\n"
            + run.gen_data
            + "\n```\n\nGENERATED_SOLVER_OUTPUT\n```\n"
            + run.gen_output
            + "\n```\n\nCURRENT_ANALYSIS\n"
            + json.dumps(current_analysis, indent=2)
            + "\n\nPATCH_EVAL\n"
            + json.dumps(patch_eval, indent=2)
        )


class AgenticOrchestrator:
    def __init__(
        self,
        *,
        error_analyzer: ErrorAnalyzerAgent,
        patch_tester: PatchTesterAgent,
        expert_reviewer: ExpertReviewAgent,
        cfg: WorkflowConfig,
    ):
        self.error_analyzer = error_analyzer
        self.patch_tester = patch_tester
        self.expert_reviewer = expert_reviewer
        self.cfg = cfg

    def execute(
        self,
        *,
        run: Any,
        gen_run_dir: str,
        out_dir: str,
        ref_stats: Dict[str, int],
        gen_stats: Dict[str, int],
        diff_stats: Dict[str, int],
    ) -> WorkflowResult:
        analysis_history: List[Dict[str, Any]] = []
        patch_attempt_history: List[Dict[str, Any]] = []

        current_record = self.error_analyzer.analyze(
            run=run,
            ref_stats=ref_stats,
            gen_stats=gen_stats,
            diff_stats=diff_stats,
            analysis_round=1,
            previous_context=None,
        )
        initial_record = copy.deepcopy(current_record)
        analysis_history.append(self._analysis_summary(current_record))

        final_patch_eval: Dict[str, Any] = {
            "status": "not_run",
            "objective_match": None,
        }
        workflow_status = "analysis_only"

        for analysis_round in range(1, self.cfg.max_analysis_cycles + 1):
            current_record["analysis_round"] = analysis_round
            current_record["workflow_state"] = "patch_evaluation"

            patch_eval = self.patch_tester.evaluate(
                gen_run_dir=gen_run_dir,
                out_dir=out_dir,
                run=run,
                analysis_record=current_record,
                analysis_round=analysis_round,
            )
            patch_eval["analysis_round"] = analysis_round
            patch_attempt_history.append(
                {
                    "analysis_round": analysis_round,
                    "status": patch_eval.get("status"),
                    "attempts_used": patch_eval.get("attempts_used"),
                    "objective_match": patch_eval.get("objective_match"),
                    "patch_type": (current_record.get("minimal_patch", {}) or {}).get("patch_type", ""),
                    "patch_text": (current_record.get("minimal_patch", {}) or {}).get("patch_text", ""),
                }
            )
            final_patch_eval = patch_eval

            status = str(patch_eval.get("status", ""))
            if status == "patch_works":
                workflow_status = "completed"
                try:
                    review = self.expert_reviewer.review(
                        run=run,
                        current_analysis=current_record,
                        patch_eval=patch_eval,
                    )
                    current_record["expert_review"] = review
                    current_record["primary_label"] = review.get("final_primary_label", current_record.get("primary_label", ""))
                    current_record["secondary_labels"] = review.get("final_secondary_labels", current_record.get("secondary_labels", []))
                    current_record["most_likely_root_cause"] = review.get(
                        "final_most_likely_root_cause",
                        current_record.get("most_likely_root_cause", ""),
                    )
                    current_record["confidence"] = review.get("confidence", current_record.get("confidence", 0.0))
                except Exception as e:
                    current_record["expert_review_error"] = str(e)
                break

            if status == "patch_not_work" and analysis_round < self.cfg.max_analysis_cycles:
                current_record = self.error_analyzer.analyze(
                    run=run,
                    ref_stats=ref_stats,
                    gen_stats=gen_stats,
                    diff_stats=diff_stats,
                    analysis_round=analysis_round + 1,
                    previous_context={
                        "reason": "patched model executed but objective did not match reference",
                        "previous_analysis": current_record,
                        "patch_eval": patch_eval,
                    },
                )
                analysis_history.append(self._analysis_summary(current_record))
                continue

            if status in {"patch_not_work", "solver_error_max_attempts", "missing_reference_objective"}:
                workflow_status = status
                break

            workflow_status = status or "unknown"
            break

        current_record["workflow"] = {
            "status": workflow_status,
            "analysis_cycles_used": current_record.get("analysis_round", 1),
            "max_analysis_cycles": self.cfg.max_analysis_cycles,
            "patch_refinement_attempts_used": final_patch_eval.get("attempts_used", 0),
            "max_patch_refinement_attempts": self.cfg.max_patch_refinement_attempts,
            "expert_review_applied": workflow_status == "completed",
        }
        current_record["analysis_history"] = analysis_history
        current_record["latest_patch_eval"] = {
            "status": final_patch_eval.get("status"),
            "objective_match": final_patch_eval.get("objective_match"),
            "attempts_used": final_patch_eval.get("attempts_used"),
        }

        workflow_trace = {
            "workflow_status": workflow_status,
            "analysis_history": analysis_history,
            "patch_attempt_history": patch_attempt_history,
        }

        return WorkflowResult(
            final_record=current_record,
            initial_record=initial_record,
            patch_eval=final_patch_eval,
            workflow_trace=workflow_trace,
        )

    @staticmethod
    def _analysis_summary(record: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "analysis_round": record.get("analysis_round"),
            "primary_label": record.get("primary_label"),
            "secondary_labels": record.get("secondary_labels", []),
            "patch_type": (record.get("minimal_patch", {}) or {}).get("patch_type", ""),
            "patch_text": (record.get("minimal_patch", {}) or {}).get("patch_text", ""),
        }
