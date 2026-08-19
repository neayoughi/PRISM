"""LangGraph evaluator-optimizer workflow for PRISM."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from .agentic_workflow import (
    ErrorAnalyzerAgent,
    ExpertReviewAgent,
    PatchTesterAgent,
    PromptBundle,
    WorkflowConfig,
    WorkflowResult,
)
from .ampl_output_parser import parse_solve_output


RouteName = Literal["expert_review", "prepare_reanalysis", "finalize"]


class WorkflowState(TypedDict, total=False):
    """State carried between graph nodes."""

    # Inputs and configuration
    run: Any
    gen_run_dir: str
    out_dir: str
    ref_stats: Dict[str, int]
    gen_stats: Dict[str, int]
    diff_stats: Dict[str, int]
    max_analysis_cycles: int
    max_patch_refinement_attempts: int

    # Problem/run and model information
    problem_id: str
    run_id: str
    generated_model_path: str
    generated_model_content: str
    generated_data_content: str
    generated_solver_output: str
    reference_model_content: str
    reference_data_content: str
    reference_solver_output: str
    reference_objective: Optional[float]

    # Evaluator-optimizer loop state
    analysis_round: int
    diagnosis_record: Dict[str, Any]
    initial_record: Dict[str, Any]
    previous_context: Optional[Dict[str, Any]]
    analysis_history: List[Dict[str, Any]]
    patch_attempt_history: List[Dict[str, Any]]
    patch_evaluation_status: str
    patch_eval: Dict[str, Any]
    solver_output: str
    error_details: Optional[str]
    objective_comparison_result: Dict[str, Any]

    # Review, tracing, and result
    expert_review_result: Dict[str, Any]
    workflow_status: str
    workflow_trace: List[str]
    final_result: WorkflowResult


def route_after_evaluation(state: WorkflowState) -> RouteName:
    """Choose the next graph node from patch status and cycle count."""

    status = str(
        state.get("patch_evaluation_status")
        or state.get("patch_eval", {}).get("status", "")
    )
    if status == "patch_works":
        return "expert_review"
    if status == "patch_not_work" and int(state.get("analysis_round", 1)) < int(
        state.get("max_analysis_cycles", 1)
    ):
        return "prepare_reanalysis"
    return "finalize"


class AgenticOrchestrator:
    """LangGraph orchestrator with the same public interface as the runner."""

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
        self.graph_builder = self._build_graph()
        self.graph = self.graph_builder.compile()

    def _build_graph(self) -> StateGraph:
        builder = StateGraph(WorkflowState)
        builder.add_node("initialize_workflow", self._initialize_workflow)
        builder.add_node("analyze_error", self._analyze_error)
        builder.add_node("evaluate_patch", self._evaluate_patch)
        builder.add_node("prepare_reanalysis", self._prepare_reanalysis)
        builder.add_node("expert_review", self._expert_review)
        builder.add_node("finalize", self._finalize)

        builder.add_edge(START, "initialize_workflow")
        builder.add_edge("initialize_workflow", "analyze_error")
        builder.add_edge("analyze_error", "evaluate_patch")
        builder.add_conditional_edges(
            "evaluate_patch",
            route_after_evaluation,
            {
                "expert_review": "expert_review",
                "prepare_reanalysis": "prepare_reanalysis",
                "finalize": "finalize",
            },
        )
        builder.add_edge("prepare_reanalysis", "analyze_error")
        builder.add_edge("expert_review", "finalize")
        builder.add_edge("finalize", END)
        return builder

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
        initial_state: WorkflowState = {
            "run": run,
            "gen_run_dir": str(Path(gen_run_dir)),
            "out_dir": str(Path(out_dir)),
            "ref_stats": ref_stats,
            "gen_stats": gen_stats,
            "diff_stats": diff_stats,
            "max_analysis_cycles": self.cfg.max_analysis_cycles,
            "max_patch_refinement_attempts": self.cfg.max_patch_refinement_attempts,
        }
        final_state = self.graph.invoke(
            initial_state,
            config={"recursion_limit": max(25, self.cfg.max_analysis_cycles * 4 + 10)},
        )
        return final_state["final_result"]

    def _initialize_workflow(self, state: WorkflowState) -> WorkflowState:
        run = state["run"]
        gen_dir = Path(state["gen_run_dir"])
        generated_model_path = self._generated_model_path(gen_dir)
        reference_objective = parse_solve_output(run.ref_output).objective
        return {
            "problem_id": str(run.problem_id),
            "run_id": str(run.run_id),
            "generated_model_path": str(generated_model_path),
            "generated_model_content": run.gen_model,
            "generated_data_content": run.gen_data,
            "generated_solver_output": run.gen_output,
            "reference_model_content": run.ref_model,
            "reference_data_content": run.ref_data,
            "reference_solver_output": run.ref_output,
            "reference_objective": reference_objective,
            "analysis_round": 1,
            "previous_context": None,
            "analysis_history": [],
            "patch_attempt_history": [],
            "patch_evaluation_status": "not_run",
            "patch_eval": {"status": "not_run", "objective_match": None},
            "solver_output": "",
            "error_details": None,
            "objective_comparison_result": {
                "reference_objective": reference_objective,
                "patched_objective": None,
                "objective_match": None,
                "abs_diff": None,
                "rel_diff": None,
            },
            "expert_review_result": {},
            "workflow_status": "analysis_only",
            "workflow_trace": ["initialize_workflow"],
        }

    def _analyze_error(self, state: WorkflowState) -> WorkflowState:
        record = self.error_analyzer.analyze(
            run=state["run"],
            ref_stats=state["ref_stats"],
            gen_stats=state["gen_stats"],
            diff_stats=state["diff_stats"],
            analysis_round=state["analysis_round"],
            previous_context=state.get("previous_context"),
        )
        history = list(state.get("analysis_history", []))
        history.append(self._analysis_summary(record))
        updates: WorkflowState = {
            "diagnosis_record": record,
            "analysis_history": history,
            "workflow_trace": [*state.get("workflow_trace", []), "analyze_error"],
        }
        if "initial_record" not in state:
            updates["initial_record"] = copy.deepcopy(record)
        return updates

    def _evaluate_patch(self, state: WorkflowState) -> WorkflowState:
        analysis_round = state["analysis_round"]
        record = copy.deepcopy(state["diagnosis_record"])
        record["analysis_round"] = analysis_round
        record["workflow_state"] = "patch_evaluation"

        patch_eval = self.patch_tester.evaluate(
            gen_run_dir=state["gen_run_dir"],
            out_dir=state["out_dir"],
            run=state["run"],
            analysis_record=record,
            analysis_round=analysis_round,
        )
        patch_eval = dict(patch_eval)
        patch_eval["analysis_round"] = analysis_round
        status = str(patch_eval.get("status", ""))
        minimal_patch = record.get("minimal_patch", {}) or {}
        attempt_history = list(state.get("patch_attempt_history", []))
        attempt_history.append(
            {
                "analysis_round": analysis_round,
                "status": patch_eval.get("status"),
                "attempts_used": patch_eval.get("attempts_used"),
                "objective_match": patch_eval.get("objective_match"),
                "patch_type": minimal_patch.get("patch_type", ""),
                "patch_text": minimal_patch.get("patch_text", ""),
            }
        )
        error_details: Optional[str] = None
        if status in {"solver_error_max_attempts", "missing_reference_objective"}:
            error_details = str(patch_eval.get("solver_output", "") or status)
        return {
            "diagnosis_record": record,
            "patch_eval": patch_eval,
            "patch_evaluation_status": status,
            "patch_attempt_history": attempt_history,
            "solver_output": str(patch_eval.get("solver_output", "") or ""),
            "error_details": error_details,
            "objective_comparison_result": {
                "reference_objective": patch_eval.get("reference_objective"),
                "patched_objective": patch_eval.get("patched_objective"),
                "objective_match": patch_eval.get("objective_match"),
                "abs_diff": patch_eval.get("abs_diff"),
                "rel_diff": patch_eval.get("rel_diff"),
            },
            "workflow_trace": [*state.get("workflow_trace", []), "evaluate_patch"],
        }

    def _prepare_reanalysis(self, state: WorkflowState) -> WorkflowState:
        return {
            "analysis_round": state["analysis_round"] + 1,
            "previous_context": {
                "reason": "patched model executed but objective did not match reference",
                "previous_analysis": state["diagnosis_record"],
                "patch_eval": state["patch_eval"],
            },
            "workflow_trace": [
                *state.get("workflow_trace", []),
                "prepare_reanalysis",
            ],
        }

    def _expert_review(self, state: WorkflowState) -> WorkflowState:
        record = copy.deepcopy(state["diagnosis_record"])
        review: Dict[str, Any] = {}
        try:
            review = self.expert_reviewer.review(
                run=state["run"],
                current_analysis=record,
                patch_eval=state["patch_eval"],
            )
            record["expert_review"] = review
            record["primary_label"] = review.get(
                "final_primary_label", record.get("primary_label", "")
            )
            record["secondary_labels"] = review.get(
                "final_secondary_labels", record.get("secondary_labels", [])
            )
            record["most_likely_root_cause"] = review.get(
                "final_most_likely_root_cause",
                record.get("most_likely_root_cause", ""),
            )
            record["confidence"] = review.get(
                "confidence", record.get("confidence", 0.0)
            )
        except Exception as exc:
            record["expert_review_error"] = str(exc)
        return {
            "diagnosis_record": record,
            "expert_review_result": review,
            "workflow_status": "completed",
            "workflow_trace": [*state.get("workflow_trace", []), "expert_review"],
        }

    def _finalize(self, state: WorkflowState) -> WorkflowState:
        record = copy.deepcopy(state["diagnosis_record"])
        patch_eval = state.get(
            "patch_eval", {"status": "not_run", "objective_match": None}
        )
        workflow_status = state.get("workflow_status", "analysis_only")
        if workflow_status != "completed":
            workflow_status = str(patch_eval.get("status", "")) or "unknown"

        record["workflow"] = {
            "status": workflow_status,
            "analysis_cycles_used": record.get("analysis_round", 1),
            "max_analysis_cycles": self.cfg.max_analysis_cycles,
            "patch_refinement_attempts_used": patch_eval.get("attempts_used", 0),
            "max_patch_refinement_attempts": self.cfg.max_patch_refinement_attempts,
            "expert_review_applied": workflow_status == "completed",
        }
        record["analysis_history"] = state.get("analysis_history", [])
        record["latest_patch_eval"] = {
            "status": patch_eval.get("status"),
            "objective_match": patch_eval.get("objective_match"),
            "attempts_used": patch_eval.get("attempts_used"),
        }
        graph_trace = [*state.get("workflow_trace", []), "finalize"]
        trace = {
            "workflow_status": workflow_status,
            "analysis_history": state.get("analysis_history", []),
            "patch_attempt_history": state.get("patch_attempt_history", []),
            "graph_nodes": graph_trace,
        }
        result = WorkflowResult(
            final_record=record,
            initial_record=state["initial_record"],
            patch_eval=patch_eval,
            workflow_trace=trace,
        )
        return {
            "diagnosis_record": record,
            "workflow_status": workflow_status,
            "workflow_trace": graph_trace,
            "final_result": result,
        }

    @staticmethod
    def _analysis_summary(record: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "analysis_round": record.get("analysis_round"),
            "primary_label": record.get("primary_label"),
            "secondary_labels": record.get("secondary_labels", []),
            "patch_type": (record.get("minimal_patch", {}) or {}).get(
                "patch_type", ""
            ),
            "patch_text": (record.get("minimal_patch", {}) or {}).get(
                "patch_text", ""
            ),
        }

    @staticmethod
    def _generated_model_path(gen_dir: Path) -> Path:
        attempts = []
        for path in gen_dir.glob("model-debug_attempt_*.mod"):
            try:
                attempts.append((int(path.stem.rsplit("_", 1)[-1]), path))
            except ValueError:
                continue
        return max(attempts, default=(-1, gen_dir / "model.mod"))[1]
