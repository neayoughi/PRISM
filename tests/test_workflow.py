from __future__ import annotations

import unittest
from unittest.mock import patch
from dataclasses import dataclass

from agents.agentic_workflow import WorkflowConfig
from agents.run_one import _prefer_current_python_on_path
from agents.workflow import (
    AgenticOrchestrator,
    route_after_evaluation,
)


@dataclass
class DummyRun:
    problem_id: str = "p1"
    run_id: str = "r1"
    ref_model: str = "ref model"
    ref_data: str = "ref data"
    ref_output: str = "objective 10 optimal solution"
    gen_model: str = "gen model"
    gen_data: str = "gen data"
    gen_output: str = "objective 8 optimal solution"


class StubErrorAnalyzer:
    def __init__(self) -> None:
        self.calls = 0

    def analyze(self, **kwargs):
        self.calls += 1
        return {
            "problem_id": "p1",
            "run_id": "r1",
            "status": {"reference": "optimal", "generated": "optimal"},
            "objective": {"reference": 10, "generated": 8, "relative_error": 0.2},
            "primary_label": "FIRST" if self.calls == 1 else "SECOND",
            "secondary_labels": [],
            "differences": [],
            "most_likely_root_cause": "diagnosis",
            "minimal_patch": {
                "patch_type": "replace",
                "patch_text": f"patch {self.calls}",
            },
            "confidence": 0.8,
            "analysis_round": kwargs["analysis_round"],
        }


class StubPatchTester:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, **kwargs):
        self.calls += 1
        works = self.calls == 2
        return {
            "status": "patch_works" if works else "patch_not_work",
            "objective_match": works,
            "attempts_used": 1,
            "reference_objective": 10,
            "patched_objective": 10 if works else 9,
            "abs_diff": 0 if works else 1,
            "rel_diff": 0 if works else 0.1,
            "solver_output": "objective_value: 10" if works else "objective_value: 9",
        }


class StubExpertReviewer:
    def review(self, **kwargs):
        return {
            "final_primary_label": "FINAL",
            "final_secondary_labels": ["SECOND"],
            "final_most_likely_root_cause": "reviewed cause",
            "review_summary": "successful patch",
            "confidence": 0.95,
        }


class RoutingTests(unittest.TestCase):
    def test_patch_works_routes_to_expert_review(self) -> None:
        self.assertEqual(
            route_after_evaluation(
                {"patch_evaluation_status": "patch_works", "analysis_round": 1, "max_analysis_cycles": 5}
            ),
            "expert_review",
        )

    def test_patch_not_work_with_cycles_remaining_routes_to_reanalysis(self) -> None:
        self.assertEqual(
            route_after_evaluation(
                {"patch_evaluation_status": "patch_not_work", "analysis_round": 1, "max_analysis_cycles": 2}
            ),
            "prepare_reanalysis",
        )

    def test_patch_not_work_at_limit_routes_to_finalize(self) -> None:
        self.assertEqual(
            route_after_evaluation(
                {"patch_evaluation_status": "patch_not_work", "analysis_round": 2, "max_analysis_cycles": 2}
            ),
            "finalize",
        )

    def test_solver_error_limit_routes_to_finalize(self) -> None:
        self.assertEqual(
            route_after_evaluation(
                {"patch_evaluation_status": "solver_error_max_attempts", "analysis_round": 1, "max_analysis_cycles": 5}
            ),
            "finalize",
        )

    def test_missing_reference_objective_routes_to_finalize(self) -> None:
        self.assertEqual(
            route_after_evaluation(
                {"patch_evaluation_status": "missing_reference_objective", "analysis_round": 1, "max_analysis_cycles": 5}
            ),
            "finalize",
        )


class GraphSmokeTests(unittest.TestCase):
    def make_orchestrator(self) -> AgenticOrchestrator:
        return AgenticOrchestrator(
            error_analyzer=StubErrorAnalyzer(),
            patch_tester=StubPatchTester(),
            expert_reviewer=StubExpertReviewer(),
            cfg=WorkflowConfig(model="dummy", max_analysis_cycles=5),
        )

    def test_graph_compiles_without_api_keys(self) -> None:
        graph = self.make_orchestrator().graph
        self.assertIsNotNone(graph)
        self.assertEqual(
            {
                "initialize_workflow",
                "analyze_error",
                "evaluate_patch",
                "prepare_reanalysis",
                "expert_review",
                "finalize",
            },
            set(graph.get_graph().nodes) - {"__start__", "__end__"},
        )

    def test_graph_preserves_reanalysis_and_final_result_contract(self) -> None:
        result = self.make_orchestrator().execute(
            run=DummyRun(),
            gen_run_dir="/tmp/run",
            out_dir="/tmp/out",
            ref_stats={"n_sets": 1},
            gen_stats={"n_sets": 1},
            diff_stats={"n_sets": 0},
        )
        self.assertEqual(result.final_record["primary_label"], "FINAL")
        self.assertEqual(result.final_record["workflow"]["status"], "completed")
        self.assertEqual(result.patch_eval["status"], "patch_works")
        self.assertEqual(len(result.workflow_trace["analysis_history"]), 2)
        self.assertIn("prepare_reanalysis", result.workflow_trace["graph_nodes"])


class CliEnvironmentTests(unittest.TestCase):
    def test_current_interpreter_is_preferred_for_ampl_subprocess(self) -> None:
        with patch.dict("os.environ", {"PATH": "/usr/bin:/bin"}, clear=False):
            _prefer_current_python_on_path()
            import os
            import sys
            from pathlib import Path

            self.assertEqual(
                os.environ["PATH"].split(os.pathsep)[0],
                str(Path(sys.executable).parent),
            )


if __name__ == "__main__":
    unittest.main()
