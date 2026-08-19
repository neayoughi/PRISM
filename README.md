# PRISM

PRISM is a LangGraph workflow for qualitative error analysis of generated AMPL optimization specifications. It compares a generated AMPL model against the problem description, reference model, data, and solver output; assigns error labels from an evolving taxonomy; proposes a minimal patch; and validates the patch by solving the corrected optimization model and comparing its objective value with the reference objective.

The workflow corresponds to Chapter 5 of Negin's PhD thesis, "Error Analysis of LLM-Generated Optimization Specifications." The study analyzes 231 non-perfect AMPL specifications from [EXEOS](https://github.com/neayoughi/EXEOS) runs, including compilation errors, runtime errors, and executable specifications with incorrect objective values. PRISM produced successful automated patches for 182 of the 231 cases, a 78.8% patch-validation rate, and the validated records were used to refine the final error taxonomy.

![PRISM approach](assets/prism-approach.png)

## Repository Contents

- `src/agents/`: LangGraph workflow, command-line entry point, analysis, patch evaluation, AMPL execution, schema, rendering, and model-provider modules.
- `prompts/`: taxonomy, task template, output schema, and prompt files used by the workflow.
- `data/EXEOS/`: selected Excel workbooks for erroneous AMPL specifications.
- `result/`: selected human-validation result files for Gemini 2.5 Pro and o4-mini.
- `assets/prism-approach.png`: approach overview figure.

Reference solutions and the full `data/Solution` directory are intentionally not included.

## Included Data

The public data subset contains:

- `data/EXEOS/AMPL4-gemini2.5/ErrorAnalysis_AMPL4-gemini2.5pro.xlsx`
- `data/EXEOS/AMPL4-o4mini/ErrorAnalysis_AMPL-o4mini.xlsx`
- `result/gemeni2.5/batch_report_gemini2.5_human_validated.csv`
- `result/gemeni2.5/batch_patch_eval_gemini2.5_human_validated.csv`
- `result/o4mini/batch_report_o4mini_human_validated.csv`
- `result/o4mini/batch_patch_eval_o4mini_human_validated.csv`

## Setup

Create and activate a Python environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

AMPL, `amplpy`, and the Gurobi AMPL module are required for patch-validation runs that execute optimization models.

Copy `config.example.json` to `config.json` and fill only the provider fields needed for your run. You can also point to a configuration file with:

```bash
export EXEOS_CONFIG_PATH=/path/to/config.json
```

Do not commit `config.json`.

## Run Tests

The graph construction and routing tests do not execute AMPL or call external model providers:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Run One Case

Provide paths to a generated run, a reference solution, and an output directory:

```bash
PYTHONPATH=src python -m agents.run_one \
  --gen_run_dir /path/to/generated/run \
  --ref_solution_dir /path/to/reference/solution \
  --out_dir reports_langgraph/example \
  --model vertex-gemini-2.5-pro \
  --max_patch_refinement_attempts 5 \
  --max_analysis_cycles 5
```

The command writes `record.json`, `initial_record.json`, `patch_eval.json`, `workflow_trace.json`, `record.csv`, `patch_eval.csv`, and a rendered Markdown report to the output directory.
