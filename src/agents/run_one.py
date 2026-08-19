"""One-run CLI using the LangGraph orchestrator."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from . import base_run_one

from .workflow import AgenticOrchestrator


def _prefer_current_python_on_path() -> None:
    """Make ``python3`` child processes use this environment."""

    interpreter_dir = str(Path(sys.executable).parent)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if not path_entries or path_entries[0] != interpreter_dir:
        os.environ["PATH"] = os.pathsep.join(
            [interpreter_dir, *[entry for entry in path_entries if entry]]
        )


def main() -> None:
    _prefer_current_python_on_path()
    base_run_one.AgenticOrchestrator = AgenticOrchestrator
    base_run_one.main()


if __name__ == "__main__":
    main()
