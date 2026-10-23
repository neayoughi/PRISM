"""One-run CLI using the LangGraph orchestrator and legacy output writer."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from . import base_run_one as legacy_run_one

from .workflow import AgenticOrchestrator


def _prefer_current_python_on_path() -> None:
    """Make legacy ``python3`` child processes use this environment.

    The unchanged AMPL evaluator starts its generated solver script with
    ``python3``. Users commonly invoke this CLI as ``.venv/bin/python`` without
    activating the environment, in which case the child would otherwise use
    the system Python and fail to import ``amplpy``.
    """

    # Do not resolve the executable symlink: in a macOS venv it points back to
    # Homebrew, while the lexical parent is the required ``.venv/bin`` folder.
    interpreter_dir = str(Path(sys.executable).parent)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if not path_entries or path_entries[0] != interpreter_dir:
        os.environ["PATH"] = os.pathsep.join(
            [interpreter_dir, *[entry for entry in path_entries if entry]]
        )


def main() -> None:
    # The legacy CLI owns argument parsing and output serialization. Replacing
    # only its imported orchestrator keeps every file/schema fully compatible.
    _prefer_current_python_on_path()
    legacy_run_one.AgenticOrchestrator = AgenticOrchestrator
    legacy_run_one.main()


if __name__ == "__main__":
    main()
