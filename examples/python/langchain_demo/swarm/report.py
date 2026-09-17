"""Run summary writer -- the report itself is written by the judge (see roles.py's
write_report tool); this is the supervisor's own account of how the run ended."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_run_summary(path: Path, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fields, indent=2, default=str))
