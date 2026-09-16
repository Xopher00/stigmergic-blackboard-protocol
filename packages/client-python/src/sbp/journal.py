"""
Append-only JSONL operation journal (P3.1): one line per blackboard operation.
"""
import json
import os
from typing import Any


class Journal:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = path
        self._seq = 0
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

    def append(self, entry: dict[str, Any]) -> None:
        entry = {"seq": self._seq, **entry}
        self._seq += 1
        try:
            with open(self._path, "a") as f:
                f.write(json.dumps(entry, separators=(",", ":")) + "\n")
        except OSError as e:
            print(f"[SBP Journal] write failed: {e}")
