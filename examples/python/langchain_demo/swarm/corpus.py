"""Read-only access to the decompiled source tree the swarm analyzes. Files are never
written to; the corpus is evidence, not a workspace (TEST_IDEAS.md: "the analyzed
files are read-only evidence").
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Corpus:
    root: Path
    files: tuple[str, ...]  # relative POSIX paths, sorted, stable order

    def read(self, path: str) -> str:
        if path not in self.files:
            raise ValueError(f"not in corpus: {path}")
        return (self.root / path).read_text(errors="replace")


def load_corpus(
    root: str | Path,
    include: tuple[str, ...] = ("**/*.java",),
    exclude: tuple[str, ...] = (),
) -> Corpus:
    root = Path(root)
    matched: set[str] = set()
    for pattern in include:
        for p in root.glob(pattern):
            if p.is_file():
                matched.add(p.relative_to(root).as_posix())
    for pattern in exclude:
        matched = {f for f in matched if not fnmatch.fnmatch(f, pattern)}
    return Corpus(root=root, files=tuple(sorted(matched)))
