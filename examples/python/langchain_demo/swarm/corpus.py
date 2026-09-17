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

    def read_lines(self, path: str, start: int, end: int) -> str:
        lines = self.read(path).splitlines()
        start = max(1, start)
        end = min(len(lines), end)
        return "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, end + 1))

    def grep(self, path: str, pattern: str, context: int = 1) -> str:
        import re
        lines = self.read(path).splitlines()
        rx = re.compile(pattern)
        hits = [i for i, line in enumerate(lines) if rx.search(line)]
        if not hits:
            return f"no match for {pattern!r} in {path}"
        shown: set[int] = set()
        for h in hits:
            shown.update(range(max(0, h - context), min(len(lines), h + context + 1)))
        return "\n".join(f"{i + 1}: {lines[i]}" for i in sorted(shown))


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
