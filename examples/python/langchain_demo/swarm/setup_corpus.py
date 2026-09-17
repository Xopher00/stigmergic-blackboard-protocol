"""Fetch and decompile InsecureBankv2 into swarm/corpus/ for --mode live. Run once by
hand: `python -m swarm.setup_corpus`. Idempotent -- skips steps whose output already
exists. Downloads real binaries, so it is never run by the test suite.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).parent
TOOLS_DIR = HERE / ".tools"
CORPUS_DIR = HERE / "corpus"
JADX_VERSION = "1.5.6"
JADX_URL = (
    f"https://github.com/skylot/jadx/releases/download/v{JADX_VERSION}/jadx-{JADX_VERSION}.zip"
)
APK_URL = (
    "https://raw.githubusercontent.com/dineshshetty/Android-InsecureBankv2/"
    "master/InsecureBankv2.apk"
)


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"already have {dest}")
        return
    print(f"fetching {url} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def _ensure_jadx() -> Path:
    jadx_bin = TOOLS_DIR / f"jadx-{JADX_VERSION}" / "bin" / "jadx"
    if jadx_bin.exists():
        return jadx_bin
    zip_path = TOOLS_DIR / f"jadx-{JADX_VERSION}.zip"
    _download(JADX_URL, zip_path)
    extract_dir = TOOLS_DIR / f"jadx-{JADX_VERSION}"
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    jadx_bin.chmod(0o755)
    return jadx_bin


def main() -> int:
    if shutil.which("java") is None:
        print("java not found on PATH -- jadx needs a JRE", file=sys.stderr)
        return 1

    jadx_bin = _ensure_jadx()
    apk_path = TOOLS_DIR / "InsecureBankv2.apk"
    _download(APK_URL, apk_path)

    if CORPUS_DIR.exists() and any(CORPUS_DIR.rglob("*.java")):
        print(f"already decompiled at {CORPUS_DIR}")
        return 0

    print(f"decompiling {apk_path} -> {CORPUS_DIR}")
    subprocess.run([str(jadx_bin), "-d", str(CORPUS_DIR), str(apk_path)], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
