"""General-purpose APK -> Corpus pipeline: decompile with jadx, then discover the
app's own package from its manifest instead of a hand-picked path. Works for any
APK, not just the one this repo happens to have been tested against.
"""
from __future__ import annotations

import subprocess
import sys
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from swarm.corpus import Corpus, load_corpus

JADX_VERSION = "1.5.6"
JADX_URL = (
    f"https://github.com/skylot/jadx/releases/download/v{JADX_VERSION}/jadx-{JADX_VERSION}.zip"
)


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetching {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def ensure_jadx(tools_dir: Path) -> Path:
    """Downloads and extracts jadx into tools_dir if not already present.
    Idempotent -- safe to call before every run."""
    jadx_bin = tools_dir / f"jadx-{JADX_VERSION}" / "bin" / "jadx"
    if jadx_bin.exists():
        return jadx_bin
    if subprocess.run(["which", "java"], capture_output=True).returncode != 0:
        raise RuntimeError("java not found on PATH -- jadx needs a JRE")
    zip_path = tools_dir / f"jadx-{JADX_VERSION}.zip"
    _download(JADX_URL, zip_path)
    extract_dir = tools_dir / f"jadx-{JADX_VERSION}"
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    jadx_bin.chmod(0o755)
    return jadx_bin


def decompile_apk(apk_path: Path, out_dir: Path, jadx_bin: Path) -> Path:
    """Runs jadx -d out_dir apk_path. Idempotent -- skips if sources/ already has
    content. Returns out_dir."""
    if out_dir.exists() and any((out_dir / "sources").rglob("*.java")):
        print(f"already decompiled at {out_dir}")
        return out_dir
    print(f"decompiling {apk_path} -> {out_dir}")
    subprocess.run([str(jadx_bin), "-d", str(out_dir), str(apk_path)], check=True)
    return out_dir


def discover_package_root(decompiled_dir: Path) -> Path:
    """Reads the app's own package name from AndroidManifest.xml (jadx decodes the
    binary manifest to readable XML at resources/AndroidManifest.xml) and returns
    the corresponding directory under sources/ -- the app's own code, as opposed to
    whatever third-party libraries got bundled and decompiled alongside it."""
    manifest = decompiled_dir / "resources" / "AndroidManifest.xml"
    if not manifest.exists():
        manifest = decompiled_dir / "AndroidManifest.xml"
    if not manifest.exists():
        raise FileNotFoundError(f"no AndroidManifest.xml found under {decompiled_dir}")
    root = ET.parse(manifest).getroot()
    package = root.get("package")
    if not package:
        raise ValueError(f"{manifest} has no package attribute")
    return decompiled_dir / "sources" / Path(*package.split("."))


def resolve_apk(apk_source: str, tools_dir: Path) -> Path:
    """apk_source is either a local file path or an http(s) URL."""
    if apk_source.startswith("http://") or apk_source.startswith("https://"):
        dest = tools_dir / Path(apk_source).name
        _download(apk_source, dest)
        return dest
    path = Path(apk_source)
    if not path.exists():
        raise FileNotFoundError(f"APK not found: {apk_source}")
    return path


def _apk_out_dir(apk_source: str, work_dir: Path) -> Path:
    # Scoped per APK (by filename stem), not one shared dir -- otherwise a second,
    # different APK would see the first app's output already there and skip.
    stem = Path(apk_source).stem or "apk"
    return work_dir / "decompiled" / stem


def dry_run_report(apk_source: str, work_dir: Path) -> dict:
    """Shows what prepare_corpus WOULD do -- no download, no jadx invocation, no
    filesystem writes. Safe to run to sanity-check a target before spending time
    or bandwidth on it."""
    tools_dir = work_dir / ".tools"
    out_dir = _apk_out_dir(apk_source, work_dir)
    jadx_bin = tools_dir / f"jadx-{JADX_VERSION}" / "bin" / "jadx"
    is_url = apk_source.startswith("http://") or apk_source.startswith("https://")
    report: dict = {
        "apk_source": apk_source,
        "apk_is_url": is_url,
        "apk_would_download_to": str(tools_dir / Path(apk_source).name) if is_url else None,
        "apk_local_path_exists": (not is_url) and Path(apk_source).exists(),
        "jadx_present": jadx_bin.exists(),
        "jadx_would_download": not jadx_bin.exists(),
        "decompiled_output_dir": str(out_dir),
        "already_decompiled": out_dir.exists() and any((out_dir / "sources").rglob("*.java")),
    }
    manifest = out_dir / "resources" / "AndroidManifest.xml"
    if manifest.exists():
        root = discover_package_root(out_dir)
        report["discovered_package_root"] = str(root)
        report["file_count"] = len(load_corpus(root).files) if root.exists() else 0
    else:
        report["discovered_package_root"] = None
        report["file_count"] = None
    return report


def prepare_corpus(
    apk_source: str,
    work_dir: Path,
    include: tuple[str, ...] = ("**/*.java",),
    exclude: tuple[str, ...] = ("R.java", "BuildConfig.java"),
) -> Corpus:
    """One call: fetch (if a URL), decompile (if not already), discover the app's
    own package root from its manifest, and load it as a Corpus. This is what
    --apk drives -- no hand-picked path required."""
    tools_dir = work_dir / ".tools"
    out_dir = _apk_out_dir(apk_source, work_dir)
    apk_path = resolve_apk(apk_source, tools_dir)
    jadx_bin = ensure_jadx(tools_dir)
    decompile_apk(apk_path, out_dir, jadx_bin)
    root = discover_package_root(out_dir)
    corpus = load_corpus(root, include=include, exclude=exclude)
    print(f"corpus: {root} ({len(corpus.files)} files)")
    return corpus


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Decompile an APK and print its discovered corpus root")
    parser.add_argument("apk", help="local path or http(s) URL to an .apk")
    parser.add_argument("--work-dir", default="swarm/.work")
    parser.add_argument("--dry-run", action="store_true",
                         help="show what would happen; no download, no jadx, no writes")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps(dry_run_report(args.apk, Path(args.work_dir)), indent=2))
        return 0
    corpus = prepare_corpus(args.apk, Path(args.work_dir))
    print(f"{len(corpus.files)} files under {corpus.root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
