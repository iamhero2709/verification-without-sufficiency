"""
modal_package.py - tar the volume into one file so it downloads reliably.

`modal volume get` cannot walk nested directory trees on some CLI versions: flat
entries come down fine, anything with subdirectories fails. Rather than fight
it, we build a single archive on the volume and download that.

    modal run modal_package.py                 # build the archive
    modal volume get triver-results paper_artifacts.tar.gz .
    tar xzf paper_artifacts.tar.gz -C ./triver-dump

Prints a manifest so you can confirm nothing important was left out.
"""
from __future__ import annotations

import json
import os
import tarfile
import time
from pathlib import Path

import modal

image = modal.Image.debian_slim(python_version="3.11")
results = modal.Volume.from_name("triver-results", create_if_missing=True)
app = modal.App("triver-package")

R = Path("/results")

# Everything the paper and the repository need. Order is cosmetic.
INCLUDE = ["results", "adv", "artifacts", "splits", "v2", "llmdecomp", "stats"]

# Intermediates that are large and regenerable. Keep them out unless asked.
SKIP_PATTERNS = ["/wandb/", "__pycache__", ".ipynb_checkpoints"]


def _skip(path: str) -> bool:
    return any(p in path for p in SKIP_PATTERNS)


@app.function(volumes={"/results": results}, cpu=4, memory=16384,
              timeout=60 * 60)
def build_archive(name: str = "paper_artifacts.tar.gz",
                  include_traces: bool = True) -> dict:
    t0 = time.time()
    out = R / name
    if out.exists():
        out.unlink()

    manifest, total_files, total_bytes = {}, 0, 0
    with tarfile.open(out, "w:gz") as tar:
        for entry in INCLUDE:
            src = R / entry
            if not src.exists():
                print(f"  skip {entry} (not present)")
                continue
            n, b = 0, 0
            for root, dirs, files in os.walk(src):
                dirs[:] = [d for d in dirs if not _skip(os.path.join(root, d))]
                for f in files:
                    fp = os.path.join(root, f)
                    if _skip(fp):
                        continue
                    rel = os.path.relpath(fp, R)
                    if not include_traces and (
                            "/adv_gen/" in fp or "/setlevel/" in fp
                            or "/v2/gen/" in fp):
                        continue
                    tar.add(fp, arcname=rel)
                    n += 1
                    b += os.path.getsize(fp)
            manifest[entry] = {"files": n, "mb": round(b / 1e6, 1)}
            total_files += n
            total_bytes += b
            print(f"  {entry:12s} {n:5d} files  {b/1e6:8.1f} MB")

    size_mb = out.stat().st_size / 1e6
    results.commit()

    info = {"archive": name, "archive_mb": round(size_mb, 1),
            "files": total_files, "uncompressed_mb": round(total_bytes / 1e6, 1),
            "ratio": round(total_bytes / max(out.stat().st_size, 1), 1),
            "manifest": manifest, "seconds": round(time.time() - t0, 1)}
    (R / "archive_manifest.json").write_text(json.dumps(info, indent=2))
    results.commit()

    print("\n" + json.dumps(info, indent=2))
    print("\n" + "=" * 68)
    print(f"  {name}  {size_mb:.1f} MB  ({total_files} files, "
          f"{info['ratio']}x compression)")
    if size_mb > 400:
        print("  Large. Consider --include-traces false for a code-and-results")
        print("  only archive, and publish traces separately.")
    print("=" * 68)
    print("\n  Download it with:")
    print(f"    modal volume get triver-results {name} .")
    print(f"    mkdir -p triver-dump && tar xzf {name} -C triver-dump")
    return info


@app.function(volumes={"/results": results}, cpu=2, memory=4096)
def tree(depth: int = 2) -> None:
    """Print the volume layout, so a missing file can be located by name."""
    for root, dirs, files in os.walk(R):
        rel = os.path.relpath(root, R)
        lvl = 0 if rel == "." else rel.count(os.sep) + 1
        if lvl > depth:
            dirs[:] = []
            continue
        pad = "  " * lvl
        print(f"{pad}{'.' if rel == '.' else os.path.basename(root)}/  "
              f"({len(files)} files)")


@app.local_entrypoint()
def main(name: str = "paper_artifacts.tar.gz", traces: bool = True,
         show_tree: bool = False):
    if show_tree:
        tree.remote()
        return
    build_archive.remote(name, traces)
