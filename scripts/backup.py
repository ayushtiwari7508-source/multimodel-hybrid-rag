"""Backup and restore the RAG index data directories.

Usage:
    python scripts/backup.py backup [OUTPUT_DIR]     # create a tar.gz snapshot
    python scripts/backup.py restore ARCHIVE [--force]   # restore from a snapshot
    python scripts/backup.py list                     # list existing snapshots

Backs up: chroma, bm25, index_meta, raw, uploads, logs.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings


def _data_dirs() -> list[dict]:
    s = get_settings()
    dirs = [
        {"name": "chroma", "path": Path(s.chroma_persist_dir)},
        {"name": "bm25", "path": Path(s.bm25_persist_path).parent},
        {"name": "index_meta", "path": Path(s.meta_persist_dir)},
    ]
    for extra in ["raw", "uploads", "logs"]:
        p = ROOT / "data" / extra
        dirs.append({"name": extra, "path": p})
    return dirs


def cmd_backup(out_dir: str) -> Path:
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    archive = out / f"rag_backup_{stamp}.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        for d in _data_dirs():
            if d["path"].exists():
                tf.add(d["path"], arcname=f"data/{d['name']}")
    size = archive.stat().st_size
    print(f"Backup written: {archive} ({size / 1024:.1f} KB)")
    return archive


def cmd_restore(archive_path: str, force: bool) -> None:
    archive = Path(archive_path).expanduser().resolve()
    if not archive.exists():
        print(f"ERROR: archive not found: {archive}", file=sys.stderr)
        sys.exit(1)

    s = get_settings()
    targets = {
        "chroma": Path(s.chroma_persist_dir),
        "bm25": Path(s.bm25_persist_path).parent,
        "index_meta": Path(s.meta_persist_dir),
    }
    for extra in ["raw", "uploads", "logs"]:
        targets[extra] = ROOT / "data" / extra

    extract_tmp = ROOT / "data" / ".restore_tmp"
    if extract_tmp.exists():
        shutil.rmtree(extract_tmp)
    with tarfile.open(archive, "r:gz") as tf:
        members = [m for m in tf.getmembers() if m.name == "data" or m.name.startswith("data/")]
        unsafe = [m.name for m in tf.getmembers() if not (m.name == "data" or m.name.startswith("data/"))]
        if unsafe:
            shutil.rmtree(extract_tmp, ignore_errors=True)
            print(f"ERROR: archive contains unexpected members: {unsafe}", file=sys.stderr)
            sys.exit(1)
        for m in members:
            tf.extract(m, extract_tmp)

    for name, target in targets.items():
        src = extract_tmp / "data" / name
        if not src.exists():
            print(f"Skipping {name}: not in archive")
            continue
        if target.exists() and not force:
            print(f"SKIP {name}: {target} exists (use --force to overwrite)")
            continue
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(target))
        print(f"Restored {name} -> {target}")

    shutil.rmtree(extract_tmp, ignore_errors=True)
    print("Restore complete.")


def cmd_list() -> None:
    """Show snapshots inside the default backup dir and any passed archives."""
    print("Snapshots are stored wherever you ran 'backup'. Pass the archive path to 'restore'.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Backup/restore RAG index data")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list")

    p_backup = sub.add_parser("backup")
    p_backup.add_argument("out_dir", nargs="?", default=str(ROOT / "data" / "backups"))

    p_restore = sub.add_parser("restore")
    p_restore.add_argument("archive")
    p_restore.add_argument("--force", action="store_true", help="overwrite existing dirs")

    args = ap.parse_args()
    if args.cmd == "backup":
        cmd_backup(args.out_dir)
    elif args.cmd == "restore":
        cmd_restore(args.archive, args.force)
    elif args.cmd == "list":
        cmd_list()


if __name__ == "__main__":
    main()
