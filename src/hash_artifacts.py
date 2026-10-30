#!/usr/bin/env python3
"""Write SHA-256 checksums for all package files except checksum manifests and archives."""
from __future__ import annotations
import argparse, hashlib
from pathlib import Path

def sha256(path: Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",default="."); ap.add_argument("--out",default="data/metadata/SHA256SUMS.txt")
    args=ap.parse_args(); root=Path(args.root).resolve(); out=(root/args.out).resolve()
    lines=[]
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.resolve()==out or p.name.endswith(".zip"): continue
        if any(part in {".git","__pycache__",".pytest_cache",".venv"} for part in p.parts): continue
        lines.append(f"{sha256(p)}  {p.relative_to(root).as_posix()}")
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"Wrote {len(lines)} checksums to {out}")
if __name__=="__main__": main()
