#!/usr/bin/env python
"""Download the public GIN macaque motor-cortex LFP MATLAB files."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from motor_lfp_utils import RAW_DIR, ensure_dirs


URLS = {
    "T": "https://gin.g-node.org/kilavik.b/Macaque_MotorCortex_LFP_Spike_VisuoMotorBehavior/raw/master/MonkeyT.mat",
    "M": "https://gin.g-node.org/kilavik.b/Macaque_MotorCortex_LFP_Spike_VisuoMotorBehavior/raw/master/MonkeyM.mat",
}


def download(url: str, out_path: Path, force: bool = False) -> None:
    if out_path.exists() and out_path.stat().st_size > 0 and not force:
        print(f"Already present: {out_path}")
        return
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as handle:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            done += len(chunk)
            if total:
                print(f"{out_path.name}: {done / total:6.1%}", end="\r")
    tmp_path.replace(out_path)
    print(f"{out_path.name}: complete")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--monkeys", nargs="+", default=["T", "M"], choices=["T", "M"])
    args = parser.parse_args()

    ensure_dirs()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    for monkey in args.monkeys:
        download(URLS[monkey], args.raw_dir / f"Monkey{monkey}.mat", force=args.force)


if __name__ == "__main__":
    main()
