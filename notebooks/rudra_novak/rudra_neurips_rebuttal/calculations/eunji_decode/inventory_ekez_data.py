#!/usr/bin/env python
"""Inventory the EKEZ LFP files and timestamp spreadsheet.

This is intentionally lightweight: it checks file shape/duration, parses the
spreadsheet windows, and writes simple CSVs for the decode unit.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


UNIT_DIR = Path(__file__).resolve().parent
REBUTTAL_DIR = UNIT_DIR.parents[1]
DATA_DIR = REBUTTAL_DIR / "data" / "ekez"
CACHE_DIR = UNIT_DIR / "cache"

N_CHANNELS_TOTAL = 32
RAW_FS = 20_000
DTYPE_BYTES = 2
PROBE_ORDER = [18, 19, 12, 13]


def parse_mmss(text: str) -> float:
    minutes, seconds = str(text).split(":")
    return int(minutes) * 60 + float(seconds)


def parse_window(text: str) -> tuple[float, float]:
    matches = re.findall(r"([0-9]+:[0-9]+(?:\.[0-9]+)?)", str(text))
    if len(matches) != 2:
        raise ValueError(f"Could not parse window: {text!r}")
    start, end = parse_mmss(matches[0]), parse_mmss(matches[1])
    if end <= start:
        raise ValueError(f"Window end must be after start: {text!r}")
    return start, end


def parse_channels(text: str) -> list[int]:
    return [int(tok.strip()) for tok in str(text).split(",") if tok.strip()]


def data_file_for_date(date: int) -> Path:
    return DATA_DIR / f"LFP0_{int(date)}.dat"


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    file_rows = []
    frame_bytes = N_CHANNELS_TOTAL * DTYPE_BYTES
    for path in sorted(DATA_DIR.glob("LFP0_*.dat")):
        size = path.stat().st_size
        samples = size // frame_bytes
        file_rows.append(
            {
                "file": path.name,
                "size_bytes": size,
                "divisible_by_32_int16_channels": size % frame_bytes == 0,
                "samples": samples,
                "duration_s": samples / RAW_FS,
                "duration_min": samples / RAW_FS / 60,
            }
        )
    file_df = pd.DataFrame(file_rows)
    file_df.to_csv(CACHE_DIR / "ekez_file_inventory.csv", index=False)

    labels = pd.read_excel(DATA_DIR / "timestamp.xlsx")
    label_rows = []
    for _, row in labels.iterrows():
        start, end = parse_window(row["window"])
        path = data_file_for_date(row["date"])
        duration = None
        within_file = False
        if path.exists():
            samples = path.stat().st_size // frame_bytes
            duration = samples / RAW_FS
            within_file = end <= duration
        channels = parse_channels(row["good_Ch"])
        label_rows.append(
            {
                "mouse_ID": row["mouse_ID"],
                "group": row["group"],
                "date": int(row["date"]),
                "phase": str(row["phase"]).strip().lower(),
                "good_Ch": ",".join(str(ch) for ch in channels),
                "start_s": start,
                "end_s": end,
                "duration_s": end - start,
                "dat_file": path.name,
                "dat_file_exists": path.exists(),
                "window_within_file": within_file,
                "channels_in_tutorial_probe_order": all(ch in PROBE_ORDER for ch in channels),
            }
        )
    label_df = pd.DataFrame(label_rows)
    label_df.to_csv(CACHE_DIR / "ekez_label_windows.csv", index=False)

    overlap_rows = []
    for (mouse, date), group in label_df.groupby(["mouse_ID", "date"]):
        rows = group.to_dict("records")
        for i, left in enumerate(rows):
            for right in rows[i + 1 :]:
                overlap = max(0.0, min(left["end_s"], right["end_s"]) - max(left["start_s"], right["start_s"]))
                if overlap > 0:
                    overlap_rows.append(
                        {
                            "mouse_ID": mouse,
                            "date": date,
                            "phase_a": left["phase"],
                            "phase_b": right["phase"],
                            "overlap_s": overlap,
                        }
                    )
    overlap_df = pd.DataFrame(overlap_rows)
    overlap_df.to_csv(CACHE_DIR / "ekez_overlap_report.csv", index=False)

    print(f"Wrote {CACHE_DIR / 'ekez_file_inventory.csv'}")
    print(f"Wrote {CACHE_DIR / 'ekez_label_windows.csv'}")
    print(f"Wrote {CACHE_DIR / 'ekez_overlap_report.csv'}")
    print(f"Files: {len(file_df)}; label rows: {len(label_df)}; overlaps: {len(overlap_df)}")


if __name__ == "__main__":
    main()
