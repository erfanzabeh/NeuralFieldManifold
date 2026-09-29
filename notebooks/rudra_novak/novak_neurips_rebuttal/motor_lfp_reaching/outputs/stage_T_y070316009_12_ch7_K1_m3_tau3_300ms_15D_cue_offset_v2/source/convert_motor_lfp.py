#!/usr/bin/env python
"""Convert Confais/Kilavik/Riehle macaque LFP MATLAB files into labeled arrays."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from motor_lfp_utils import (
    CONVERTED_DIR,
    DIRECTION_LABELS,
    FS,
    GO_SAMPLE,
    TABLE_DIR,
    RAW_DIR,
    ensure_dirs,
    parse_trial_labels,
    slugify,
    write_csv,
)


def deref(handle: h5py.File, value: Any) -> h5py.Dataset | h5py.Group:
    if isinstance(value, (h5py.Dataset, h5py.Group)):
        return value
    return handle[value]


def h5_dataset_value(handle: h5py.File, ref: Any) -> Any:
    obj = deref(handle, ref)
    if isinstance(obj, h5py.Group):
        return obj
    data = obj[()]
    matlab_class = obj.attrs.get("MATLAB_class", b"")
    if isinstance(matlab_class, bytes):
        matlab_class = matlab_class.decode("utf-8", errors="replace")
    if matlab_class == "char":
        return "".join(chr(int(x)) for x in np.asarray(data).ravel()).strip()
    return data


def h5_string(handle: h5py.File, group: h5py.Group, field: str, row_idx: int) -> str:
    value = h5_dataset_value(handle, group[field][row_idx, 0])
    if isinstance(value, str):
        return value
    arr = np.asarray(value).ravel()
    if arr.size == 0:
        return ""
    scalar = arr[0]
    if isinstance(scalar, bytes):
        return scalar.decode("utf-8", errors="replace")
    if np.issubdtype(arr.dtype, np.number):
        if arr.size == 1:
            return str(int(scalar)) if float(scalar).is_integer() else str(float(scalar))
        return "_".join(str(int(x)) for x in arr if np.isfinite(x))
    return str(scalar)


def h5_scalar_string(handle: h5py.File, group: h5py.Group, field: str, row_idx: int) -> str:
    value = h5_dataset_value(handle, group[field][row_idx, 0])
    if isinstance(value, str):
        return value
    arr = np.asarray(value, dtype=float).ravel()
    if arr.size == 0:
        return ""
    scalar = arr[0]
    if np.isnan(scalar):
        return ""
    return str(int(scalar)) if float(scalar).is_integer() else str(float(scalar))


def h5_cell_dataset(handle: h5py.File, group: h5py.Group, field: str, row_idx: int) -> h5py.Dataset:
    obj = deref(handle, group[field][row_idx, 0])
    if not isinstance(obj, h5py.Dataset):
        raise TypeError(f"Expected {field} to reference a cell dataset, got {type(obj).__name__}")
    return obj


def h5_cell_array(handle: h5py.File, cell_ds: h5py.Dataset, cond_idx: int) -> np.ndarray:
    ref = cell_ds[cond_idx, 0] if cell_ds.ndim == 2 else cell_ds[cond_idx]
    obj = deref(handle, ref)
    if isinstance(obj, h5py.Group):
        raise TypeError(f"Expected numeric cell item, got group {obj.name}")
    return np.asarray(obj[()])


def trial_matrix_from_lfp_cell(handle: h5py.File, lfp_cell_ds: h5py.Dataset, cond_idx: int) -> np.ndarray:
    arr = np.asarray(h5_cell_array(handle, lfp_cell_ds, cond_idx), dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.shape[0] >= GO_SAMPLE + 1 and arr.shape[0] > arr.shape[1]:
        return arr.T.astype(np.float32, copy=False)
    if arr.shape[1] >= GO_SAMPLE + 1:
        return arr.astype(np.float32, copy=False)
    return arr if arr.shape[0] <= arr.shape[1] else arr.T


def trial_column(arr: np.ndarray, trial_idx: int) -> np.ndarray:
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr[min(trial_idx, arr.shape[0] - 1)].reshape(1)
    if arr.shape[1] > trial_idx:
        return arr[:, trial_idx]
    if arr.shape[0] > trial_idx:
        return arr[trial_idx, :]
    return np.array([], dtype=float)


def trial_vector_value(arr: np.ndarray, trial_idx: int, default: float = np.nan) -> float:
    vec = np.asarray(arr, dtype=float).ravel()
    if vec.size == 0 or trial_idx >= vec.size:
        return default
    value = vec[trial_idx]
    return float(value) if np.isfinite(value) else default


def convert_entry_h5(
    handle: h5py.File,
    group: h5py.Group,
    row_idx: int,
    animal: str,
    source_pair_index: int,
    force: bool = False,
) -> tuple[dict[str, object] | None, pd.DataFrame]:
    session_id = h5_string(handle, group, "Session_ID", row_idx)
    lfp_id = h5_scalar_string(handle, group, "LFP_ID", row_idx)
    cell_id = h5_string(handle, group, "Cell_ID", row_idx)
    twodir = h5_scalar_string(handle, group, "Twodir", row_idx)
    if not session_id or not lfp_id:
        return None, pd.DataFrame()

    uid = f"monkey{animal}_session-{slugify(session_id)}_lfp-{slugify(lfp_id)}"
    out_path = CONVERTED_DIR / f"{uid}.npz"
    if out_path.exists() and not force:
        with np.load(out_path, allow_pickle=True) as existing:
            n_trials = int(existing["lfp"].shape[0])
            n_samples = int(existing["lfp"].shape[1])
            direction = existing["direction"].astype(int)
            delay_label = existing["delay_label"].astype(str)
        manifest_row = manifest_from_labels(
            uid=uid,
            animal=animal,
            session_id=session_id,
            lfp_id=lfp_id,
            cell_id=cell_id,
            twodir=twodir,
            n_trials=n_trials,
            n_samples=n_samples,
            source_pair_index=source_pair_index,
            direction=direction,
            delay_label=delay_label,
            converted_path=out_path,
        )
        return manifest_row, condition_count_rows(uid, animal, session_id, lfp_id, direction, delay_label)

    lfp_cells = h5_cell_dataset(handle, group, "lfp", row_idx)
    codes_cells = h5_cell_dataset(handle, group, "TrialCodesCorr", row_idx)
    chron_cells = h5_cell_dataset(handle, group, "TrialChronolOrderCorr", row_idx)
    traject = deref(handle, group["Traject"][row_idx, 0])
    movement_cells = traject["MvtOnset"] if isinstance(traject, h5py.Group) and "MvtOnset" in traject else None

    lfp_trials: list[np.ndarray] = []
    direction_labels: list[int] = []
    delay_labels: list[str] = []
    trial_type_codes: list[int] = []
    condition_codes: list[int] = []
    source_condition_index: list[int] = []
    source_trial_index: list[int] = []
    original_trial_number: list[int] = []
    movement_onset_ms: list[float] = []

    n_conditions = int(lfp_cells.shape[0])
    for cond_idx, lfp_cell in enumerate(lfp_cells):
        trial_matrix = trial_matrix_from_lfp_cell(handle, lfp_cells, cond_idx)
        if trial_matrix.size == 0:
            continue
        codes_matrix = h5_cell_array(handle, codes_cells, cond_idx)
        chron_vector = h5_cell_array(handle, chron_cells, cond_idx)
        movement_vector = h5_cell_array(handle, movement_cells, cond_idx) if movement_cells is not None else np.array([])
        for trial_idx in range(trial_matrix.shape[0]):
            codes = trial_column(codes_matrix, trial_idx)
            direction, delay_label, trial_type_code, condition_code = parse_trial_labels(
                codes=codes,
                cond_idx=cond_idx,
                n_conditions=n_conditions,
            )
            lfp_trials.append(trial_matrix[trial_idx])
            direction_labels.append(direction)
            delay_labels.append(delay_label)
            trial_type_codes.append(trial_type_code)
            condition_codes.append(condition_code)
            source_condition_index.append(cond_idx + 1)
            source_trial_index.append(trial_idx + 1)
            original_trial_number.append(int(trial_vector_value(chron_vector, trial_idx, default=-1)))
            movement_onset_ms.append(trial_vector_value(movement_vector, trial_idx, default=np.nan))

    if not lfp_trials:
        return None, pd.DataFrame()

    min_samples = min(len(trial) for trial in lfp_trials)
    if min_samples <= GO_SAMPLE:
        return None, pd.DataFrame()
    lfp = np.vstack([trial[:min_samples] for trial in lfp_trials]).astype(np.float32, copy=False)
    direction_arr = np.asarray(direction_labels, dtype=np.int16)
    delay_arr = np.asarray(delay_labels, dtype=object)

    np.savez(
        out_path,
        lfp=lfp,
        direction=direction_arr,
        delay_label=delay_arr,
        trial_type_code=np.asarray(trial_type_codes, dtype=np.int16),
        condition_code=np.asarray(condition_codes, dtype=np.int16),
        source_condition_index=np.asarray(source_condition_index, dtype=np.int16),
        source_trial_index=np.asarray(source_trial_index, dtype=np.int32),
        original_trial_number=np.asarray(original_trial_number, dtype=np.int32),
        movement_onset_ms=np.asarray(movement_onset_ms, dtype=np.float32),
        sampling_rate_hz=np.asarray(FS, dtype=np.int16),
        go_sample=np.asarray(GO_SAMPLE, dtype=np.int16),
        monkey=np.asarray(animal),
        session_id=np.asarray(session_id),
        lfp_id=np.asarray(lfp_id),
        cell_id=np.asarray(cell_id),
        twodir=np.asarray(twodir),
        source_pair_index=np.asarray(source_pair_index, dtype=np.int32),
    )

    manifest_row = manifest_from_labels(
        uid=uid,
        animal=animal,
        session_id=session_id,
        lfp_id=lfp_id,
        cell_id=cell_id,
        twodir=twodir,
        n_trials=lfp.shape[0],
        n_samples=lfp.shape[1],
        source_pair_index=source_pair_index,
        direction=direction_arr,
        delay_label=delay_arr.astype(str),
        converted_path=out_path,
    )
    return manifest_row, condition_count_rows(uid, animal, session_id, lfp_id, direction_arr, delay_arr.astype(str))


def manifest_from_labels(
    uid: str,
    animal: str,
    session_id: str,
    lfp_id: str,
    cell_id: str,
    twodir: str,
    n_trials: int,
    n_samples: int,
    source_pair_index: int,
    direction: np.ndarray,
    delay_label: np.ndarray,
    converted_path: Path,
) -> dict[str, object]:
    direction_valid = direction[np.isin(direction, DIRECTION_LABELS)]
    delay_valid = delay_label[np.isin(delay_label, ["short", "long"])]
    row: dict[str, object] = {
        "lfp_uid": uid,
        "monkey": animal,
        "session_id": session_id,
        "lfp_id": lfp_id,
        "source_cell_id": cell_id,
        "twodir": twodir,
        "source_pair_index": source_pair_index,
        "n_trials": int(n_trials),
        "n_samples": int(n_samples),
        "sampling_rate_hz": FS,
        "go_sample_zero_based": GO_SAMPLE,
        "n_directions": int(len(np.unique(direction_valid))),
        "n_delay_types": int(len(np.unique(delay_valid))),
        "has_all_6_directions": bool(set(np.unique(direction_valid)) == set(DIRECTION_LABELS)),
        "has_short_and_long": bool(set(np.unique(delay_valid)) == {"short", "long"}),
        "converted_path": str(converted_path.relative_to(CONVERTED_DIR.parent)),
    }
    for direction_id in DIRECTION_LABELS:
        row[f"direction_{direction_id}_count"] = int(np.sum(direction == direction_id))
    for delay in ("short", "long", "unknown"):
        row[f"{delay}_delay_count"] = int(np.sum(delay_label == delay))
    return row


def condition_count_rows(
    uid: str,
    animal: str,
    session_id: str,
    lfp_id: str,
    direction: np.ndarray,
    delay_label: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for direction_id in DIRECTION_LABELS:
        for delay in ("short", "long"):
            rows.append(
                {
                    "lfp_uid": uid,
                    "monkey": animal,
                    "session_id": session_id,
                    "lfp_id": lfp_id,
                    "direction": direction_id,
                    "delay": delay,
                    "n_trials": int(np.sum((direction == direction_id) & (delay_label == delay))),
                }
            )
    return pd.DataFrame(rows)


def convert_animal(mat_path: Path, force: bool = False, max_entries: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    seen: set[tuple[str, str, str]] = set()
    manifest_rows: list[dict[str, object]] = []
    count_tables: list[pd.DataFrame] = []

    animal = "T" if "T" in mat_path.stem else "M" if "M" in mat_path.stem else mat_path.stem
    with h5py.File(mat_path, "r") as handle:
        group = handle["Monkey"]
        n_entries = int(group["lfp"].shape[0])
        entry_indices = list(range(n_entries))
        if max_entries is not None:
            entry_indices = entry_indices[:max_entries]

        for pair_idx in tqdm(entry_indices, desc=f"Converting Monkey {animal}"):
            session_id = h5_string(handle, group, "Session_ID", pair_idx)
            lfp_id = h5_scalar_string(handle, group, "LFP_ID", pair_idx)
            key = (animal, session_id, lfp_id)
            if key in seen:
                continue
            seen.add(key)
            row, counts = convert_entry_h5(
                handle,
                group,
                row_idx=pair_idx,
                animal=animal,
                source_pair_index=pair_idx,
                force=force,
            )
            if row is not None:
                manifest_rows.append(row)
            if not counts.empty:
                count_tables.append(counts)

    manifest = pd.DataFrame(manifest_rows)
    condition_counts = pd.concat(count_tables, ignore_index=True) if count_tables else pd.DataFrame()
    return manifest, condition_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-entries", type=int, default=None, help="Debug option: inspect only the first N pair entries per monkey.")
    parser.add_argument("--monkeys", nargs="+", default=["T", "M"], choices=["T", "M"])
    args = parser.parse_args()

    ensure_dirs()
    all_manifest = []
    all_counts = []
    for monkey in args.monkeys:
        mat_path = args.raw_dir / f"Monkey{monkey}.mat"
        if not mat_path.exists():
            raise FileNotFoundError(f"Missing {mat_path}; download the GIN files first.")
        manifest, counts = convert_animal(mat_path, force=args.force, max_entries=args.max_entries)
        all_manifest.append(manifest)
        all_counts.append(counts)

    manifest_df = pd.concat(all_manifest, ignore_index=True) if all_manifest else pd.DataFrame()
    counts_df = pd.concat(all_counts, ignore_index=True) if all_counts else pd.DataFrame()
    write_csv(manifest_df, TABLE_DIR / "lfp_manifest.csv")
    write_csv(counts_df, TABLE_DIR / "condition_counts.csv")
    print(f"Converted {len(manifest_df)} unique LFP recordings.")
    print(f"Wrote {TABLE_DIR / 'lfp_manifest.csv'}")
    print(f"Wrote {TABLE_DIR / 'condition_counts.csv'}")


if __name__ == "__main__":
    main()
