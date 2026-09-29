"""Archive the superseded run and reuse only exactly unchanged stage inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

import numpy as np
import pandas as pd

from prego_geometric_fits import array_hash, sha256, write_json
from stage_geometry_decoding import UNIT, SUPERSEDED

CHECKPOINT = "57d5b2a90a7430234e1275ca6ea178a3d943a4fa"
OLD_CARTOON = UNIT / "outputs/macaque_task_six_stages_v1"
OLD_ROOTS = (SUPERSEDED, OLD_CARTOON)


def archive_previous():
    repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    remote = subprocess.check_output(
        ["git", "ls-remote", "origin", "refs/heads/novak/eeg_decode_update"], text=True).split()[0]
    subprocess.run(["git", "merge-base", "--is-ancestor", CHECKPOINT, remote], check=True)
    destination = repo.parent / "NeuralFieldManifold_archives" / f"stage_onset_{CHECKPOINT[:7]}"
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "superseded_stage_and_cartoon.tar.gz"
    receipt_path = destination / "manifest.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        assert sha256(archive) == receipt["archive_sha256"]
        return receipt
    files = [p for root in OLD_ROOTS for p in sorted(root.rglob("*")) if p.is_file()]
    assert all(root.is_dir() for root in OLD_ROOTS) and files
    hashes = {str(p.relative_to(UNIT / "outputs")): sha256(p) for p in files}
    tracked = {}
    for root in OLD_ROOTS:
        data = subprocess.check_output(["git", "ls-tree", "-rz", CHECKPOINT, "--", str(root.relative_to(repo))])
        for entry in data.split(b"\0"):
            if not entry:
                continue
            header, name = entry.split(b"\t", 1)
            path = repo / name.decode()
            body = path.read_bytes()
            blob = hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest()
            assert blob == header.decode().split()[2], f"Not preserved by pushed commit: {path}"
            tracked[str(path.relative_to(UNIT / "outputs"))] = blob
    with tarfile.open(archive, "w:gz") as handle:
        for p in files:
            handle.add(p, arcname=str(p.relative_to(UNIT / "outputs")), recursive=False)
    with tarfile.open(archive, "r:gz") as handle:
        assert {m.name for m in handle.getmembers()} == set(hashes)
        for member in handle.getmembers():
            assert hashlib.sha256(handle.extractfile(member).read()).hexdigest() == hashes[member.name]
    receipt = dict(checkpoint=CHECKPOINT, remote_confirmed=remote, archive=str(archive),
                   archive_sha256=sha256(archive), file_sha256=hashes,
                   tracked_files=len(tracked), additional_local_files=len(files)-len(tracked),
                   archive_content_verified=True, old_roots=[str(p) for p in OLD_ROOTS])
    write_json(receipt_path, receipt)
    return receipt


def reuse_unchanged(root, metadata, raw, processed, clouds):
    """Reuse four stages only after exact trial, slice, input, and source checks."""
    old = pd.read_csv(SUPERSEDED / "inputs/window_metadata.csv", keep_default_na=False)
    old_config = json.loads((SUPERSEDED / "config.json").read_text())
    config = json.loads((root / "config.json").read_text())
    for key in ("preprocessing", "fitter", "m", "tau_samples", "feature_columns", "classifier"):
        assert old_config[key] == config[key], key
    assert json.loads((SUPERSEDED / "environment.json").read_text()) == json.loads((root / "environment.json").read_text())
    hashes = json.loads((SUPERSEDED / "source_input_hashes.json").read_text())
    for name, digest in hashes.items():
        if Path(name).name not in ("stage_geometry_decoding.py", "run_stage_geometry_decoding.py"):
            assert sha256(name) == digest, name
    old_npz = SUPERSEDED / "inputs/windows.npz"
    old_frozen = json.loads((SUPERSEDED / "frozen_input_hashes.json").read_text())
    assert sha256(old_npz) == old_frozen[str(old_npz.resolve())]
    audit = []
    with np.load(old_npz) as saved:
        previous = {k: saved[k] for k in saved.files}
        for row in metadata.itertuples():
            i = row.window_index
            prior = old.iloc[i]
            for name in ("original_trial_number", "stage_name", "raw_trial_index", "heldout_fold_zero_based", "direction"):
                assert getattr(row, name) == prior[name], (i, name)
            changed = row.stage_name in ("post_TC", "post_SC")
            if not changed:
                assert row.start_sample == prior.start_sample and row.end_sample_exclusive == prior.end_sample_exclusive
                np.testing.assert_array_equal(raw[i], previous["raw"][i])
                processed[i] = previous["processed"][i]
                clouds[i] = previous["clouds"][i]
                source = SUPERSEDED / "checkpoints" / f"window_{i:04d}.json"
                result = json.loads(source.read_text())
                assert result["cloud_sha256"] == array_hash(clouds[i]) == prior.cloud_sha256
                assert result["original_trial_number"] == row.original_trial_number
                assert result["stage_name"] == row.stage_name
                target = root / "checkpoints" / source.name
                shutil.copy2(source, target)
                assert sha256(source) == sha256(target)
            else:
                assert row.start_sample > prior.start_sample
                assert not np.array_equal(raw[i], previous["raw"][i])
            audit.append(dict(window_index=i, original_trial_number=row.original_trial_number,
                              stage_name=row.stage_name, old_start_sample=int(prior.start_sample),
                              new_start_sample=row.start_sample, shift_ms=row.start_sample-int(prior.start_sample),
                              reused_fit=not changed, old_cloud_sha256=prior.cloud_sha256,
                              old_preprocessing_error=prior.preprocessing_error))
    audit = pd.DataFrame(audit)
    assert audit.reused_fit.sum() == 792 and (~audit.reused_fit).sum() == 396
    audit.to_csv(root / "inputs/correction_audit.csv", index=False)
    return audit


def remove_superseded(root):
    """Delete only verified archived outputs after the corrected run passes."""
    assert json.loads((root / "validation.json").read_text())["passed"]
    assert json.loads((root / "figure_validation.json").read_text())["passed"]
    receipt = json.loads((root / "superseded_archive.json").read_text())
    assert sha256(receipt["archive"]) == receipt["archive_sha256"]
    for old_root in OLD_ROOTS:
        actual = {str(p.relative_to(UNIT / "outputs")): sha256(p)
                  for p in old_root.rglob("*") if p.is_file()}
        expected = {p: h for p, h in receipt["file_sha256"].items() if p.startswith(old_root.name + "/")}
        assert actual == expected, f"Superseded files changed since backup: {old_root}"
    for old_root in OLD_ROOTS:
        shutil.rmtree(old_root)
    write_json(root / "cleanup.json", dict(deleted=[str(p) for p in OLD_ROOTS],
               archive=receipt["archive"], archive_sha256=receipt["archive_sha256"],
               historical_commit=CHECKPOINT, raw_inputs_deleted=False, earlier_analyses_deleted=False))
