#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import h5py
import numpy as np
import requests


DATASET = "xjtuqianbo/wiception3d"
VERSION = 1
BASE = "https://www.kaggle.com/api/v1/datasets/download"
LIST_PATHS = {
    "train": "wifi_dataset/train_data/train_data_list.txt",
    "test": "wifi_dataset/test_data/test_data_list.txt",
}
THREAD_LOCAL = threading.local()


def session() -> requests.Session:
    if not hasattr(THREAD_LOCAL, "session"):
        THREAD_LOCAL.session = requests.Session()
        THREAD_LOCAL.session.headers["User-Agent"] = "wifi-vision-research/0.1"
    return THREAD_LOCAL.session


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_file(path: Path, remote_path: str) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        if remote_path.endswith(".npy"):
            value = np.load(path, mmap_mode="r")
            return value.ndim == 3 and value.shape[1:] == (14, 3)
        if remote_path.endswith(".mat"):
            with h5py.File(path, "r") as handle:
                return handle["csi_out"].shape == (20, 30, 3, 3)
        if remote_path.endswith(".txt"):
            return bool(path.read_text(encoding="utf-8").strip())
    except Exception:
        return False
    return True


def download(remote_path: str, destination: Path, retries: int = 9) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if valid_file(destination, remote_path):
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    url = f"{BASE}/{DATASET}/{quote(remote_path, safe='')}?datasetVersionNumber={VERSION}"
    for attempt in range(retries):
        start = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={start}-"} if start else {}
        try:
            response = session().get(url, headers=headers, stream=True, timeout=(20, 90))
            if response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", "0") or 0)
                raise requests.HTTPError(
                    f"rate limited; retry_after={retry_after}", response=response
                )
            response.raise_for_status()
            append = start > 0 and response.status_code == 206
            if not append:
                start = 0
            mode = "ab" if append else "wb"
            with partial.open(mode) as handle:
                for chunk in response.iter_content(128 * 1024):
                    if chunk:
                        handle.write(chunk)
            expected_tail = int(response.headers.get("content-length", "0"))
            if expected_tail and partial.stat().st_size != start + expected_tail:
                raise IOError("downloaded size does not match Content-Length")
            if not valid_file(partial, remote_path):
                raise IOError("downloaded file failed format/shape validation")
            os.replace(partial, destination)
            return destination
        except Exception as error:
            if attempt + 1 == retries:
                raise
            retry_after = 0.0
            if isinstance(error, requests.HTTPError) and error.response is not None:
                retry_after = float(error.response.headers.get("Retry-After", "0") or 0)
            delay = max(retry_after, min(60.0, 2.0 ** attempt)) + random.uniform(0.0, 1.0)
            time.sleep(delay)
    raise RuntimeError(f"unreachable: failed to download {remote_path}")


def spread_selection(names: list[str], prefix: str, count: int) -> list[str]:
    """Round-robin across clip IDs so a tiny sample is not one contiguous moment."""
    groups: dict[str, list[str]] = {}
    for name in names:
        if not name.startswith(prefix + "_"):
            continue
        parts = name.split("_")
        groups.setdefault(parts[1], []).append(name)
    for values in groups.values():
        values.sort(key=lambda x: int(x.rsplit("_", 1)[-1]))
    selected: list[str] = []
    while len(selected) < count and any(groups.values()):
        for key in sorted(groups, key=lambda x: int(x)):
            if groups[key] and len(selected) < count:
                index = len(groups[key]) // 2
                selected.append(groups[key].pop(index))
    if len(selected) < count:
        raise ValueError(f"Only found {len(selected)} samples for {prefix}; requested {count}")
    return selected


def download_many(tasks: list[tuple[str, Path]], workers: int) -> list[Path]:
    results: list[Path] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download, remote, local): remote for remote, local in tasks}
        for index, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if index % 50 == 0 or index == len(tasks):
                print(f"  downloaded/validated {index}/{len(tasks)} files", flush=True)
    return results


def build_sample(root: Path, samples_per_group: int, workers: int) -> Path:
    lists_dir = root / "lists"
    for split, remote in LIST_PATHS.items():
        download(remote, lists_dir / f"{split}_data_list.txt")
    names = {
        split: [x for x in (lists_dir / f"{split}_data_list.txt").read_text().splitlines() if x]
        for split in LIST_PATHS
    }
    plan = [
        ("train", "S11", spread_selection(names["train"], "S11", samples_per_group)),
        ("train", "S21", spread_selection(names["train"], "S21", samples_per_group)),
        ("test", "S11", spread_selection(names["test"], "S11", samples_per_group)),
        ("test", "S21", spread_selection(names["test"], "S21", samples_per_group)),
        ("test", "S51", spread_selection(names["test"], "S51", samples_per_group)),
    ]

    label_tasks: list[tuple[str, Path]] = []
    for split, group, sample_ids in plan:
        for sample_id in sample_ids:
            remote = f"wifi_dataset/{split}_data/keypoint/{sample_id}.npy"
            label_tasks.append((remote, root / split / "keypoint" / f"{sample_id}.npy"))
    print(f"Downloading labels for {len(label_tasks)} planned samples...")
    download_many(label_tasks, workers)

    accepted: list[tuple[str, str, str, Path]] = []
    for split, group, sample_ids in plan:
        for sample_id in sample_ids:
            pose_path = root / split / "keypoint" / f"{sample_id}.npy"
            if np.load(pose_path, mmap_mode="r").shape == (1, 14, 3):
                accepted.append((split, group, sample_id, pose_path))
    if len(accepted) != 5 * samples_per_group:
        raise ValueError("Some planned samples were not single-person frames")

    csi_tasks: list[tuple[str, Path]] = []
    for split, _group, sample_id, _pose in accepted:
        remote = f"wifi_dataset/{split}_data/csi/{sample_id}.mat"
        csi_tasks.append((remote, root / split / "csi" / f"{sample_id}.mat"))
    print(f"Downloading CSI for {len(csi_tasks)} accepted samples...")
    download_many(csi_tasks, workers)

    records = []
    for split, group, sample_id, pose_path in accepted:
        csi_path = root / split / "csi" / f"{sample_id}.mat"
        records.append({
            "sample_id": sample_id,
            "split": split,
            "capture_group": group,
            "csi_path": str(csi_path.resolve()),
            "pose_path": str(pose_path.resolve()),
            "csi_bytes": csi_path.stat().st_size,
            "pose_bytes": pose_path.stat().st_size,
            "csi_sha256": sha256(csi_path),
            "pose_sha256": sha256(pose_path),
        })
    manifest = {
        "source": "https://www.kaggle.com/datasets/xjtuqianbo/wiception3d",
        "source_project": "https://aiotgroup.github.io/Person-in-WiFi-3D/",
        "dataset_version": VERSION,
        "selection": (
            "S11 and S21 train plus temporal test frames; S51 held-out capture group; "
            "one-person frames only"
        ),
        "resume_behavior": "valid completed files are skipped; .part files resume when the server returns HTTP 206",
        "samples": records,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest: {manifest_path} ({len(records)} samples)")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/wiception3d_small"))
    parser.add_argument("--samples-per-group", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    build_sample(args.root.resolve(), args.samples_per_group, args.workers)


if __name__ == "__main__":
    main()

