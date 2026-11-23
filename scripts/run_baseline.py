#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wifi_vision import (  # noqa: E402
    evaluate_and_save,
    load_complex_csi,
    load_single_person_pose,
    plot_csi_and_pose,
    plot_joint_errors,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data/wiception3d_small/manifest.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts/baseline_run")
    parser.add_argument("--components", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=20.0)
    args = parser.parse_args()

    metrics = evaluate_and_save(args.manifest, args.output, args.components, args.alpha)
    prediction_data = np.load(args.output / "test_predictions.npz")
    manifest = json.loads(args.manifest.read_text())["samples"]
    first_test = next(
        row for row in manifest if row["split"] == "test" and row["capture_group"] == "S51"
    )
    csi = load_complex_csi(first_test["csi_path"])
    truth = load_single_person_pose(first_test["pose_path"])
    plot_csi_and_pose(
        csi, truth, prediction_data["heldout_prediction"][0], args.output / "csi_and_pose.png"
    )
    plot_joint_errors(metrics, args.output / "joint_errors.png")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

