from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Order evidenced by Table 2 of the Person-in-WiFi 3D paper and root=[5,7] in its code.
JOINT_NAMES = [
    "neck", "head", "L shoulder", "R shoulder", "L elbow", "L hip",
    "R elbow", "R hip", "L hand", "L knee", "R hand", "R knee", "L ankle", "R ankle",
]
SKELETON_EDGES = [
    (0, 1), (0, 2), (2, 4), (4, 8), (0, 3), (3, 6), (6, 10),
    (2, 5), (3, 7), (5, 7), (5, 9), (9, 12), (7, 11), (11, 13),
]


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_complex_csi(path: str | Path) -> np.ndarray:
    """Load Person-in-WiFi 3D HDF5 MATLAB CSI as complex [time, subcarrier, tx, rx]."""
    with h5py.File(path, "r") as handle:
        raw = handle["csi_out"][()]
    if raw.dtype.fields and {"real", "imag"}.issubset(raw.dtype.fields):
        csi = raw["real"] + 1j * raw["imag"]
    else:
        csi = np.asarray(raw, dtype=np.complex128)
    if csi.ndim != 4:
        raise ValueError(f"Expected 4D CSI, got {csi.shape} from {path}")
    return csi


def load_single_person_pose(path: str | Path) -> np.ndarray:
    pose = np.asarray(np.load(path), dtype=np.float64)
    if pose.shape != (1, 14, 3):
        raise ValueError(f"Expected one-person pose [1,14,3], got {pose.shape} from {path}")
    return pose[0]


def csi_features(csi: np.ndarray) -> np.ndarray:
    """Compact amplitude/phase moments; avoids keeping raw frames in RAM."""
    amplitude = np.log1p(np.abs(csi))
    phase = np.unwrap(np.angle(csi), axis=0)
    phase_delta = np.diff(phase, axis=0)
    blocks = [
        amplitude.mean(axis=0),
        amplitude.std(axis=0),
        np.cos(phase_delta).mean(axis=0),
        np.sin(phase_delta).mean(axis=0),
        phase_delta.std(axis=0),
    ]
    features = np.concatenate([block.ravel() for block in blocks]).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("CSI feature vector contains NaN or infinity")
    return features


def load_manifest_dataset(manifest_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    manifest_path = Path(manifest_path)
    records = json.loads(manifest_path.read_text(encoding="utf-8"))["samples"]
    features, poses, splits, groups = [], [], [], []
    for record in records:
        features.append(csi_features(load_complex_csi(record["csi_path"])))
        poses.append(load_single_person_pose(record["pose_path"]))
        splits.append(record["split"])
        groups.append(record["capture_group"])
    return (
        np.stack(features),
        np.stack(poses),
        np.asarray(splits),
        np.asarray(groups),
    )


def mpjpe_mm(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.linalg.norm(y_true - y_pred, axis=-1).mean() * 1000.0)


def root_relative(poses: np.ndarray) -> np.ndarray:
    mid_hip = poses[:, [5, 7]].mean(axis=1, keepdims=True)
    return poses - mid_hip


def per_joint_error_mm(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.linalg.norm(y_true - y_pred, axis=-1).mean(axis=0) * 1000.0


def train_ridge_baseline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    max_components: int = 64,
    alpha: float = 20.0,
) -> Pipeline:
    components = max(2, min(max_components, x_train.shape[0] - 1, x_train.shape[1]))
    model = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=components, whiten=True, random_state=42)),
        ("ridge", Ridge(alpha=alpha)),
    ])
    model.fit(x_train, y_train.reshape(len(y_train), -1))
    return model


def evaluate_and_save(
    manifest_path: str | Path,
    output_dir: str | Path,
    max_components: int = 64,
    alpha: float = 20.0,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    x, y, split, groups = load_manifest_dataset(manifest_path)
    train_mask = split == "train"
    in_domain_mask = (split == "test") & np.isin(groups, ["S11", "S21"])
    heldout_mask = (split == "test") & (groups == "S51")
    if train_mask.sum() < 2 or in_domain_mask.sum() < 1 or heldout_mask.sum() < 1:
        raise ValueError("Manifest needs at least 2 train and 1 test samples")

    model = train_ridge_baseline(x[train_mask], y[train_mask], max_components, alpha)
    train_mean = y[train_mask].mean(axis=0)

    def score(mask: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
        pred = model.predict(x[mask]).reshape(-1, 14, 3)
        truth = y[mask]
        mean_pose = np.broadcast_to(train_mean, truth.shape)
        result = {
            "n": int(mask.sum()),
            "groups": sorted(np.unique(groups[mask]).tolist()),
            "ridge_mpjpe_mm": mpjpe_mm(truth, pred),
            "mean_pose_mpjpe_mm": mpjpe_mm(truth, mean_pose),
            "ridge_root_relative_mpjpe_mm": mpjpe_mm(root_relative(truth), root_relative(pred)),
            "mean_pose_root_relative_mpjpe_mm": mpjpe_mm(root_relative(truth), root_relative(mean_pose)),
        }
        return result, truth, pred

    in_domain, in_truth, in_pred = score(in_domain_mask)
    heldout, heldout_truth, heldout_pred = score(heldout_mask)

    metrics = {
        "task": "single-person absolute 3D joint regression",
        "split": "S11+S21 train; temporal in-domain test on S11+S21; capture-group holdout on S51",
        "n_train": int(train_mask.sum()),
        "n_features": int(x.shape[1]),
        "capture_groups_train": sorted(np.unique(groups[train_mask]).tolist()),
        "in_domain_temporal": in_domain,
        "heldout_capture_group": heldout,
        "per_joint_error_mm": dict(
            zip(JOINT_NAMES, per_joint_error_mm(heldout_truth, heldout_pred).round(3).tolist())
        ),
        "important_limit": (
            "This is a tiny engineering baseline on a capture-group holdout, not a reproduction "
            "of the paper and not a subject-disjoint benchmark."
        ),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    joblib.dump(model, output_dir / "ridge_3d_baseline.joblib")
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        in_domain_truth=in_truth,
        in_domain_prediction=in_pred,
        heldout_truth=heldout_truth,
        heldout_prediction=heldout_pred,
    )
    return metrics


def plot_csi_and_pose(
    csi: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray | None,
    output_path: str | Path,
) -> None:
    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    ax1 = fig.add_subplot(1, 2, 1)
    heat = np.log1p(np.abs(csi[:, :, 0, 0])).T
    image = ax1.imshow(heat, aspect="auto", origin="lower", cmap="viridis")
    ax1.set_title("CSI amplitude: one Tx/Rx pair")
    ax1.set_xlabel("Packet in 20-sample window")
    ax1.set_ylabel("Subcarrier (30)")
    fig.colorbar(image, ax=ax1, label="log(1 + amplitude)")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    for pose, label, color, style in [
        (truth, "Ground truth", "#0072B2", "-"),
        (prediction, "Ridge prediction", "#D55E00", "--"),
    ]:
        if pose is None:
            continue
        ax2.scatter(pose[:, 0], pose[:, 2], pose[:, 1], s=22, color=color, label=label)
        for a, b in SKELETON_EDGES:
            ax2.plot(
                pose[[a, b], 0], pose[[a, b], 2], pose[[a, b], 1],
                color=color, linestyle=style, linewidth=1.5,
            )
    ax2.set_title("3D skeleton (actual coordinates)")
    ax2.set_xlabel("Horizontal (m)")
    ax2.set_ylabel("Depth (m)")
    ax2.set_zlabel("Vertical (m)")
    ax2.legend(loc="upper left")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_joint_errors(metrics: dict, output_path: str | Path) -> None:
    names = list(metrics["per_joint_error_mm"])
    values = [metrics["per_joint_error_mm"][name] for name in names]
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    ax.bar(np.arange(len(names)), values, color="#0072B2")
    ax.set_xticks(np.arange(len(names)), names, rotation=45, ha="right")
    ax.set_ylabel("Mean Euclidean error (mm)")
    ax.set_title("Held-out capture-group error by joint")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

