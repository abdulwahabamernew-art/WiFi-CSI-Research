from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wifi_vision import csi_features, load_complex_csi, load_single_person_pose, mpjpe_mm, root_relative


def test_real_sample_shapes():
    sample = ROOT / "data/wiception_sample"
    csi = load_complex_csi(sample / "S11_01_308.mat")
    pose = load_single_person_pose(sample / "S11_01_308.npy")
    assert csi.shape == (20, 30, 3, 3)
    assert pose.shape == (14, 3)
    assert csi_features(csi).shape == (1350,)


def test_metrics_zero_for_identical_pose():
    pose = np.zeros((2, 14, 3), dtype=float)
    assert mpjpe_mm(pose, pose) == 0.0
    assert np.array_equal(root_relative(pose), pose)

