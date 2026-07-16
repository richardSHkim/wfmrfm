# Copyright (c) 2026, WFMRFM project.
# SPDX-License-Identifier: Apache-2.0

"""Method-A post-process: add single-arm EEF (xyz + rot6d, 9D) columns to a GR00T-LeRobot
dataset produced by Arena's joint-space HDF5->LeRobot converter, without touching the
(bimanual-hardcoded) Arena converter in the submodule.

Pipeline position:
    collect HDF5 -> Arena convert_hdf5_to_lerobot.py (joint-space) -> THIS SCRIPT -> stats.py -> finetune

What it does, per episode parquet ``episode_XXXXXX.parquet`` (index i):
  1. Reads the single-arm end-effector pose from the matching HDF5 trajectory
     (obs group for ``observation.eef_pose``, action group for ``action.eef_pose``),
     dropping the last frame to match the converter's ``[:-1]`` framing. The recorded pose
     MUST be in the ``base_link`` (arm-mount) frame, NOT world — the Go2 base moves, so a
     world-frame EEF drifts and couples the arm target to locomotion. This script does not
     transform frames; it trusts the HDF5 pose is already ``base_link``-relative.
  2. Converts (pos[3], quat_wxyz[4]) -> 9D (xyz + rot6d), where rot6d is the first two
     ROWS of the rotation matrix flattened -- identical to GR00T's
     ``EndEffectorPose._matrix_to_rot6d`` (gr00t/data/state_action/pose.py). This is the
     representation the modality config's ``format=XYZ_ROT6D`` expects the dataset to store;
     absolute->relative is handled later by the processor.
  3. Adds columns ``observation.eef_pose`` and ``action.eef_pose`` and rewrites the parquet.
  4. Installs the canonical EEF-primary ``modality.json`` (this dir) into ``meta/``.

The HDF5 field names are configurable because the real teleop collection schema is TBD; the
defaults document the expected contract. Quaternions are assumed wxyz (Isaac Lab convention).

Run inside the GR00T uv env (needs numpy, scipy, pandas, pyarrow, h5py):
    python add_single_arm_eef.py \
        --lerobot-dir <data_root>/<hdf5_stem>/lerobot \
        --hdf5 <data_root>/<hdf5_name>.hdf5
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

_HERE = Path(__file__).resolve().parent


def quat_wxyz_pos_to_xyz_rot6d(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """(N,3) position + (N,4) wxyz quaternion -> (N,9) [x,y,z, rot6d] matching GR00T rot6d.

    rot6d = rotation_matrix[:2, :].flatten() (first two rows), see
    EndEffectorPose._matrix_to_rot6d in gr00t/data/state_action/pose.py.
    """
    pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64).reshape(-1, 4)
    assert len(pos) == len(quat_wxyz), f"pos/quat length mismatch: {len(pos)} vs {len(quat_wxyz)}"
    # scipy expects xyzw; our quats are wxyz.
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    mats = Rotation.from_quat(quat_xyzw).as_matrix()  # (N,3,3)
    rot6d = mats[:, :2, :].reshape(len(mats), 6)  # first two rows, row-major
    return np.concatenate([pos, rot6d], axis=1)  # (N,9)


def _read_eef(group: h5py.Group, pos_key: str, quat_key: str, length: int) -> np.ndarray:
    """Read (pos, quat) from an HDF5 group, drop the last frame, return (length, 9) xyz+rot6d."""
    assert pos_key in group, f"'{pos_key}' not in HDF5 group (keys: {list(group.keys())})"
    assert quat_key in group, f"'{quat_key}' not in HDF5 group (keys: {list(group.keys())})"
    pos = np.asarray(group[pos_key])[:-1]
    quat = np.asarray(group[quat_key])[:-1]
    out = quat_wxyz_pos_to_xyz_rot6d(pos, quat)
    assert len(out) == length, f"eef length {len(out)} != parquet length {length}"
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lerobot-dir", required=True, type=Path, help="Converter output dir (contains data/ meta/).")
    p.add_argument("--hdf5", required=True, type=Path, help="Source HDF5 the converter consumed.")
    p.add_argument("--obs-eef-pos", default="eef_pos", help="obs-group EEF position key (m, base_link frame).")
    p.add_argument("--obs-eef-quat", default="eef_quat", help="obs-group EEF quaternion key (wxyz, base_link frame).")
    p.add_argument("--action-eef-pos", default="eef_pos", help="action-group EEF position key (m, base_link frame).")
    p.add_argument("--action-eef-quat", default="eef_quat", help="action-group EEF quaternion key (wxyz, base_link frame).")
    p.add_argument("--modality-json", default=_HERE / "modality.json", type=Path,
                   help="Canonical EEF-primary modality.json to install into meta/.")
    a = p.parse_args()

    data_dir = a.lerobot_dir / "data"
    parquets = sorted(data_dir.rglob("episode_*.parquet"))
    assert parquets, f"no episode parquets under {data_dir}"

    with h5py.File(a.hdf5, "r") as hf:
        hdf5_data = hf["data"]
        traj_ids = list(hdf5_data.keys())  # converter enumerates in this order -> episode_index
        assert len(traj_ids) >= len(parquets), (
            f"{len(traj_ids)} HDF5 trajectories < {len(parquets)} parquets"
        )
        for pq in parquets:
            ep = int(pq.stem.split("_")[1])
            traj = hdf5_data[traj_ids[ep]]
            df = pd.read_parquet(pq)
            n = len(df)
            obs_eef = _read_eef(traj["obs"], a.obs_eef_pos, a.obs_eef_quat, n)
            act_eef = _read_eef(traj["action"], a.action_eef_pos, a.action_eef_quat, n)
            df["observation.eef_pose"] = [row for row in obs_eef]
            df["action.eef_pose"] = [row for row in act_eef]
            df.to_parquet(pq)
            print(f"[eef] episode {ep:06d}: +observation.eef_pose +action.eef_pose ({n} frames)")

    # Install the canonical EEF-primary modality.json.
    meta_modality = a.lerobot_dir / "meta" / "modality.json"
    shutil.copy(a.modality_json, meta_modality)
    print(f"[eef] installed {a.modality_json} -> {meta_modality}")
    print("[eef] DONE. Next: regenerate stats, e.g.\n"
          "  python gr00t/data/stats.py --dataset-path "
          f"{a.lerobot_dir} --embodiment-tag NEW_EMBODIMENT")


if __name__ == "__main__":
    main()
