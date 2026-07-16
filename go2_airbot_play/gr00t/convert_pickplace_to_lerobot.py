# Copyright (c) 2026, WFMRFM project.
# SPDX-License-Identifier: Apache-2.0

"""Direct HDF5 -> GR00T-LeRobot (v2) converter for the Go2+Airbot maple-table pick-place
dataset produced by ``scripts/collect_pickplace_dataset.py`` (Arena RecorderManager, IK EEF
actions, static Go2 base).

Why not the Arena joint-space converter? That one is bimanual-joint oriented; this collection
is single-arm with (a) IK EEF-pose actions (not joint targets), (b) relative ``obs/joint_pos``
(we need absolute), and (c) an already-base-frame EEF (``eef_pose_in_base``). So we emit the
EEF-primary Phase-1 layout (matches ``modality_arm.json`` / ``go2_airbot_arm_data_config.py``)
directly. Absolute->relative EEF is left to the GR00T processor at train time.

Layout produced (per frame):
  observation.state          = [arm(6 abs joints), gripper(1 abs, g2_joint m)]    (7)
  observation.eef_pose       = [xyz, rot6d]  from eef_pose_in_base (base_link)     (9)
  action                     = [gripper(1 abs target, m)]                          (1)
  action.eef_pose            = [xyz, rot6d]  from the IK pose command (base frame) (9)
  observation.images.front   = go2_front_cam_rgb mp4  (exterior/front view)
  observation.images.wrist   = wrist_cam_rgb mp4       (wrist D435i view)
  annotation.human.task_description = task_index -> meta/tasks.jsonl

Run in the Arena/Isaac container:
  /isaac-sim/python.sh go2_airbot_play/gr00t/convert_pickplace_to_lerobot.py \
      --hdf5 scratchpad_out/datasets/go2airbot_pickplace.hdf5 \
      --out  scratchpad_out/datasets/go2airbot_pickplace/lerobot \
      --instruction "pick up the cube and place it in the bowl"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torchvision
from scipy.spatial.transform import Rotation

# Isaac Lab articulation DOF layout (21) verified from the recorded init pose:
ARM_SLICE = slice(12, 18)   # joint1..joint6 (absolute, rad)
GRIPPER_IDX = 20            # g2_joint driver (absolute opening, m)
FPS = 50                    # control rate: sim dt 0.005 * decimation 4 = 0.02 s
# Camera views: LeRobot key -> HDF5 camera_obs key. Order matters (fed to the model in order).
# overview_cam_rgb is a third-person eval view; omitted from the policy observation.
CAMERAS = {"front": "go2_front_cam_rgb", "wrist": "wrist_cam_rgb"}


def pos_quat_wxyz_to_xyz_rot6d(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """(N,3)+(N,4 wxyz) -> (N,9) [xyz, rot6d]; rot6d = rotation matrix first two rows flattened
    (matches gr00t EndEffectorPose._matrix_to_rot6d)."""
    pos = np.asarray(pos, np.float64).reshape(-1, 3)
    quat_wxyz = np.asarray(quat_wxyz, np.float64).reshape(-1, 4)
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    mats = Rotation.from_quat(quat_xyzw).as_matrix()
    rot6d = mats[:, :2, :].reshape(len(mats), 6)
    return np.concatenate([pos, rot6d], axis=1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hdf5", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path, help="LeRobot dataset dir to create.")
    p.add_argument("--instruction", default="pick up the cube and place it in the bowl")
    p.add_argument("--action-space", choices=["eef", "joint"], default="eef",
                   help="eef = relative-EEF arm (action.eef_pose 9D); joint = absolute arm joints (action 7D).")
    p.add_argument("--modality-json", default=None, type=Path,
                   help="Override the meta/modality.json template (default: modality_arm.json for eef, "
                        "modality_joint.json for joint).")
    a = p.parse_args()
    if a.modality_json is None:
        _here = Path(__file__).resolve().parent
        a.modality_json = _here / ("modality_joint.json" if a.action_space == "joint" else "modality_arm.json")

    out = a.out
    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    vid_dirs = {}
    for key in CAMERAS:
        vid_dirs[key] = out / "videos" / "chunk-000" / f"observation.images.{key}"
        vid_dirs[key].mkdir(parents=True, exist_ok=True)
    meta = out / "meta"
    meta.mkdir(parents=True, exist_ok=True)

    try:
        import pandas as pd  # noqa: F401
    except ImportError:
        pass

    import pandas as pd

    f = h5py.File(a.hdf5, "r")
    data = f["data"]
    demos = sorted(data.keys(), key=lambda x: int(x.split("_")[1]))

    episodes_info = []
    total = 0
    ep_out = 0          # contiguous output episode index (corrupt inputs are skipped)
    skipped = []
    for demo in demos:
        d = data[demo]
        # Read every needed array (incl. camera frames) up-front; skip the whole episode if any
        # read fails (e.g. a corrupt gzip chunk) BEFORE writing partial files. GR00T requires
        # every episode to carry all video keys, so a partial episode is not acceptable.
        try:
            n = int(d["actions"].shape[0])
            sj = np.asarray(d["states/articulation/robot/joint_position"])  # (n,21) absolute
            assert sj.shape[0] == n, f"{demo}: state/action length mismatch {sj.shape[0]} vs {n}"
            eef_obs = np.asarray(d["eef_pose_in_base"])                     # (n,7) pos+quat wxyz
            acts = np.asarray(d["actions"])                                # (n,8) [pos3,quat4,grip]
            grip_action = np.asarray(d["processed_actions"])[:, 9:10].astype(np.float64)  # (n,1) g2_joint
            cam_frames = {}
            for key, hdf5_cam in CAMERAS.items():
                fr = np.asarray(d[f"camera_obs/{hdf5_cam}"])               # (n,H,W,3) uint8
                assert fr.shape[0] == n, f"{demo} {hdf5_cam}: {fr.shape[0]} frames != {n}"
                cam_frames[key] = fr
        except Exception as e:
            skipped.append(demo)
            print(f"[conv] SKIP {demo}: unreadable ({type(e).__name__}: {str(e).splitlines()[0][:70]})", flush=True)
            continue

        arm = sj[:, ARM_SLICE].astype(np.float64)                      # (n,6) arm joints (absolute)
        grip_state = sj[:, GRIPPER_IDX:GRIPPER_IDX + 1].astype(np.float64)  # (n,1) g2_joint (achieved)
        obs_state = np.concatenate([arm, grip_state], axis=1)          # (n,7) arm(6)+gripper(1)

        cols = {
            "observation.state": list(obs_state),
            "timestamp": np.arange(n, dtype=np.float64) / FPS,
            "annotation.human.task_description": np.zeros(n, dtype=np.int64),
            "task_index": np.zeros(n, dtype=np.int64),
            "episode_index": np.full(n, ep_out, dtype=np.int64),
            "frame_index": np.arange(n, dtype=np.int64),
            "index": np.arange(total, total + n, dtype=np.int64),
            "next.reward": np.concatenate([np.zeros(n - 1), [1.0]]).astype(np.float64),
            "next.done": np.concatenate([np.zeros(n - 1, bool), [True]]),
        }
        if a.action_space == "joint":
            # ABSOLUTE joint action = [arm 6 joints (achieved), gripper 1 (commanded)] = 7D.
            # delta_indices [0..15] -> model predicts the next-16-step joint trajectory.
            act_joint = np.concatenate([arm, grip_action], axis=1)     # (n,7)
            cols["action"] = list(act_joint)
        else:
            obs_eef9 = pos_quat_wxyz_to_xyz_rot6d(eef_obs[:, :3], eef_obs[:, 3:7])  # (n,9)
            act_eef9 = pos_quat_wxyz_to_xyz_rot6d(acts[:, :3], acts[:, 3:7])        # (n,9)
            cols["observation.eef_pose"] = list(obs_eef9)
            cols["action"] = list(grip_action)                         # (n,1) gripper only
            cols["action.eef_pose"] = list(act_eef9)                   # (n,9) EEF pose
        df = pd.DataFrame(cols)
        df.to_parquet(out / "data" / "chunk-000" / f"episode_{ep_out:06d}.parquet")

        for key in CAMERAS:
            torchvision.io.write_video(str(vid_dirs[key] / f"episode_{ep_out:06d}.mp4"),
                                       torch.from_numpy(cam_frames[key]), FPS, video_codec="h264")

        episodes_info.append({"episode_index": ep_out, "tasks": [a.instruction], "length": n})
        total += n
        print(f"[conv] {demo} -> episode_{ep_out:06d}: {n} frames  eef|state|act ok", flush=True)
        ep_out += 1
    f.close()
    if skipped:
        print(f"[conv] SKIPPED {len(skipped)} corrupt episode(s): {skipped}", flush=True)

    # meta/
    with open(meta / "tasks.jsonl", "w") as fh:
        fh.write(json.dumps({"task_index": 0, "task": a.instruction}) + "\n")
    with open(meta / "episodes.jsonl", "w") as fh:
        for e in episodes_info:
            fh.write(json.dumps(e) + "\n")
    import shutil
    shutil.copy(a.modality_json, meta / "modality.json")
    _arm_grip_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
    if a.action_space == "joint":
        low_dim_features = {
            "observation.state": {"dtype": "float32", "shape": [7], "names": _arm_grip_names},
            "action": {"dtype": "float32", "shape": [7], "names": _arm_grip_names},
        }
    else:
        low_dim_features = {
            "observation.state": {"dtype": "float32", "shape": [7], "names": _arm_grip_names},
            "observation.eef_pose": {"dtype": "float32", "shape": [9]},
            "action": {"dtype": "float32", "shape": [1], "names": ["gripper"]},
            "action.eef_pose": {"dtype": "float32", "shape": [9]},
        }
    info = {
        "codebase_version": "v2.1", "robot_type": "go2_airbot_play",
        "total_episodes": len(episodes_info), "total_frames": total, "total_tasks": 1,
        "total_videos": len(episodes_info), "total_chunks": 1, "chunks_size": 1000, "fps": FPS,
        "splits": {"train": f"0:{len(episodes_info)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            **low_dim_features,
            **{f"observation.images.{key}": {"dtype": "video", "shape": [180, 320, 3],
                                             "names": ["height", "width", "channel"]} for key in CAMERAS},
        },
    }
    with open(meta / "info.json", "w") as fh:
        json.dump(info, fh, indent=4)
    print(f"[conv] DONE: {len(episodes_info)} episodes, {total} frames -> {out}")
    print("[conv] next: /isaac-sim/python.sh gr00t/data/stats.py --dataset-path "
          f"{out} --embodiment-tag NEW_EMBODIMENT")


if __name__ == "__main__":
    main()
