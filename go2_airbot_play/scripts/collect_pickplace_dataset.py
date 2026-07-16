# Copyright (c) 2026, WFMRFM project.
# SPDX-License-Identifier: Apache-2.0

"""Headless scripted-expert pick-and-place dataset collection for the Go2+Airbot maple-table env.

Swaps the arm to a differential-IK (absolute EE pose) action, drives a top-down pick-and-place
with UNIFORM speed (lerp position + slerp orientation), randomises the cube/bowl per episode, and
records obs (3 cameras + proprio) + actions to a robomimic-style HDF5 via Arena's RecorderManager
(EXPORT_SUCCEEDED_ONLY). Convert the HDF5 to GR00T-LeRobot with
``isaaclab_arena_gr00t/lerobot/convert_hdf5_to_lerobot.py``.

    /isaac-sim/python.sh go2_airbot_play/scripts/collect_pickplace_dataset.py \
        --num_demos 10 --dataset_file /path/out.hdf5
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

_p = argparse.ArgumentParser()
_p.add_argument("--num_demos", type=int, default=10, help="Number of SUCCESSFUL demos to collect.")
_p.add_argument("--max_attempts", type=int, default=0, help="Max episodes to try (0 = 3*num_demos).")
_p.add_argument("--dataset_file", type=str,
                default="/mnt/nas2/users/shkim/work/projects/wfmrfm/scratchpad_out/datasets/go2airbot_pickplace.hdf5")
_p.add_argument("--cam_h", type=int, default=180)
_p.add_argument("--cam_w", type=int, default=320)
_p.add_argument("--seed", type=int, default=0)
_a = _p.parse_args()

app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

import os  # noqa: E402
import random  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402

from isaaclab.controllers import DifferentialIKControllerCfg  # noqa: E402
from isaaclab.envs.mdp.actions.actions_cfg import (  # noqa: E402
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.managers import ActionTermCfg, DatasetExportMode, RecorderTerm, RecorderTermCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402
from isaaclab_arena.cli.isaaclab_arena_cli import (  # noqa: E402
    arena_env_builder_cfg_from_argparse,
    get_isaaclab_arena_cli_parser,
)
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder  # noqa: E402
from isaaclab_arena.utils.isaaclab_utils.recorders import ArenaEnvRecorderManagerCfg  # noqa: E402

import go2_airbot_play.environment as gae  # noqa: E402

SPEED = 0.12       # m/s linear
ANG_SPEED = 50.0   # deg/s angular
TABLE_Z = 0.40     # cuboid top (arm comfortable working height)


@configclass
class IKActionCfg:
    arm_action: ActionTermCfg = DifferentialInverseKinematicsActionCfg(
        asset_name="robot", joint_names=["joint[1-6]"], body_name="g2_base_link",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        scale=1.0,
    )
    gripper_action: ActionTermCfg = BinaryJointPositionActionCfg(
        asset_name="robot", joint_names=["g2_joint", "g2_left_joint", "g2_right_joint"],
        open_command_expr={"g2_joint": 0.072, "g2_left_joint": 0.036, "g2_right_joint": 0.036},
        close_command_expr={"g2_joint": 0.0, "g2_left_joint": 0.0, "g2_right_joint": 0.0},
    )


class _EefPoseInBaseRecorder(RecorderTerm):
    """Records the arm-tip (g2_base_link) ABSOLUTE pose in the base_link frame each step:
    [x, y, z, qw, qx, qy, qz]. GR00T's processor converts abs -> relative on its side."""

    def record_pre_step(self):
        robot = self._env.scene["robot"]
        if not hasattr(self, "_ie"):
            bn = list(robot.data.body_names)
            self._ie = bn.index("g2_base_link")
            self._ib = bn.index("base_link")
        pos = wp.to_torch(robot.data.body_pos_w)
        quat = wp.to_torch(robot.data.body_quat_w)
        if pos.dim() == 2:  # (num_bodies, 3) single-env fallback
            pos = pos.unsqueeze(0); quat = quat.unsqueeze(0)
        p, q = subtract_frame_transforms(
            pos[:, self._ib], quat[:, self._ib], pos[:, self._ie], quat[:, self._ie]
        )
        return "eef_pose_in_base", torch.cat([p, q], dim=-1)


@configclass
class _EefPoseInBaseRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = _EefPoseInBaseRecorder


@configclass
class PickPlaceRecorderManagerCfg(ArenaEnvRecorderManagerCfg):
    """Arena recorder + the absolute arm-tip pose (base_link frame) for GR00T."""

    record_eef_pose_in_base = _EefPoseInBaseRecorderCfg()


def slerp(q0, q1, t):
    q0 = np.asarray(q0, float) / np.linalg.norm(q0); q1 = np.asarray(q1, float) / np.linalg.norm(q1)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1 = -q1; d = -d
    if d > 0.9995:
        r = q0 + t * (q1 - q0); return r / np.linalg.norm(r)
    th = math.acos(d) * t
    q2 = q1 - q0 * d; q2 /= np.linalg.norm(q2)
    return q0 * math.cos(th) + q2 * math.sin(th)


def ang_between(q0, q1):
    return 2.0 * math.acos(min(1.0, abs(float(np.dot(np.asarray(q0, float), np.asarray(q1, float))))))


def main():
    random.seed(_a.seed); np.random.seed(_a.seed)
    num_demos = _a.num_demos
    max_attempts = _a.max_attempts or (3 * num_demos)
    out_dir = os.path.dirname(_a.dataset_file)
    fname = os.path.splitext(os.path.basename(_a.dataset_file))[0]
    os.makedirs(out_dir, exist_ok=True)

    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
    env_args = argparse.Namespace(
        enable_cameras=True, object="rubiks_cube_hot3d_robolab", destination="bowl_ycb_robolab",
        table_z=TABLE_Z, robot_x=-0.15, episode_length_s=1000.0, fix_base=True, hdr=None,
    )
    factory = gae.Go2AirbotPlayMapleTablePickPlaceEnvironment()
    arena_env = factory.get_env(env_args)
    arena_env.embodiment.action_config = IKActionCfg()
    cc = arena_env.embodiment.camera_config
    for cam in (cc.wrist_cam, cc.go2_front_cam, cc.overview_cam):
        cam.height = _a.cam_h; cam.width = _a.cam_w  # smaller (16:9 preserved) -> manageable dataset

    builder = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli))
    env_cfg, _ = builder.compose_manager_cfg()
    env_cfg.recorders = PickPlaceRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = out_dir
    env_cfg.recorders.dataset_filename = fname
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    # NOTE: the env auto-resets on the success termination INSIDE env.step, and that reset's
    # record_pre_reset auto-exports the finished episode (success taken from the termination
    # term). So we DON'T export manually; we just detect the terminated flag.
    env = builder.make_registered(env_cfg=env_cfg)
    u = env.unwrapped
    dev = u.device
    step_dt = float(u.step_dt)
    step_len = SPEED * step_dt
    max_ang = math.radians(ANG_SPEED) * step_dt
    scene = u.scene
    robot = scene["robot"]
    ee_i = list(robot.data.body_names).index("g2_base_link")
    base = np.array([-0.15, 0.0, 0.40])
    DOWN = np.array([0.0, 1.0, 0.0, 0.0])
    success_term = env_cfg.terminations.success

    def ee_w():
        return wp.to_torch(robot.data.body_pos_w).reshape(-1, 3)[ee_i].cpu().numpy()

    def ee_quat():
        return wp.to_torch(robot.data.body_quat_w).reshape(-1, 4)[ee_i].cpu().numpy()

    def set_obj(name, x, y, z):
        pose = torch.tensor([[x, y, z, 1.0, 0.0, 0.0, 0.0]], device=dev, dtype=torch.float32)
        scene[name].write_root_pose_to_sim(pose)
        scene[name].write_root_velocity_to_sim(torch.zeros((1, 6), device=dev))

    def is_success():
        return bool(success_term.func(u, **success_term.params).reshape(-1)[0])

    def act(pos_w, quat, grip):
        a = torch.zeros((1, 8), device=dev)
        a[0, 0:3] = torch.tensor(pos_w - base, device=dev, dtype=torch.float32)
        a[0, 3:7] = torch.tensor(quat, device=dev, dtype=torch.float32)
        a[0, 7] = grip
        return a

    def goto(cur_p, cur_q, tgt_p, tgt_q, grip):
        cur_p = np.asarray(cur_p, float); tgt_p = np.asarray(tgt_p, float)
        n = max(1, int(math.ceil(np.linalg.norm(tgt_p - cur_p) / step_len)),
                int(math.ceil(ang_between(cur_q, tgt_q) / max_ang)))
        for i in range(1, n + 1):
            res = env.step(act(cur_p + (tgt_p - cur_p) * (i / n), slerp(cur_q, tgt_q, i / n), grip))
            if bool(res[2].reshape(-1)[0]):  # terminated (success) -> auto-reset+auto-export happened
                return tgt_p, tgt_q, True
        return tgt_p, tgt_q, False

    def hold(pos, quat, grip, steps):
        for _ in range(steps):
            res = env.step(act(pos, quat, grip))
            if bool(res[2].reshape(-1)[0]):
                return True
        return False

    def run_expert(cube_xy, bowl_xy):
        cx, cy = cube_xy; bx, by = bowl_xy
        KEYS = [
            (np.array([cx, cy, 0.62]), +1, 5),
            (np.array([cx, cy, 0.48]), +1, 8),
            (np.array([cx, cy, 0.48]), -1, 30),
            (np.array([cx, cy, 0.64]), -1, 5),
            (np.array([bx, by, 0.64]), -1, 5),
            (np.array([bx, by, 0.57]), -1, 8),
            (np.array([bx, by, 0.57]), +1, 15),
        ]
        cur_p, cur_q = ee_w(), ee_quat()
        for tgt, grip, h in KEYS:
            cur_p, cur_q, done = goto(cur_p, cur_q, tgt, DOWN, grip)
            if not done:
                done = hold(cur_p, cur_q, grip, h)
            if done:
                return True
        return False

    print(f"[collect] target {num_demos} demos, max {max_attempts} attempts; cam {_a.cam_w}x{_a.cam_h}", flush=True)
    valid = 0  # count of successful (non-empty) demos; recorder's counter also tallies empty artifacts
    for ep in range(max_attempts):
        env.reset()
        cube_xy = (round(random.uniform(0.27, 0.33), 3), round(random.uniform(0.06, 0.16), 3))
        bowl_xy = (round(random.uniform(0.27, 0.33), 3), round(random.uniform(-0.18, -0.10), 3))
        # Place objects at rest height right after reset (no settling steps needed) so the
        # recorded episode runs cleanly from reset to success (no empty-episode artifacts).
        set_obj("rubiks_cube_hot3d_robolab", *cube_xy, TABLE_Z + 0.03)
        set_obj("bowl_ycb_robolab", *bowl_xy, TABLE_Z + 0.03)
        ok = run_expert(cube_xy, bowl_xy)  # success termination -> env auto-resets & auto-exports the demo
        valid += int(ok)
        print(f"[ep {ep:03d}] cube={cube_xy} bowl={bowl_xy} success={ok} valid={valid}/{num_demos}", flush=True)
        if valid >= num_demos:
            break

    env.close()
    # Drop any 0-sample demo groups (recorder can emit a stray empty post-reset episode) and
    # renumber to contiguous demo_0.. ; fix the 'total' sample count.
    import h5py
    with h5py.File(_a.dataset_file, "a") as hf:
        data = hf["data"]
        empties = [d for d in list(data.keys()) if int(data[d].attrs.get("num_samples", 0)) == 0]
        for d in empties:
            del data[d]
        for i, d in enumerate(sorted(data.keys(), key=lambda x: int(x.split("_")[1]))):
            if d != f"demo_{i}":
                data.move(d, f"demo_{i}")
        data.attrs["total"] = int(sum(int(data[d].attrs["num_samples"]) for d in data))
        n_valid = len(data.keys())
    print(f"[done] {n_valid} valid demos (removed {len(empties)} empty) -> {_a.dataset_file}", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
