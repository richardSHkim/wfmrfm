# Copyright (c) 2026, WFMRFM project.
# SPDX-License-Identifier: Apache-2.0

"""Custom Arena closed-loop policy for the Go2 + Airbot Play GR00T (Phase-1) embodiment.

Why a bespoke policy instead of Arena's ``Gr00tRemoteClosedloopPolicy``? That class routes
GR00T output through a G1/GR1/DROID-specific *joint-space* translation bridge
(``gr00t_core.build_gr00t_action_tensor`` + ``joints_conversion``) whose ``task_mode`` /
``embodiment_tag`` branches are hard-coded, and it does not handle a single-arm **EEF** action.
Our Phase-1 policy is single-arm relative-EEF + gripper, so we talk to the same GR00T server
(GR00T's native ``PolicyClient`` over ZMQ) but do our own, fully out-of-tree translation:

    observation (built here, straight from the sim scene)
        video.front / video.wrist   (uint8, resized to the dataset resolution)
        state.arm_eef (9)            absolute EEF pose in base_link frame [xyz, rot6d]
        state.arm (6)                absolute arm joint angles (joint1..joint6)
        state.gripper (1)            absolute gripper opening (g2_joint, m)
        language.annotation.human.task_description
              |
              v  GR00T server (real checkpoint OR ReplayPolicy)
        action.arm_eef (H, 9)        absolute EEF target [xyz, rot6d] (base_link frame)
        action.gripper (H, 1)        absolute opening target (m)
              |
              v  translate here
    env action (num_envs, 8) = [x, y, z, qw, qx, qy, qz, grip_binary]
        -> DifferentialInverseKinematicsActionCfg (arm, 7) + BinaryJointPositionActionCfg (1)

The arm EEF is stored/served **absolute in the base_link frame** (GR00T's processor converts
absolute<->relative internally); the IK term's pose command is likewise in the robot base
frame, so ``action.arm_eef[:3]`` feeds the IK position directly and ``rot6d`` -> quaternion
(wxyz) feeds the orientation. This mirrors ``scripts/collect_pickplace_dataset.py``.

Register/select via a dotted ``--policy_type``::

    --policy_type go2_airbot_play.eval.gr00t_eef_policy.Go2AirbotEefClosedloopPolicy

Importing this module runs ``@register_policy`` so ``build_policy_from_cli`` can read the
typed config and turn its fields into ``--remote_host`` / ``--remote_port`` / ... CLI flags.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from dataclasses import dataclass

from gr00t.policy.server_client import PolicyClient as Gr00tPolicyClient

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg


@dataclass
class Go2AirbotEefClosedloopPolicyCfg(PolicyCfg):
    """Config for the Go2 + Airbot Play GR00T EEF closed-loop policy.

    Every field below becomes an auto-generated ``--<field>`` CLI flag (via
    ``build_policy_from_cli``). Field names deliberately avoid collisions with the shared
    runner flags (``device``, ``num_envs``, ``language_instruction``, ...).
    """

    remote_host: str = "localhost"
    """GR00T policy server hostname."""

    remote_port: int = 5555
    """GR00T policy server port."""

    remote_api_token: str | None = None
    """Optional policy-server API token."""

    remote_kill_on_exit: bool = False
    """Kill the GR00T server when the rollout ends. Declared here so it also materializes as
    the ``--remote_kill_on_exit`` CLI flag that Arena's ``policy_runner`` reads at shutdown
    (the runner references ``args_cli.remote_kill_on_exit`` but defines it nowhere itself)."""

    action_horizon: int = 16
    """Length of the action chunk the server returns per query (delta-index count)."""

    execution_horizon: int = 16
    """Number of chunk steps executed open-loop before re-querying the server. Must equal the
    server-side ReplayPolicy ``execution_horizon`` for a faithful replay; <= action_horizon."""

    gripper_open_threshold: float = 0.036
    """Gripper opening (m) above which the binary gripper term is commanded OPEN."""

    policy_device: str = "cuda"
    """Device for Arena-side tensors built by this policy."""

    front_cam_key: str = "go2_front_cam_rgb"
    """camera_obs key for the Go2 front (exterior) view -> GR00T video 'front'."""

    wrist_cam_key: str = "wrist_cam_rgb"
    """camera_obs key for the wrist D435i view -> GR00T video 'wrist'."""

    image_height: int = 180
    """Height the RGB frames are resized to before sending to the server."""

    image_width: int = 320
    """Width the RGB frames are resized to before sending to the server."""


def _decode_ndarray(v):
    """Return a numpy array from a server action value.

    Cross-version shim: the GR00T model server (Isaac-GR00T ab88b50) serializes ndarrays with
    ``msgpack-numpy`` (``{b'nd', b'type', b'shape', b'data'}``), but the Arena-vendored client
    (e29d8fc) decodes with a different tag, so array values arrive as raw dicts. Reconstruct
    them here; pass through if already an ndarray.
    """
    if isinstance(v, np.ndarray):
        return v
    if isinstance(v, dict):
        get = lambda *ks: next((v[k] for k in ks if k in v), None)  # noqa: E731 (bytes/str keys)
        if get(b"nd", "nd"):
            dtype = get(b"type", "type")
            dtype = dtype.decode() if isinstance(dtype, bytes) else dtype
            shape = tuple(get(b"shape", "shape"))
            return np.frombuffer(get(b"data", "data"), dtype=np.dtype(dtype)).reshape(shape)
    return np.asarray(v)


def _rot6d_to_quat_wxyz(rot6d: np.ndarray) -> np.ndarray:
    """Convert a 6D rotation (first two rows of R, flattened) to a wxyz quaternion.

    Inverse of gr00t ``EndEffectorPose._matrix_to_rot6d`` (which takes ``R[:2, :]``). Rows are
    Gram-Schmidt orthonormalized to tolerate un-normalized model output.
    """
    from scipy.spatial.transform import Rotation

    r0 = rot6d[:3].astype(np.float64)
    r1 = rot6d[3:6].astype(np.float64)
    n0 = np.linalg.norm(r0)
    r0 = r0 / n0 if n0 > 1e-8 else np.array([1.0, 0.0, 0.0])
    r1 = r1 - np.dot(r0, r1) * r0
    n1 = np.linalg.norm(r1)
    r1 = r1 / n1 if n1 > 1e-8 else np.array([0.0, 1.0, 0.0])
    r2 = np.cross(r0, r1)
    mat = np.stack([r0, r1, r2], axis=0)  # rows == R[:3, :]
    quat_xyzw = Rotation.from_matrix(mat).as_quat()
    return quat_xyzw[[3, 0, 1, 2]]  # -> wxyz


@register_policy
class Go2AirbotEefClosedloopPolicy(PolicyBase[Go2AirbotEefClosedloopPolicyCfg]):
    """GR00T closed-loop policy: single-arm relative-EEF + gripper, static base (Phase-1)."""

    name = "go2airbot_gr00t_eef"

    def __init__(self, config: Go2AirbotEefClosedloopPolicyCfg):
        super().__init__(config)
        self.device = config.policy_device
        self.task_description: str | None = None

        # Per-env action chunk buffer: (num_envs, execution_horizon, 8) + a per-call cursor.
        self._chunk: torch.Tensor | None = None
        self._cursor: int = 0

        # Sim joint / body indices, resolved lazily on the first observation.
        self._arm_joint_ids: list[int] | None = None
        self._gripper_joint_id: int | None = None
        self._eef_body_id: int | None = None
        self._base_body_id: int | None = None

        client = Gr00tPolicyClient(
            host=config.remote_host, port=config.remote_port, api_token=config.remote_api_token, strict=False
        )
        if not client.ping():
            raise ConnectionError(
                f"Cannot reach GR00T policy server at {config.remote_host}:{config.remote_port}. "
                "Start it with go2_airbot_play/gr00t/eval/serve_go2airbot_gr00t.py first."
            )
        self._client: Gr00tPolicyClient | None = client

    # --------------------------------------------------------------------- #
    # Policy interface
    # --------------------------------------------------------------------- #

    def set_task_description(self, task_description: str | None) -> str:
        assert task_description, "No language instruction available for the GR00T policy."
        self.task_description = task_description
        return self.task_description

    def get_action(self, env: gym.Env, observation: dict) -> torch.Tensor:
        assert self._client is not None, "Policy has been closed."
        if self._chunk is None or self._cursor >= self._chunk.shape[1]:
            self._chunk = self._query_chunk(env, observation)
            self._cursor = 0
        action = self._chunk[:, self._cursor]
        self._cursor += 1
        return action

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        assert self._client is not None, "Policy has been closed."
        # Restart the server-side replay/plan and drop the local buffer so the next
        # get_action re-queries from the top of the (reset) episode.
        self._client.reset()
        self._chunk = None
        self._cursor = 0

    @property
    def is_remote(self) -> bool:
        return True

    def shutdown_remote(self, kill_server: bool = False) -> None:
        client = self._client
        if client is None:
            return
        try:
            if kill_server:
                try:
                    client.kill_server()
                except Exception as e:  # noqa: BLE001 - best-effort shutdown
                    print(f"[go2airbot_gr00t_eef] kill_server failed: {e}")
            socket = getattr(client, "socket", None)
            context = getattr(client, "context", None)
            if socket is not None:
                socket.close(linger=0)
            if context is not None:
                context.term()
        finally:
            self._client = None

    def close(self) -> None:
        self.shutdown_remote(kill_server=False)

    # --------------------------------------------------------------------- #
    # Internals
    # --------------------------------------------------------------------- #

    def _resolve_indices(self, robot) -> None:
        joint_names = list(getattr(robot, "joint_names", None) or robot.data.joint_names)
        self._arm_joint_ids = [joint_names.index(f"joint{i}") for i in range(1, 7)]
        self._gripper_joint_id = joint_names.index("g2_joint")
        body_names = list(robot.data.body_names)
        self._eef_body_id = body_names.index("g2_base_link")
        self._base_body_id = body_names.index("base_link")

    def _query_chunk(self, env: gym.Env, observation: dict) -> torch.Tensor:
        """Build the GR00T observation, call the server, and translate the returned action
        chunk to an (num_envs, execution_horizon, 8) env-action tensor."""
        assert self.task_description is not None, "task_description not set"
        u = env.unwrapped
        robot = u.scene["robot"]
        if self._arm_joint_ids is None:
            self._resolve_indices(robot)

        num_envs = robot.data.joint_pos.shape[0]
        policy_obs = self._build_gr00t_observation(observation, u.scene, robot, num_envs)
        action_dict, _info = self._client.get_action(policy_obs)

        arm_eef = _decode_ndarray(action_dict["arm_eef"]).astype(np.float64)  # (N, H, 9)
        gripper = _decode_ndarray(action_dict["gripper"]).astype(np.float64)  # (N, H, 1)
        horizon = min(self.config.execution_horizon, arm_eef.shape[1])

        env_action = torch.zeros((num_envs, horizon, 8), dtype=torch.float32, device=self.device)
        for n in range(num_envs):
            for t in range(horizon):
                pos = arm_eef[n, t, :3]
                quat_wxyz = _rot6d_to_quat_wxyz(arm_eef[n, t, 3:9])
                env_action[n, t, 0:3] = torch.as_tensor(pos, dtype=torch.float32)
                env_action[n, t, 3:7] = torch.as_tensor(quat_wxyz, dtype=torch.float32)
                env_action[n, t, 7] = 1.0 if gripper[n, t, 0] > self.config.gripper_open_threshold else -1.0
        return env_action

    def _build_gr00t_observation(self, observation: dict, scene, robot, num_envs: int) -> dict:
        # --- video: front + wrist, resized to the training resolution, (N, 1, H, W, 3) uint8.
        front = self._get_camera_rgb(observation, scene, self.config.front_cam_key, "go2_front_cam")
        wrist = self._get_camera_rgb(observation, scene, self.config.wrist_cam_key, "wrist_cam")

        # --- state: arm_eef (9, base_link frame) + arm joints (6) + gripper opening (1).
        from isaaclab.utils.math import subtract_frame_transforms

        jp = self._to_torch(robot.data.joint_pos)  # (N, num_joints); joint_pos is a warp array
        arm = jp[:, self._arm_joint_ids].detach().cpu().numpy().astype(np.float32)          # (N, 6)
        gripper = jp[:, self._gripper_joint_id : self._gripper_joint_id + 1]
        gripper = gripper.detach().cpu().numpy().astype(np.float32)                          # (N, 1)

        pos_w = self._to_torch(robot.data.body_pos_w)
        quat_w = self._to_torch(robot.data.body_quat_w)
        if pos_w.dim() == 2:  # single-env fallback (num_bodies, 3/4)
            pos_w = pos_w.unsqueeze(0)
            quat_w = quat_w.unsqueeze(0)
        p, q = subtract_frame_transforms(
            pos_w[:, self._base_body_id], quat_w[:, self._base_body_id],
            pos_w[:, self._eef_body_id], quat_w[:, self._eef_body_id],
        )
        arm_eef = self._pos_quat_wxyz_to_xyz_rot6d(
            p.detach().cpu().numpy(), q.detach().cpu().numpy()
        ).astype(np.float32)                                                                 # (N, 9)

        return {
            "video": {
                "front": front.reshape(num_envs, 1, *front.shape[1:]),
                "wrist": wrist.reshape(num_envs, 1, *wrist.shape[1:]),
            },
            "state": {
                "arm_eef": arm_eef.reshape(num_envs, 1, 9),
                "arm": arm.reshape(num_envs, 1, 6),
                "gripper": gripper.reshape(num_envs, 1, 1),
            },
            "language": {
                "annotation.human.task_description": [[self.task_description] for _ in range(num_envs)],
            },
        }

    def _get_camera_rgb(self, observation: dict, scene, obs_key: str, scene_key: str) -> np.ndarray:
        """Fetch an (N, H, W, 3) uint8 RGB array, resized to (image_height, image_width).

        Prefer the rendered ``camera_obs`` group (what the recorder captured); fall back to the
        scene camera sensor if the obs group / key is absent.
        """
        arr = None
        cam_obs = observation.get("camera_obs") if isinstance(observation, dict) else None
        if cam_obs is not None and obs_key in cam_obs:
            arr = cam_obs[obs_key]
        else:
            try:
                arr = scene[scene_key].data.output["rgb"]
            except (KeyError, AttributeError) as e:
                raise KeyError(
                    f"camera '{obs_key}' not in observation['camera_obs'] and scene sensor "
                    f"'{scene_key}' unavailable ({e}); run with --enable_cameras."
                )
        t = arr if isinstance(arr, torch.Tensor) else torch.as_tensor(np.asarray(arr))
        if t.dim() == 3:  # (H, W, C) single env
            t = t.unsqueeze(0)
        t = t[..., :3]
        # Resize to the training resolution with bilinear interpolation (NCHW).
        h, w = self.config.image_height, self.config.image_width
        if t.shape[1] != h or t.shape[2] != w:
            chw = t.permute(0, 3, 1, 2).float()
            chw = torch.nn.functional.interpolate(chw, size=(h, w), mode="bilinear", align_corners=False)
            t = chw.permute(0, 2, 3, 1)
        return t.round().clamp(0, 255).to(torch.uint8).detach().cpu().numpy()

    @staticmethod
    def _to_torch(x):
        if isinstance(x, torch.Tensor):
            return x
        import warp as wp  # noqa: PLC0415

        return wp.to_torch(x)

    @staticmethod
    def _pos_quat_wxyz_to_xyz_rot6d(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
        """(N,3)+(N,4 wxyz) -> (N,9) [xyz, rot6d]; rot6d = R first two rows flattened."""
        from scipy.spatial.transform import Rotation

        pos = np.asarray(pos, np.float64).reshape(-1, 3)
        quat_wxyz = np.asarray(quat_wxyz, np.float64).reshape(-1, 4)
        quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
        mats = Rotation.from_quat(quat_xyzw).as_matrix()
        rot6d = mats[:, :2, :].reshape(len(mats), 6)
        return np.concatenate([pos, rot6d], axis=1)
