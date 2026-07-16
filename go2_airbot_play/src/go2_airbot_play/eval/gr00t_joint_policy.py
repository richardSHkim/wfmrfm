# Copyright (c) 2026, WFMRFM project.
# SPDX-License-Identifier: Apache-2.0

"""Custom Arena closed-loop policy for the Go2 + Airbot Play GR00T **joint** (Phase-1) embodiment.

Joint-space sibling of ``gr00t_eef_policy.Go2AirbotEefClosedloopPolicy``. Simpler: the model's
action is absolute arm joints + gripper, so there is NO IK, NO rot6d, and NO EEF proprio/FK.

    observation (from the sim scene)
        video.front / video.wrist   (uint8, resized to the dataset resolution)
        state.arm (6)               absolute arm joint angles (joint1..joint6)
        state.gripper (1)           absolute gripper opening (g2_joint, m)
        language.annotation.human.task_description
              |
              v  GR00T server (joint checkpoint)
        action.arm (H, 6)           absolute arm joint targets
        action.gripper (H, 1)       absolute opening target (m)
              |
              v  translate here
    env action (num_envs, 7) = [joint1..joint6, grip_binary]
        -> JointPositionActionCfg (arm, 6, absolute) + BinaryJointPositionActionCfg (1)

Pair with the eval env's ``--arm_action joint`` (absolute JointPositionActionCfg). Select via a
dotted ``--policy_type go2_airbot_play.eval.gr00t_joint_policy.Go2AirbotJointClosedloopPolicy``.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from dataclasses import dataclass

from gr00t.policy.server_client import PolicyClient as Gr00tPolicyClient

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg

# Reuse the cross-version ndarray decode shim (server ab88b50 msgpack-numpy vs client e29d8fc).
from go2_airbot_play.eval.gr00t_eef_policy import _decode_ndarray


@dataclass
class Go2AirbotJointClosedloopPolicyCfg(PolicyCfg):
    """Config for the Go2 + Airbot Play GR00T joint-space closed-loop policy.

    Fields become auto-generated ``--<field>`` CLI flags (avoid collisions with the shared
    runner flags: ``device``, ``num_envs``, ``language_instruction``, ...).
    """

    remote_host: str = "localhost"
    remote_port: int = 5555
    remote_api_token: str | None = None
    remote_kill_on_exit: bool = False
    """Also materializes the ``--remote_kill_on_exit`` flag that policy_runner reads at shutdown."""

    action_horizon: int = 16
    execution_horizon: int = 16
    gripper_open_threshold: float = 0.036
    policy_device: str = "cuda"
    front_cam_key: str = "go2_front_cam_rgb"
    wrist_cam_key: str = "wrist_cam_rgb"
    image_height: int = 180
    image_width: int = 320


@register_policy
class Go2AirbotJointClosedloopPolicy(PolicyBase[Go2AirbotJointClosedloopPolicyCfg]):
    """GR00T closed-loop policy: absolute arm joints + gripper, static base (Phase-1)."""

    name = "go2airbot_gr00t_joint"

    def __init__(self, config: Go2AirbotJointClosedloopPolicyCfg):
        super().__init__(config)
        self.device = config.policy_device
        self.task_description: str | None = None
        self._chunk: torch.Tensor | None = None
        self._cursor: int = 0
        self._arm_joint_ids: list[int] | None = None
        self._gripper_joint_id: int | None = None

        client = Gr00tPolicyClient(
            host=config.remote_host, port=config.remote_port, api_token=config.remote_api_token, strict=False
        )
        if not client.ping():
            raise ConnectionError(
                f"Cannot reach GR00T policy server at {config.remote_host}:{config.remote_port}."
            )
        self._client: Gr00tPolicyClient | None = client

    # ---------------------- Policy interface -------------------

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
                except Exception as e:  # noqa: BLE001
                    print(f"[go2airbot_gr00t_joint] kill_server failed: {e}")
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

    # ---------------------- Internals -------------------

    def _resolve_indices(self, robot) -> None:
        joint_names = list(getattr(robot, "joint_names", None) or robot.data.joint_names)
        self._arm_joint_ids = [joint_names.index(f"joint{i}") for i in range(1, 7)]
        self._gripper_joint_id = joint_names.index("g2_joint")

    def _query_chunk(self, env: gym.Env, observation: dict) -> torch.Tensor:
        assert self.task_description is not None, "task_description not set"
        u = env.unwrapped
        robot = u.scene["robot"]
        if self._arm_joint_ids is None:
            self._resolve_indices(robot)
        num_envs = robot.data.joint_pos.shape[0]

        policy_obs = self._build_gr00t_observation(observation, u.scene, robot, num_envs)
        action_dict, _info = self._client.get_action(policy_obs)

        arm = _decode_ndarray(action_dict["arm"]).astype(np.float64)          # (N, H, 6)
        gripper = _decode_ndarray(action_dict["gripper"]).astype(np.float64)  # (N, H, 1)
        horizon = min(self.config.execution_horizon, arm.shape[1])

        env_action = torch.zeros((num_envs, horizon, 7), dtype=torch.float32, device=self.device)
        env_action[:, :, 0:6] = torch.as_tensor(arm[:, :horizon, :], dtype=torch.float32, device=self.device)
        grip_open = torch.as_tensor(gripper[:, :horizon, 0] > self.config.gripper_open_threshold, device=self.device)
        env_action[:, :, 6] = torch.where(grip_open, 1.0, -1.0)
        return env_action

    def _build_gr00t_observation(self, observation: dict, scene, robot, num_envs: int) -> dict:
        front = self._get_camera_rgb(observation, scene, self.config.front_cam_key, "go2_front_cam")
        wrist = self._get_camera_rgb(observation, scene, self.config.wrist_cam_key, "wrist_cam")

        jp = self._to_torch(robot.data.joint_pos)  # (N, num_joints); joint_pos is a warp array
        arm = jp[:, self._arm_joint_ids].detach().cpu().numpy().astype(np.float32)               # (N, 6)
        gripper = jp[:, self._gripper_joint_id : self._gripper_joint_id + 1]
        gripper = gripper.detach().cpu().numpy().astype(np.float32)                               # (N, 1)

        return {
            "video": {
                "front": front.reshape(num_envs, 1, *front.shape[1:]),
                "wrist": wrist.reshape(num_envs, 1, *wrist.shape[1:]),
            },
            "state": {
                "arm": arm.reshape(num_envs, 1, 6),
                "gripper": gripper.reshape(num_envs, 1, 1),
            },
            "language": {
                "annotation.human.task_description": [[self.task_description] for _ in range(num_envs)],
            },
        }

    def _get_camera_rgb(self, observation: dict, scene, obs_key: str, scene_key: str) -> np.ndarray:
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
        if t.dim() == 3:
            t = t.unsqueeze(0)
        t = t[..., :3]
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
