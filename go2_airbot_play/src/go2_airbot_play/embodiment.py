# Copyright (c) 2026, WFMRFM project.
# SPDX-License-Identifier: Apache-2.0

"""Arena embodiment for the Go2 + Airbot Play + wrist D435i robot.

Minimal embodiment for loading / zero-action smoke tests: it wires
:data:`GO2_AIRBOT_PLAY_CFG` into an Arena scene with joint-position arm control and a
binary gripper. Legs carry no action term (held at the standing default by their PD),
so this is a static stand, not a locomotion controller — that is added later.
"""

from dataclasses import MISSING

import isaaclab.envs.mdp as mdp_isaac_lab
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg, JointPositionActionCfg
from isaaclab.managers import ActionTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import CameraCfg, TiledCameraCfg
from isaaclab.utils import configclass

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.terms.events import reset_all_articulation_joints
from isaaclab_arena.utils.pose import Pose

from .assets import GO2_AIRBOT_PLAY_CFG

# Actual link prim paths inside the spawned USD. The converter nests the articulation
# links under ``Robot/Geometry/<link>`` (instanceable layout), so cameras must be mounted
# on these real paths — mounting on a non-existent parent silently renders from a bogus
# default pose (all cameras end up identical / below the floor).
_BASE_LINK = "{ENV_REGEX_NS}/Robot/Geometry/base_link"
_WRIST_OPTICAL_FRAME = (
    _BASE_LINK
    + "/dock/converter/base_link2/link1/link2/link3/link4/link5/link6"
    + "/eef_connect_base_link/g2_base_link/camera_mount_link/wrist_camera_bottom_screw_frame"
    + "/wrist_camera_link/wrist_camera_color_frame/wrist_camera_color_optical_frame"
)
# Wrist D435i colour-optical frame offset is baked into the USD, so the Arena camera is
# spawned directly on that frame with an IDENTITY offset — it then exactly matches the
# D435i RGB sensor pose (floor toward the image bottom; any tilt is the real arm pose).
_WRIST_CAM_PRIM = _WRIST_OPTICAL_FRAME + "/wrist_cam"
# Go2's built-in front RealSense frame.
_INTEL_CAMERA_FRAME = _BASE_LINK + "/intel_camera"


@register_asset
class Go2AirbotPlayEmbodiment(EmbodimentBase):
    """Unitree Go2 quadruped + Airbot Play 6-DoF arm + wrist D435i."""

    name = "go2_airbot_play"
    tags = ["embodiment", "default"]
    default_arm_mode = ArmMode.SINGLE_ARM

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
        is_tiled_camera: bool = False,
    ):
        super().__init__(enable_cameras, initial_pose, concatenate_observation_terms, arm_mode)
        self.scene_config = Go2AirbotPlaySceneCfg()
        self.scene_config.robot = GO2_AIRBOT_PLAY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.action_config = Go2AirbotPlayActionCfg()
        self.observation_config = Go2AirbotPlayObservationsCfg()
        self.observation_config.policy.concatenate_terms = self.concatenate_observation_terms
        self.event_config = Go2AirbotPlayEventCfg()
        self.camera_config = Go2AirbotPlayCameraCfg()
        self.camera_config._is_tiled_camera = is_tiled_camera

    def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
        return "g2_base_link"


@configclass
class Go2AirbotPlaySceneCfg:
    """Scene additions from the embodiment (robot is assigned in the constructor)."""

    robot: ArticulationCfg | None = None

    # Flat ground the quadruped stands on. Background rooms (e.g. galileo_locomanip) sit
    # at their own z and do NOT provide a floor at the robot's spawn height, so without
    # this the robot free-falls. Kept at z=0 to match the robot's 0.4 m standing height.
    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )


@configclass
class Go2AirbotPlayActionCfg:
    """Joint-position arm control + binary parallel gripper."""

    arm_action: ActionTermCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint[1-6]"],
        scale=0.5,
        use_default_offset=True,
    )
    # g2_joint drives the opening (0..0.072 m); g2_left/right mimic at x0.5 in the URDF but
    # load as independent DOFs, so command all three consistently here.
    gripper_action: ActionTermCfg = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["g2_joint", "g2_left_joint", "g2_right_joint"],
        open_command_expr={"g2_joint": 0.072, "g2_left_joint": 0.036, "g2_right_joint": 0.036},
        close_command_expr={"g2_joint": 0.0, "g2_left_joint": 0.0, "g2_right_joint": 0.0},
    )


@configclass
class Go2AirbotPlayObservationsCfg:
    """Minimal proprioceptive observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp_isaac_lab.last_action)
        joint_pos = ObsTerm(func=mdp_isaac_lab.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp_isaac_lab.joint_vel_rel)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class Go2AirbotPlayEventCfg:
    """Reset joints to the standing / ready default on episode reset."""

    reset_all = EventTerm(func=reset_all_articulation_joints, mode="reset")


@configclass
class Go2AirbotPlayCameraCfg:
    """Wrist D435i colour camera + a third-person chase camera on the robot base.

    The ``overview_cam`` is a third-person eval view (recorded via ``--record_camera_video``).
    It is a real Camera SENSOR fixed to the env prim, looking at the robot workspace from an
    offset of ~[-1.5,-1.5,1.5] — because the headless viewport recorder ignores ViewerCfg, a
    sensor at that pose is the robust way to capture a third-person vantage of the robot.
    """

    wrist_cam: CameraCfg | TiledCameraCfg = MISSING
    go2_front_cam: CameraCfg | TiledCameraCfg = MISSING
    overview_cam: CameraCfg | TiledCameraCfg = MISSING

    def __post_init__(self):
        is_tiled = getattr(self, "_is_tiled_camera", False)
        CameraClass = TiledCameraCfg if is_tiled else CameraCfg
        # Wrist Intel RealSense D435i RGB, intrinsics matched to the D435i colour sensor:
        # FOV 69.4 deg (H) x 42.5 deg (V), native 16:9 (1920x1080 max; 1280x720 used here to
        # keep render cost reasonable). Pinhole model, so focal_length is derived from the
        # 69.4 deg horizontal FOV at Isaac's 20.955 mm horizontal aperture, and the vertical
        # aperture is set to width:height (square pixels) so the 42.5 deg vertical FOV follows.
        # Identity offset on the colour-optical frame == exact D435i RGB pose.
        # Source: Intel RealSense D435i product specs (RGB 1920x1080@30, FOV 69.4x42.5x77 deg).
        self.wrist_cam = CameraClass(
            prim_path=_WRIST_CAM_PRIM,
            update_period=0.0,
            height=720,
            width=1280,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=15.131,
                horizontal_aperture=20.955,
                vertical_aperture=11.787,
                clipping_range=(0.05, 20.0),
            ),
            offset=CameraClass.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0), convention="ros"),
        )
        # Go2 built-in front camera, intrinsics matched to Unitree Go2's stock front camera:
        # ultra-wide 120 deg FOV, 1280x720 (16:9). The 120 deg is taken as the HORIZONTAL FOV
        # (Unitree quotes a single number; if it is actually the diagonal, hFOV ~= 113 deg --
        # adjust focal_length accordingly). Pinhole focal_length derived from 120 deg hFOV at
        # 20.955 mm aperture; square-pixel vertical aperture gives ~88.5 deg vertical FOV.
        # NOTE: the real lens is wide-angle/fisheye; a pinhole matches the FOV envelope but not
        # the barrel distortion -- use FisheyeCameraCfg + calibration if distortion fidelity is
        # required. Mounted on the intel_camera frame, looking forward and slightly down.
        # Source: Unitree Go2 specs (front camera 1280x720, 120 deg ultra-wide-angle).
        self.go2_front_cam = CameraClass(
            prim_path=_INTEL_CAMERA_FRAME + "/go2_front_cam",
            update_period=0.0,
            height=720,
            width=1280,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=6.049,
                horizontal_aperture=20.955,
                vertical_aperture=11.787,
                clipping_range=(0.05, 40.0),
            ),
            offset=CameraClass.OffsetCfg(
                pos=(0.02, 0.0, 0.0),
                rot=(-0.57670, 0.57670, -0.40916, 0.40916),
                convention="ros",
            ),
        )
        # Third-person eval camera, following the Arena droid pattern (CameraCfg +
        # convention="opengl" + a pre-computed offset). ENV-FIXED (mounted on the env prim,
        # not the robot) so it is not occluded by the robot / does not tumble with the base.
        # Eye ~= workspace + [-1.5,-1.5,1.5], looking down-forward at the robot workspace
        # (quaternion pre-computed for the opengl convention, where the camera looks along -Z).
        self.overview_cam = CameraClass(
            prim_path="{ENV_REGEX_NS}/overview_cam",
            update_period=0.0,
            height=720,
            width=1280,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 40.0)),
            offset=CameraClass.OffsetCfg(
                pos=(-1.10, -1.38, 1.56),
                rot=(0.42471, -0.17592, -0.33985, 0.82047),
                convention="opengl",
            ),
        )
