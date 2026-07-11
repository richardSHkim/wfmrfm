# go2_airbot_play

Isaac Lab / Arena extension for the **Unitree Go2 + Airbot Play + wrist D435i** embodiment.

This is the IsaacLab-specific integration layer for the robot. The language-agnostic
robot description (URDF/USD/ROS meshes) lives separately in the
`third_party/go2_airbot_play_description` submodule; this package only holds the code that
depends on Isaac Lab / Arena (actuators, init state, embodiment, environment).

## Layout

```
src/go2_airbot_play/
  assets/articulation_cfg.py   # GO2_AIRBOT_PLAY_CFG (ArticulationCfg): spawn + init + actuators
  ...                          # (embodiment / environment / task — added for Arena registration)
```

`GO2_AIRBOT_PLAY_CFG` binds the converted USD to three actuator groups:
`base_legs` (12 Go2 legs, native UNITREE_GO2_CFG DC-motor gains for RL transfer),
`arm` (6 Airbot Play joints, position PD — provisional gains), and `gripper`.

## Install

Requires an environment with Isaac Sim + Isaac Lab and the Arena submodule installed:

```bash
/isaac-sim/python.sh -m pip install -e .
```

The USD path is resolved automatically relative to the repo; override with the
`GO2_AIRBOT_PLAY_USD` environment variable if the asset is mounted elsewhere.
