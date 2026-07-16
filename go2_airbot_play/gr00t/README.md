# GR00T N1.7 — Go2 + Airbot Play (decoupled WBC)

Preparation scaffolding to fine-tune **Isaac GR00T N1.7** as the high-level policy (RFM) for
the Unitree **Go2 quadruped + Airbot Play 6-DoF arm**, in a **decoupled whole-body-control**
architecture. Follows the GR00T new-embodiment guides
(`third_party/Isaac-GR00T/getting_started/finetune_new_embodiment.md`, `data_config.md`).

Everything here is **out-of-tree** (this dir) — it does not modify the `Isaac-GR00T` or
`IsaacLab-Arena` submodules.

## Architecture: what GR00T controls vs what it does not

```
          camera(s) + proprio + language
                        │
                        ▼
      ┌─────────────────────────────────────┐
      │           GR00T N1.7 (VLA)           │   ← we fine-tune this
      └─────────────────────────────────────┘
        │        │              │         │
   arm_eef(9)  gripper(1)  base_height(1) navigate(3=[vx,vy,wz])
        │        │              └────┬────────┘
        ▼        ▼                   ▼
   Airbot arm  gripper      RL locomotion policy (ALREADY TRAINED, frozen)
   (IK / joint targets)     eval/locomotion/.../exported/policy.onnx
                            └─► 12 Go2 leg joint targets @ 50 Hz
```

* GR00T outputs the **arm** (relative EEF), the **gripper**, and two **high-level base
  commands**. It never regresses the 12 leg joints.
* The 12 leg joints come from our **already-trained RL locomotion policy**
  (see the `go2-airbot-locomotion-training` memory / `eval/locomotion/`). That policy's
  observation includes exactly `velocity_commands(3) = [vx, vy, wz]` and
  `base_height_command(1)` — which is precisely `navigate_command` + `base_height_command`
  emitted here. The interface lines up 1:1.

This mirrors the working G1 decoupled-WBC precedent
(`IsaacLab-Arena/isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_config.py`), adapted from
bimanual G1 to our single-arm quadruped and switched to a **relative-EEF arm** (GR00T N1.7's
recommended, cross-embodiment representation — `Isaac-GR00T/README.md` feature #1).

## Action / state contract

VLA action = **14 dims** (arm 9 + gripper 1 + base_height 1 + navigate 3). VLA state = **16
dims** (arm_eef 9 + arm joints 6 + gripper 1). Camera: single ego view (wrist D435i).

| Modality | key | dims | rep / type / format | LeRobot column |
|---|---|---|---|---|
| action | `arm_eef` | 9 | RELATIVE · EEF · XYZ_ROT6D | `action.eef_pose` |
| action | `gripper` | 1 | ABSOLUTE · NON_EEF · DEFAULT | `action` [6:7] |
| action | `base_height_command` | 1 | ABSOLUTE · NON_EEF · DEFAULT | `teleop.base_height_command` |
| action | `navigate_command` | 3 | ABSOLUTE · NON_EEF · DEFAULT | `teleop.navigate_command` |
| state | `arm_eef` | 9 | — | `observation.eef_pose` |
| state | `arm` | 6 | — | `observation.state` [0:6] |
| state | `gripper` | 1 | — | `observation.state` [6:7] |
| video | `ego_view` | — | — | `observation.images.ego_view` |
| language | `annotation.human.task_description` | — | — | `task_index` → `meta/tasks.jsonl` |

* `arm_eef` is stored **absolute, in the `base_link` (arm-mount) frame** as `[x, y, z, rot6d(6)]`
  where `rot6d` = the first two rows of the rotation matrix, flattened (matches GR00T's
  `EndEffectorPose._matrix_to_rot6d`). The processor converts absolute→relative at
  train/inference time (`state_key="arm_eef"`).
  * **Frame is part of the contract**: EEF pose MUST be expressed relative to `base_link`, NOT
    the world. The Go2 base moves (locomotion), so world-frame EEF drifts unbounded as the
    robot walks — bad precision and it couples the arm target to base motion. `base_link` frame
    keeps positions bounded (~±1 m) and decouples the arm from locomotion (which the RL loco
    policy owns). Do NOT pre-store *relative* EEF: GR00T requires **absolute** in the dataset
    and computes relative itself against the current-timestep reference (verified in
    `gr00t/data/state_action/state_action_processor.py:apply_action`); pre-baking relative
    gives no accuracy gain, breaks the `rep=RELATIVE` path, and locks the action horizon.
* `navigate_command = [vx, vy, wz]` and `base_height_command = [h]` are the RL locomotion
  policy's command inputs (ranges: `vx, vy ∈ ±1`, `wz ∈ ±1`, `h ∈ [0.20, 0.36] m`).

## Files

| File | Role |
|---|---|
| `go2_airbot_wbc_data_config.py` | The `NEW_EMBODIMENT` modality config (EEF-primary). Used for **both** `--modality-config-path` at fine-tune AND the Arena closed-loop policy `modality_config_path` at deploy. |
| `modality.json` | Canonical **final** dataset `meta/modality.json` (EEF-primary). Installed by the post-process. |
| `modality_converter.json` | **Converter-input** template (joint-only; no `arm_eef`). Keeps the converter from padding a bogus joint group. |
| `joints/gr00t_go2_airbot_joint_space.yaml` | GR00T (policy) joint order: `arm` (joint1-6) + `gripper` (g2_joint). |
| `joints/go2_airbot_joint_space.yaml` | Isaac Lab (sim) joint order of the recorded HDF5 arrays. |
| `info.json` | LeRobot `info.json` template (features regenerated by the converter). |
| `convert_go2_airbot_config.yaml` | `Gr00tDatasetConfig` for Arena's HDF5→LeRobot converter. |
| `add_single_arm_eef.py` | **Method-A** post-process: adds single-arm EEF (rot6d) columns + installs `modality.json`. |
| `finetune_go2_airbot.sh` | Fine-tune launcher (+ open-loop eval command in comments). |

## End-to-end pipeline

```
1. COLLECT   Go2+Airbot HDF5 (arm+gripper joints, EEF pos/quat, base_height_cmd,
             navigate_cmd, wrist RGB, language). Sim scripted-expert starting point:
             go2_airbot_play/scripts/collect_pickplace_dataset.py (needs joint + eef +
             teleop-command recording wired to the contract — see "Gaps" below).

2. CONVERT   (Arena repo root, in its container)
             python isaaclab_arena_gr00t/lerobot/convert_hdf5_to_lerobot.py \
                 --yaml_file <this dir>/convert_go2_airbot_config.yaml
             → joint-space GR00T-LeRobot dataset (no EEF yet).

3. EEF       (GR00T uv env)
             python add_single_arm_eef.py \
                 --lerobot-dir <data_root>/<hdf5_stem>/lerobot \
                 --hdf5 <data_root>/<hdf5_name>.hdf5
             → adds observation.eef_pose / action.eef_pose (9D), installs modality.json.

4. STATS     (GR00T uv env, from Isaac-GR00T repo root)
             uv run python gr00t/data/stats.py \
                 --dataset-path <lerobot_dir> --embodiment-tag NEW_EMBODIMENT

5. FINETUNE  cd third_party/Isaac-GR00T
             bash <this dir>/finetune_go2_airbot.sh <lerobot_dir>

6. EVAL      open-loop sanity (see finetune_go2_airbot.sh comment):
             plot predicted vs GT on traj 0; MSE/MAE should fall across checkpoints.

7. DEPLOY    Arena Gr00tRemoteClosedloopPolicy with modality_config_path =
             go2_airbot_wbc_data_config.py. Route arm_eef→arm IK, gripper→gripper,
             base_height_command + navigate_command → the RL locomotion ActionTerm.
```

## Design choices (rationale)

* **Relative EEF arm** (not joint / not absolute): GR00T N1.7's headline representation for
  cross-embodiment transfer and the 20K-hour human-video prior (`README.md` feature #1). Our
  joint-space teleop (PiPER→Airbot) still records absolute EEF via FK, so this is available.
  The dataset stores **absolute** poses (GR00T's required contract) and the processor makes
  them relative — we do not pre-store relative (see the `arm_eef` frame note above).
* **EEF in `base_link` frame** (not world): decouples the arm target from Go2 base motion and
  keeps poses numerically bounded on a mobile manipulator.
* **base/navigate = ABSOLUTE**: they are per-step setpoints / velocity commands, not deltas.
* **Method A (post-process) for EEF**: Arena's converter hardcodes bimanual EEF
  (`assert eef_pose.shape == (length, 14)`) and single-cam POV, and lives in a submodule we
  don't edit. Running it joint-space-only (omit `left_/right_eef_*`) makes it skip EEF
  cleanly; we then add single-arm 9D EEF ourselves. Zero submodule changes.
* **Single `ego_view` (wrist)**: the converter emits one POV video. A second exterior view is
  desirable for a mobile manipulator but needs a converter multi-cam extension.

## Gaps / TODO before a real fine-tune

1. **Collection schema** — wire the HDF5 recorder to the contract: obs `robot_joint_pos`
   (7 = arm6+gripper1), `processed_actions` (7), obs+action EEF `eef_pos`/`eef_quat` (wxyz),
   `base_height_cmd` (1), `navigate_cmd` (3), wrist RGB under `camera_obs`. Verify the
   `*_name_sim` keys in `convert_go2_airbot_config.yaml` against what the recorder writes.
   **EEF `eef_pos`/`eef_quat` MUST be in the `base_link` frame** (not world) — record the arm
   tip pose relative to the arm-mount link, per the contract above.
2. **`add_single_arm_eef.py` is untested against real data** (no dataset exists yet) —
   validate column shapes/alignment on the first collected HDF5; the rot6d math is verified
   against GR00T's convention.
3. **Sim-order YAML** assumes a 7-DOF (arm+gripper) recording; widen to the full 21-DOF
   articulation order if the collection records all joints.
4. **Multi-cam** (wrist + front) and **larger action horizon (50)** are optional upgrades
   (both require regenerating stats; multi-cam also needs a converter change).
