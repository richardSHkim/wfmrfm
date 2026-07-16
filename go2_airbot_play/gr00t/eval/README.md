# GR00T closed-loop evaluation — Go2 + Airbot Play (Phase-1)

Deploy a fine-tuned **GR00T N1.7** checkpoint (or, before one exists, a **dataset replay**) as
the high-level policy in **IsaacLab-Arena** and measure pick-place success. This is the
`DEPLOY` step of `../README.md`, built **out-of-tree** (no submodule edits).

```
                       ZMQ (tcp:5555)
  serve_go2airbot_gr00t.py  ───────────────▶  go2_airbot_play.eval.gr00t_eef_policy
  (GR00T server:                             (Arena closed-loop policy client)
   ReplayPolicy or Gr00tPolicy)                     │  arm_eef(9)+gripper(1)
        ▲                                            ▼  → [xyz, quat, grip] (8)
        │ obs {video, state, language}      Go2AirbotPlayMapleTablePickPlaceEvalEnvironment
        └────────────────────────────────── (IK-EEF arm + binary gripper, base fixed)
                                                     │
                                            policy_runner.py → success metric + HTML report
```

## Pieces

| File | Role |
|---|---|
| `../../src/go2_airbot_play/environment.py` → `Go2AirbotPlayMapleTablePickPlaceEvalEnvironment` | Eval env: swaps the arm to differential-IK **EEF-pose** control (`_EefIkActionCfg`) — the action space GR00T outputs — keeps the base welded, renders front+wrist at 180×320. |
| `../../src/go2_airbot_play/eval/gr00t_eef_policy.py` → `Go2AirbotEefClosedloopPolicy` | Custom Arena `@register_policy`: talks to the GR00T server via GR00T's `PolicyClient`, builds the `{video, state, language}` obs straight from the sim scene, and translates `action.arm_eef`(9D abs, base_link) + `action.gripper` → the env's 8-vector `[x,y,z,qw,qx,qy,qz,grip]`. Bypasses Arena's G1-specific joint bridge. |
| `serve_go2airbot_gr00t.py` | GR00T server. `--dataset-path` → `ReplayPolicy` (no model); `--model-path` → `Gr00tPolicy` (real checkpoint). Registers the `NEW_EMBODIMENT` modality config first. |
| `make_stats_json.py` | Writes `meta/stats.json` (required by the dataset loader for the replay path). |
| `run_eval.sh` | One-shot: serve → wait → `policy_runner` → stop. Single job + HTML report. |
| `eval_jobs_config.json` | Multi-job config for `eval_runner.py` (aggregated success + report). |

## A. Rehearsal with `ReplayPolicy` (do this BEFORE the checkpoint arrives)

Validates every moving part — obs extraction, action translation, IK tracking, env stepping,
metric/report generation — with **no trained model**, by having the server replay the collected
dataset's recorded actions. Run **inside the Arena container**, from the repo root:

```bash
# 0. one-time: the replay path's dataset loader requires meta/stats.json
/isaac-sim/python.sh go2_airbot_play/gr00t/eval/make_stats_json.py \
    --dataset-path scratchpad_out/datasets/go2airbot_pickplace/lerobot

# 1. serve + run + report
bash go2_airbot_play/gr00t/eval/run_eval.sh replay \
    scratchpad_out/datasets/go2airbot_pickplace/lerobot 10
```

Outputs land under `scratchpad_out/eval/go2airbot_gr00t/<timestamped>/` (per-episode results,
camera mp4s, HTML report).

**What "success" means here:** `ReplayPolicy` replays episode 0's arm trajectory. Unless the eval
scene spawns the cube/bowl at that episode's object positions, the replayed motion won't
actually grasp them — so **the success rate is not the signal**. The signal is that the pipeline
runs cleanly and the **wrist-camera video reproduces the demonstrated reach→grasp→place motion**
(the IK is tracking the replayed EEF poses). That confirms the plumbing is correct.

## B. Real checkpoint (when training finishes)

```bash
# scp the checkpoint from the training server, then:
bash go2_airbot_play/gr00t/eval/run_eval.sh model /path/to/checkpoint-XXXX 20
```

Here the object-randomized scene + a competent policy should yield real successes. For an
aggregated multi-object sweep with a combined report, start the server manually and use
`eval_runner.py --eval_jobs_config go2_airbot_play/gr00t/eval/eval_jobs_config.json`.

## Contract notes / gotchas

1. **EEF frame + representation.** `action.arm_eef` is absolute `[xyz, rot6d]` in the **base_link**
   frame; the IK term's pose command is in the robot base frame, so position feeds through
   directly and `rot6d`→quaternion(wxyz) feeds orientation. Matches the collection script.
2. **Cameras must be on.** Pass `--enable_cameras` (run_eval.sh does). The policy reads
   `observation['camera_obs']['{go2_front_cam,wrist_cam}_rgb']`, falling back to the scene sensor.
3. **Scene must match training.** `--table_z 0.40 --robot_x -0.15 --object rubiks_cube_hot3d_robolab
   --destination bowl_ycb_robolab` reproduce `scripts/collect_pickplace_dataset.py`.
4. **`execution_horizon` alignment.** The client executes `execution_horizon` chunk steps before
   re-querying; for a faithful replay it must equal the server's `--execution-horizon` (both 16).
5. **Static base (Phase-1).** No locomotion command is consumed. When mobile-base WBC data +
   checkpoint exist, extend the eval env with a locomotion ActionTerm (loading
   `eval/locomotion/.../exported/policy.onnx`) and switch to `--no-fix_base` + the full-WBC config.
6. **Training-server caveat (not an eval issue):** GR00T `ab88b50`'s `stats.py` raises
   `NotImplementedError` for a RELATIVE **EEF** action while building `relative_stats.json`.
   `make_stats_json.py` here only calls `generate_stats` (absolute, per-column), which is all the
   replay path needs — but the training pipeline's `stats.py` step must handle the EEF-relative
   case separately.
```
