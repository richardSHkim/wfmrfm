# HANDOFF — GR00T N1.7 fine-tune on the training server

Self-contained handoff for running the Go2 + Airbot Play GR00T fine-tune on a **different
server**. Everything data-side was prepared and reviewed on the collection server; this doc is
the training server's runbook. For design depth see `README.md` in this directory.

---

## 1. What this is

Fine-tune **Isaac GR00T N1.7** as the high-level policy (RFM) for a **Unitree Go2 quadruped +
Airbot Play 6-DoF arm + gripper**, in a **decoupled whole-body-control (WBC)** architecture:

- GR00T outputs **arm (relative EEF) + gripper**, and (in the full WBC variant) **base_height +
  navigate commands**. It does **not** regress the 12 leg joints.
- The 12 leg joints come from a **separately-trained RL locomotion policy** (already done, not
  needed for this fine-tune). That policy consumes the base_height/navigate commands GR00T emits.
- Custom robot ⇒ everything is registered under GR00T's `NEW_EMBODIMENT` tag.

---

## 2. What you receive

**A. In-repo scaffolding** — `go2_airbot_play/gr00t/` (travels with the git repo):

| File | Purpose |
|---|---|
| `go2_airbot_arm_data_config.py` | **Phase-1** modality config: arm_eef (rel EEF) + gripper. **Use this for the provided dataset.** |
| `go2_airbot_wbc_data_config.py` | Full decoupled-WBC modality config (adds base_height + navigate). For future mobile data. |
| `modality_arm.json` / `modality.json` | Canonical `meta/modality.json` for Phase-1 / full WBC. |
| `convert_pickplace_to_lerobot.py` | Direct HDF5→LeRobot converter used to build the provided dataset (reference). |
| `add_single_arm_eef.py`, `modality_converter.json`, `convert_go2_airbot_config.yaml`, `joints/`, `info.json` | Full-WBC datagen path via Arena's converter (for future mobile data). |
| `finetune_go2_airbot.sh` | Fine-tune launcher (takes the modality config as arg 3). |
| `README.md` | Full contract + rationale. |

**B. Dataset** — transferred **separately** (it lives under `scratchpad_out/`, which is NOT in
git). ~2.6 MB, self-contained GR00T-LeRobot v2:

```
go2airbot_pickplace/lerobot/
  data/chunk-000/episode_0000{00..09}.parquet
  videos/chunk-000/observation.images.{front,wrist}/episode_0000{00..09}.mp4
  meta/{modality.json, info.json, episodes.jsonl, tasks.jsonl}     # NOTE: no stats.json yet
```

`scp -r` this whole `lerobot/` dir to the training server.

---

## 3. The dataset — properties & review result

Built from `scripts/collect_pickplace_dataset.py` (Arena RecorderManager, scripted IK expert).

- **10 episodes, 4525 frames, 50 Hz.** Task: "pick up the cube and place it in the bowl".
- **Phase-1 / static base**: the Go2 base was **fixed** during collection, so there is **no
  base_height/navigate signal**. ⇒ train with the **arm+gripper** config, not the full WBC one
  (a constant base-command column would break min-max normalization).
- Action space (14→) **10 dims here**: `arm_eef` 9D (relative EEF, xyz+rot6d) + `gripper` 1D.
- **EEF is in the `base_link` (arm-mount) frame**, absolute. GR00T's processor converts
  absolute→relative at train time — the dataset stores absolute on purpose; do not "fix" this.
- Two cameras: `front` (Go2 exterior view) + `wrist` (Intel D435i), both 180×320.

**Review (passed, on the collection server):** no NaN/Inf; rot6d rows orthonormal; arm joints in
range; gripper ∈ [0, 0.072] m; base-frame EEF bounded (x[.14,.48] y[-.18,.16] z[.15,.31]);
action leads the next observed pose by ~1.7 cm (correct causal imitation signal); per-step
relative translation mean 1.8 cm / max 7 cm (smooth, well-scaled); all 10 videos decode with
matching frame counts and show the cube→grasp→bowl trajectory.

---

## 4. Runbook (training server, GPU)

GR00T runs in a **uv** environment; the system/Isaac python is not enough (missing `tyro` etc.).

```bash
# 0. Get the code + env
git clone/pull <this repo>          # brings go2_airbot_play/gr00t/
cd <repo>/third_party/Isaac-GR00T   # GR00T N1.7, pinned at commit ab88b50
uv sync --all-extras                # creates .venv (downloads torch/CUDA stack)

# Paths
DS=/path/to/go2airbot_pickplace/lerobot            # the transferred dataset
CFG=<repo>/go2_airbot_play/gr00t/go2_airbot_arm_data_config.py

# 1. Generate normalization stats (REQUIRED — not shipped with the dataset)
# ⚠️ KNOWN BLOCKER — read "6. Known blocker" below before running this. stats.py CRASHES on
#    our relative-EEF arm when it builds relative_stats.json. If you don't need relative stats,
#    generate ONLY the absolute stats.json instead of running the full stats.py main():
#      uv run python -c "from gr00t.data.stats import generate_stats; \
#          generate_stats('$DS')"     # writes meta/stats.json only
# The full command (expects meta/stats.json + meta/relative_stats.json) is:
uv run python gr00t/data/stats.py \
    --dataset-path "$DS" --embodiment-tag NEW_EMBODIMENT --modality-config-path "$CFG"

# 2. Fine-tune  (args: <dataset> <output_dir> <modality_config>)
bash <repo>/go2_airbot_play/gr00t/finetune_go2_airbot.sh "$DS" /path/to/out "$CFG"
# = launch_finetune.py --base-model-path nvidia/GR00T-N1.7-3B --embodiment-tag NEW_EMBODIMENT
#   --modality-config-path $CFG  (adjust --max-steps/--global-batch-size for your GPU)

# 3. Open-loop sanity eval (should track GT on traj 0; MSE/MAE fall across checkpoints)
uv run python gr00t/eval/open_loop_eval.py \
    --dataset-path "$DS" --embodiment-tag NEW_EMBODIMENT \
    --model-path /path/to/out/checkpoint-XXXX \
    --traj-ids 0 --action-horizon 16 --steps 400 --modality-keys arm_eef gripper
```

Interpreting eval: this is 10 episodes, so expect good fit on training trajs and weak held-out
generalization — that's data scarcity, not a bug. Watch that MSE falls steadily across saved
checkpoints (see `Isaac-GR00T/getting_started/finetune_new_embodiment.md`).

---

## 5. Gotchas (read before you run)

1. **Use `go2_airbot_arm_data_config.py` for this dataset** (Phase-1). The full
   `go2_airbot_wbc_data_config.py` expects base_height/navigate columns this dataset does not have.
2. **`delta_indices` (action horizon = 16) must match between stats and training.** If you change
   it, re-run `stats.py` or normalization will error.
3. **GR00T version**: `Isaac-GR00T` is pinned at `ab88b50`. The modality configs `import gr00t...`
   — run stats/finetune from that same checkout's uv env so the API matches.
4. **Store-absolute contract**: EEF is absolute in `base_link` frame; relative is computed by the
   processor. Do not pre-convert to relative.
5. **base_link frame**: EEF must be arm-mount-relative (not world) — the base moves on the real
   robot. Already satisfied in this dataset; keep it when collecting more.

---

## 6. Known blocker: `stats.py` crashes on the relative-EEF arm

**Symptom.** Running `gr00t/data/stats.py` with our `NEW_EMBODIMENT` config fails while building
`relative_stats.json`:

```
NotImplementedError: EEF action is not yet supported, need to handle rotation transformation
                     based on action format
  (submodules/Isaac-GR00T/gr00t/data/stats.py:186)
```

**Cause.** `stats.py`'s `main()` runs `generate_stats()` (absolute per-column stats) **then**
`generate_rel_stats()` (relative-action stats). `generate_rel_stats()` only processes action
keys with `rep == RELATIVE` — and our arm action `arm_eef` is **RELATIVE + EEF**
(`go2_airbot_arm_data_config.py` / `go2_airbot_wbc_data_config.py`). That routes into the EEF
branch (stats.py:168+), which computes the relative chunking and then **unconditionally
`raise`s** at GR00T commit `ab88b50` (the rot6d-rotation serialization is guarded off as
not-yet-implemented). The `gripper` action (ABSOLUTE) is fine — the single relative-EEF arm key
takes the whole `stats.py` down.

**Impact / what to verify.** The dataset loader (`LeRobotEpisodeLoader`) treats `meta/stats.json`
(absolute) as **required** but `meta/relative_stats.json` as **optional** (loaded only if
present). So the key question is **whether fine-tuning actually needs `relative_stats.json`**:

- **If not needed** → generate only the absolute stats and proceed. That is what step 1's
  fallback does (`from gr00t.data.stats import generate_stats; generate_stats(DS)` → writes
  `meta/stats.json` only; no `tyro`, no EEF branch). The eval side uses exactly this
  (`go2_airbot_play/gr00t/eval/make_stats_json.py`).
- **If needed** → either (a) patch the EEF branch locally (on a submodule branch) to serialize
  the already-computed `traj` to 9D `[xyz, rot6d]` and return it — the `relative_chunking` is
  computed; only the trailing `raise` blocks it — or (b) as a last resort switch the arm to a
  non-relative representation (loses N1.7's relative-EEF prior; not recommended).

**Action:** confirm whether the current fine-tune run completed `stats.py` or already worked
around this. If it worked around it, record how here so eval/data docs match.

**Note:** this is a **training-side** issue only. The closed-loop **eval** replay path
(`go2_airbot_play/gr00t/eval/`) is unaffected — it uses `generate_stats` (absolute) alone and
never touches relative stats.

---

## 7. Next data (to unlock full WBC)

The full 14-dim decoupled-WBC action (arm + gripper + base_height + navigate) needs **mobile
teleop data** where the Go2 base moves. When that exists, use the full-WBC datagen path
(`convert_go2_airbot_config.yaml` → Arena converter → `add_single_arm_eef.py` → stats) and
`go2_airbot_wbc_data_config.py`. A Phase-1 checkpoint is a valid warm-start (same arm
representation).
