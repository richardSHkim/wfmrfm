# Copyright (c) 2026, WFMRFM project.
# SPDX-License-Identifier: Apache-2.0

"""Serve a Go2 + Airbot Play GR00T policy (real checkpoint OR dataset replay) over ZMQ.

This is the server half of the closed-loop eval. It mirrors GR00T's
``gr00t/eval/run_gr00t_server.py`` but (a) registers our ``NEW_EMBODIMENT`` modality config
first (so the ReplayPolicy path can resolve it) and (b) constructs the ``PolicyServer``
directly, sidestepping a bug in ``run_gr00t_server.main`` that dereferences ``model_path``
before its ``None`` check (which crashes the replay path). No submodule edits.

Two modes:

  # Replay the collected dataset's actions (NO trained model) — validates the whole eval
  # pipeline end-to-end before any checkpoint exists:
  /isaac-sim/python.sh go2_airbot_play/gr00t/eval/serve_go2airbot_gr00t.py \
      --dataset-path scratchpad_out/datasets/go2airbot_pickplace/lerobot \
      --execution-horizon 16 --port 5555

  # Serve a fine-tuned checkpoint (when it arrives from the training server):
  /isaac-sim/python.sh go2_airbot_play/gr00t/eval/serve_go2airbot_gr00t.py \
      --model-path /path/to/checkpoint-XXXX --port 5555

The client is the Arena policy ``go2_airbot_play.eval.gr00t_eef_policy`` driven by
``policy_runner.py`` (see ``run_eval.sh``).
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

# Register the Go2+Airbot NEW_EMBODIMENT modality config (Phase-1: arm_eef + gripper) so
# gr00t's MODALITY_CONFIGS[NEW_EMBODIMENT] is populated for the ReplayPolicy path. Loaded by
# file path because go2_airbot_play/gr00t/ is a data dir, not an importable package.
_DATA_CONFIG = Path(__file__).resolve().parent.parent / "go2_airbot_arm_data_config.py"


def _register_modality_config() -> None:
    spec = importlib.util.spec_from_file_location("_go2airbot_arm_data_config", _DATA_CONFIG)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # runs register_modality_config(..., NEW_EMBODIMENT)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", default=None, help="Fine-tuned checkpoint dir. Omit for replay mode.")
    p.add_argument("--dataset-path", default=None, help="LeRobot dataset dir for ReplayPolicy (replay mode).")
    p.add_argument("--embodiment-tag", default="new_embodiment")
    p.add_argument("--device", default="cuda")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--execution-horizon", type=int, default=16, help="Replay steps advanced per query.")
    p.add_argument("--video-backend", default="torchcodec", help="ReplayPolicy video decode backend.")
    p.add_argument("--no-strict", action="store_true", help="Disable strict obs/action validation.")
    a = p.parse_args()

    import gr00t as _gr00t

    print(f"[serve] gr00t package: {_gr00t.__file__}")

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.server_client import PolicyServer

    tag = EmbodimentTag(a.embodiment_tag)
    strict = not a.no_strict

    if a.model_path is not None:
        # Real checkpoint: the model dir carries its own modality/processor config (baked at
        # fine-tune), so no register_modality_config() is needed here.
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        print(f"[serve] Gr00tPolicy checkpoint: {a.model_path}  (embodiment={tag})")
        policy = Gr00tPolicy(embodiment_tag=tag, model_path=a.model_path, device=a.device, strict=strict)
    elif a.dataset_path is not None:
        # Replay: resolve the modality config from our registration.
        _register_modality_config()
        from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
        from gr00t.policy.replay_policy import ReplayPolicy

        modality_configs = MODALITY_CONFIGS[tag.value]
        print(
            f"[serve] ReplayPolicy dataset: {a.dataset_path}  (embodiment={tag}, "
            f"execution_horizon={a.execution_horizon}, backend={a.video_backend})"
        )
        policy = ReplayPolicy(
            dataset_path=a.dataset_path,
            modality_configs=modality_configs,
            execution_horizon=a.execution_horizon,
            video_backend=a.video_backend,
            strict=strict,
        )
    else:
        raise SystemExit("Provide --model-path (real model) or --dataset-path (replay).")

    print(f"[serve] listening on tcp://{a.host}:{a.port}")
    PolicyServer(policy=policy, host=a.host, port=a.port).run()


if __name__ == "__main__":
    main()
