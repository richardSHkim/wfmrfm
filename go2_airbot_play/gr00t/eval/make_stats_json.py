# Copyright (c) 2026, WFMRFM project.
# SPDX-License-Identifier: Apache-2.0

"""Generate ``meta/stats.json`` for a Go2+Airbot LeRobot dataset (for the ReplayPolicy path).

``LeRobotEpisodeLoader`` hard-asserts ``meta/stats.json`` exists, but our converter does not
ship it (normalization stats are otherwise produced on the training server). This calls GR00T's
own ``generate_stats`` (plain per-column mean/std/min/max/q01/q99 over the float features) — no
``tyro`` needed and, unlike the full ``stats.py`` ``main()``, it does NOT invoke
``generate_rel_stats`` (which raises ``NotImplementedError`` for a RELATIVE **EEF** action at
GR00T ab88b50). ReplayPolicy replays raw actions and never needs the relative stats, so this is
sufficient to run the closed-loop rehearsal.

    /isaac-sim/python.sh go2_airbot_play/gr00t/eval/make_stats_json.py \
        --dataset-path scratchpad_out/datasets/go2airbot_pickplace/lerobot
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-path", required=True, type=Path)
    a = p.parse_args()

    from gr00t.data.stats import generate_stats

    generate_stats(a.dataset_path)
    stats_path = a.dataset_path / "meta" / "stats.json"
    print(f"[stats] wrote {stats_path}" if stats_path.exists() else f"[stats] FAILED to write {stats_path}")


if __name__ == "__main__":
    main()
