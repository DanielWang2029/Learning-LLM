"""Write the toy task definition (prompts + reward scheme) to data/.

The RL demo generates its own rollouts; this just serializes the task so you can
read it. Run with:  python data/generate_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import (R_MAX, TAG_LEGEND, all_prompts, prompt_str, reward)

DATA_DIR = Path(__file__).resolve().parent


def main() -> None:
    prompts = all_prompts(4)
    # Show the reward for a few example responses to the same prompt.
    a, b = 3, 4
    demo_responses = {
        "{3+4}[7]": reward(a, b, "{3+4}[7]")["reward"],
        "{}[7]": reward(a, b, "{}[7]")["reward"],
        "{3+4}[5]": reward(a, b, "{3+4}[5]")["reward"],
        "7": reward(a, b, "7")["reward"],
    }
    payload = {
        "description": "RL task: given a+b=, emit {think}[answer]. Reward = "
                       "format + showing work + correct answer.",
        "tag_legend": TAG_LEGEND,
        "reward_max": R_MAX,
        "prompts": [prompt_str(a, b) for a, b in prompts],
        "example_rewards_for_3+4": demo_responses,
    }
    out = DATA_DIR / "task.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote task with {len(prompts)} prompts -> {out}")


if __name__ == "__main__":
    main()
