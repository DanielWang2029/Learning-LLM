"""From-scratch GRPO (Group Relative Policy Optimization), introduced in DeepSeekMath.

Based on "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open
Language Models" (2024, arXiv:2402.03300), §4.1. The paper PDF lives next to this
package in ``DeepSeekMath/deepseekmath.pdf``.

GRPO is the critic-free RL algorithm behind DeepSeek-R1 and much of the recent
reasoning-RL work. This package implements the algorithm itself — group sampling,
group-relative advantages, and the clipped surrogate with a KL-to-reference
penalty — and trains a tiny Transformer policy on a toy arithmetic task using
reward alone.
"""

from .grpo import group_advantages, grpo_loss, kl_to_reference
from .policy import PolicyConfig, TransformerPolicy
from .task import (VOCAB_SIZE, EOS, all_prompts, decode_answer, encode_prompt,
                   max_answer_len, reward)

__all__ = [
    "group_advantages", "grpo_loss", "kl_to_reference",
    "PolicyConfig", "TransformerPolicy",
    "VOCAB_SIZE", "EOS", "all_prompts", "decode_answer", "encode_prompt",
    "max_answer_len", "reward",
]
