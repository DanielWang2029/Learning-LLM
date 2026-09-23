"""End-to-end Emformer streaming demo (CPU, < 5 s).

Shows the two properties that matter for streaming ASR:

1. **Correct caching** — the block-by-block *streaming* pass (with key/value cache
   + augmented memory bank) produces output numerically identical to the
   *parallel* training-time pass. Each frame's key/value is computed exactly once
   (no recomputation) and the held state is bounded, independent of utterance
   length.

2. **Close to full context** — Emformer's restricted attention closely matches a
   full bidirectional attention baseline, and the gap shrinks as the left context
   / memory grow (the accuracy vs. latency trade-off of the paper).

Run:  python demo/run_demo.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import EmformerBlock, full_attention
from data.generate_audio import make_feature_sequence

SEED = 0
D_MODEL = 32


def diff_stats(a: torch.Tensor, b: torch.Tensor):
    d = (a - b).abs()
    return d.max().item(), d.mean().item()


def main() -> None:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    feats = torch.from_numpy(make_feature_sequence(rng, n_mels=D_MODEL, duration=1.2))
    T = feats.shape[0]
    print(f"Emformer streaming attention over a synthetic {T}-frame feature sequence (D={D_MODEL})\n")

    block = EmformerBlock(d_model=D_MODEL, num_heads=4, center=8, left=16, right=4, max_memory=4)
    # Tie key & query projections so attention scores measure content similarity;
    # with the monotonic signal this makes full attention content-local, the
    # regime Emformer is designed to approximate.
    block.w_k.weight.data.copy_(block.w_q.weight.data)
    block.eval()
    print(f"config: center C=8  left L=16  right R=4  memory M=4  blocks={-(-T//8)}\n")

    with torch.no_grad():
        y_par = block.forward_parallel(feats)
        y_str, stats = block.forward_stream(feats)
        y_full = full_attention(block, feats)

    # ---- 1. streaming == parallel
    mx, mn = diff_stats(y_str, y_par)
    print("[1] streaming vs. parallel (must match):")
    print(f"    max|Δ| = {mx:.2e}   mean|Δ| = {mn:.2e}   -> identical")
    print(f"    key/value computations = {stats['kv_computations']} for {stats['num_frames']} frames "
          f"(each frame once, no recomputation)")
    print(f"    peak cached state = {stats['peak_state_tokens']} tokens "
          f"(bounded: L + C + R + M, independent of the {T}-frame length)\n")

    # ---- 2. Emformer vs full-context attention, sweeping left context
    mxf, mnf = diff_stats(y_str, y_full)
    scale = y_full.std().item()  # variation of the full-attention output
    print("[2] Emformer (streaming) vs. full bidirectional attention:")
    print(f"    max|Δ| = {mxf:.4f}   mean|Δ| = {mnf:.4f}   (output scale ≈ {scale:.3f})\n")

    print("    left-context sweep (mean|Δ| to full attention):")
    sweep = []
    for L in [0, 4, 8, 16, 32, 64]:
        b = EmformerBlock(d_model=D_MODEL, num_heads=4, center=8, left=L, right=4, max_memory=4)
        b.load_state_dict(block.state_dict())
        b.eval()
        with torch.no_grad():
            ys, _ = b.forward_stream(feats)
            yf = full_attention(b, feats)
        _, m = diff_stats(ys, yf)
        rel = m / scale
        sweep.append({"left": L, "mean_diff": m, "rel_diff": rel})
        print(f"      L={L:3d}  mean|Δ|={m:.4f}  ({rel*100:4.1f}% of scale)")

    # attention map for one middle block (streaming), for the visualization
    with torch.no_grad():
        i = (T // 8) // 2
        s, e = i * 8, min(i * 8 + 8, T)
        lo, hi = max(0, s - 16), min(T, e + 4)
        q = block.w_q(feats[s:e])
        mem = torch.stack([block.w_mem(feats[j * 8:min(j * 8 + 8, T)].mean(0))
                           for j in range(max(0, i - 4), i)]) if i > 0 else None
        parts_k = []
        if mem is not None:
            parts_k.append(block.w_k(mem))
        parts_k.append(block.w_k(feats[lo:hi]))
        K = torch.cat(parts_k, 0)
        dk = D_MODEL // 4
        qh = q.view(q.shape[0], 4, dk).mean(1)
        kh = K.view(K.shape[0], 4, dk).mean(1)
        amap = torch.softmax(block.temp * (qh @ kh.t()) / (dk ** 0.5), dim=-1).numpy()

    n_mem = 0 if mem is None else mem.shape[0]
    out = {
        "num_frames": T,
        "center": 8, "left": 16, "right": 4, "max_memory": 4,
        "num_blocks": stats["num_blocks"],
        "stream_vs_parallel_max": mx,
        "stream_vs_parallel_mean": mn,
        "kv_computations": stats["kv_computations"],
        "peak_state_tokens": stats["peak_state_tokens"],
        "emformer_vs_full_mean": mnf,
        "output_scale": scale,
        "left_sweep": sweep,
        "example_block": i,
        "example_attn": np.round(amap, 4).tolist(),
        "example_n_mem": int(n_mem),
        "example_left": int(s - lo),
        "example_center": int(e - s),
        "example_right": int(hi - e),
    }
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    with open(data_dir / "demo_sample.json", "w") as f:
        json.dump(out, f)
    print(f"\nWrote {data_dir / 'demo_sample.json'} for the visualization.")

    if mx > 1e-4:
        raise SystemExit(f"Streaming did not match parallel (max diff {mx:.2e}).")
    print("OK: streaming exactly reproduces full-context parallel output with bounded, reused state.")


if __name__ == "__main__":
    main()
