"""End-to-end DSA demo for GLM-5 (CPU, < 60s).

Mirrors GLM-5's DSA "warm-up" recipe: freeze a (toy) pretrained attention
layer and train ONLY the lightning indexer so that its top-k selection agrees
with where the dense attention actually puts its mass. We then show that
top-k sparse attention:

  * recovers almost all of the true attention mass (recall -> ~1.0),
  * produces outputs nearly identical to full attention (cosine -> ~1.0),
  * while scoring only k << L keys per query (big compute saving).

Run with:  python demo/run_demo.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(1)

# Make `src` importable regardless of the current working directory.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import (  # noqa: E402
    LightningIndexer,
    attention_mass_recall,
    dsa_attention,
    full_attention,
)


def load_sequence():
    data_path = ROOT / "data" / "sample_sequence.json"
    if not data_path.exists():
        sys.path.insert(0, str(ROOT / "data"))
        import generate_data  # noqa: E402

        generate_data.main()
    payload = json.loads(data_path.read_text())
    x = torch.tensor(payload["embeddings"], dtype=torch.float32).unsqueeze(0)
    return x, payload["groups"]


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.reshape(-1), b.reshape(-1), dim=0
    ).item()


def main() -> None:
    torch.manual_seed(0)
    t0 = time.time()

    x, groups = load_sequence()
    b, L, d = x.shape
    top_k = 8  # attend to only 8 of L keys per query

    # A frozen "pretrained" attention layer: q = k = x, v = x @ Wv.
    Wv = torch.empty(d, d)
    torch.nn.init.orthogonal_(Wv)
    q = k = x
    v = x @ Wv

    full_out, full_attn = full_attention(q, k, v)

    indexer = LightningIndexer(d_model=d, d_index=16, n_heads=2)

    # --- Baseline: an untrained indexer (essentially random selection). ---
    with torch.no_grad():
        _, sel0 = dsa_attention(q, k, v, indexer(x), top_k)
        recall0 = attention_mass_recall(full_attn, sel0).item()

    # --- Train ONLY the indexer to match the dense attention distribution. ---
    opt = torch.optim.Adam(indexer.parameters(), lr=5e-3)
    target = full_attn.detach()  # (b, L, L) rows sum to 1
    print(f"Sequence: L={L}, d={d} | top_k={top_k} "
          f"(scoring {top_k}/{L} = {100*top_k/L:.0f}% of keys)\n")
    for step in range(1, 401):
        opt.zero_grad()
        scores = indexer(x)
        logp = torch.log_softmax(scores, dim=-1)
        # KL(target || indexer): teach the indexer where attention concentrates.
        loss = torch.nn.functional.kl_div(logp, target, reduction="batchmean")
        loss.backward()
        opt.step()
        if step % 80 == 0 or step == 1:
            with torch.no_grad():
                _, sel = dsa_attention(q, k, v, indexer(x), top_k)
                rec = attention_mass_recall(full_attn, sel).item()
            print(f"step {step:3d} | KL {loss.item():.4f} | mass-recall {rec*100:5.1f}%")

    # --- Final evaluation. ---
    with torch.no_grad():
        idx_scores = indexer(x)
        dsa_out, sel = dsa_attention(q, k, v, idx_scores, top_k)
        recall = attention_mass_recall(full_attn, sel).item()
        fidelity = cosine(dsa_out, full_out)

    compute_saved = 1.0 - top_k / L
    elapsed = time.time() - t0

    print("\n=== DSA results ===")
    print(f"attention mass recall : {recall0*100:5.1f}% (untrained) "
          f"-> {recall*100:5.1f}% (trained indexer)")
    print(f"output cosine vs full : {fidelity:.4f}")
    print(f"keys scored per query : {top_k} of {L}  "
          f"(compute saved: {compute_saved*100:.0f}%)")
    print(f"elapsed               : {elapsed:.1f}s")

    # --- Write a small artifact for the visualization (slice of the data). ---
    view = 16  # show a 16x16 corner of the attention for clarity
    full_slice = full_attn[0, :view, :view].tolist()
    sel_first = sel[0, :view].tolist()  # selected key indices per query (row)
    # compute-vs-length curve for the linear-vs-quadratic chart
    lengths = [64, 128, 256, 512, 1024, 2048, 4096]
    full_cost = [n * n for n in lengths]
    dsa_cost = [n * top_k for n in lengths]
    out = {
        "seq_len": L,
        "d_model": d,
        "top_k": top_k,
        "recall_untrained": recall0,
        "recall_trained": recall,
        "output_cosine": fidelity,
        "compute_saved": compute_saved,
        "view": view,
        "full_attention": full_slice,
        "selected_keys": sel_first,
        "groups": groups[:view],
        "cost_curve": {"lengths": lengths, "full": full_cost, "dsa": dsa_cost},
    }
    art = ROOT / "data" / "dsa_result.json"
    art.write_text(json.dumps(out))
    print(f"\nWrote {art}")

    assert recall > 0.9, "indexer failed to concentrate on the right keys"
    assert fidelity > 0.9, "sparse output diverged from full attention"
    print("OK: DSA top-k approximates full attention while scoring far fewer keys.")


if __name__ == "__main__":
    main()
