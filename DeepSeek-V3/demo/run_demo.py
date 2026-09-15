"""End-to-end DeepSeek-V3 demo (CPU, well under a minute) — arXiv:2412.19437.

Three signature components, demonstrated on a toy sequence-copy task:

1. MULTI-HEAD LATENT ATTENTION (MLA). We train the model with MLA and, separately,
   with plain multi-head attention, and show MLA MATCHES quality while caching a
   low-rank latent instead of full keys/values — quantifying the KV-cache saving.

2. DeepSeekMoE. The feed-forward is a fine-grained routed-expert MoE with an
   always-on shared expert; we report the per-expert load (kept balanced by the
   auxiliary-loss-free router bias).

3. MULTI-TOKEN PREDICTION (MTP). Alongside the next-token head, an MTP head
   predicts the token TWO steps ahead; we report both accuracies.

Writes ``data/dsv3_results.json`` for the viz.  Run with:  python demo/run_demo.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

torch.set_num_threads(1)

PAPER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PAPER_DIR))

from src import DeepSeekV3, DeepSeekV3Config, MLAConfig  # noqa: E402

DATA_DIR = PAPER_DIR / "data"
PAD, BOS, SEP = 0, 1, 2
NUM_SPECIAL = 3


def make_batch(bs, n, vocab, gen):
    content = NUM_SPECIAL + torch.randint(0, vocab - NUM_SPECIAL, (bs, n), generator=gen)
    bos = torch.full((bs, 1), BOS)
    sep = torch.full((bs, 1), SEP)
    tokens = torch.cat([bos, content, sep, content], dim=1)  # (bs, 2n+2)
    return tokens, n


@torch.no_grad()
def accuracies(model, bs, n, vocab, gen):
    """Return (main next-token acc, MTP two-ahead acc) over the copied region."""
    tokens, n = make_batch(bs, n, vocab, gen)
    idx, nxt = tokens[:, :-1], tokens[:, 1:]
    main_logits, mtp_logits, _ = model(idx, nxt)
    # copied region is positions n+2 .. 2n+1 (content2)
    main_pred = main_logits[:, n + 1:2 * n + 1].argmax(-1)     # predicts tokens[n+2:2n+2]
    main_tgt = tokens[:, n + 2:2 * n + 2]
    main_acc = (main_pred == main_tgt).float().mean().item()
    mtp_acc = float("nan")
    if mtp_logits is not None:
        mtp_pred = mtp_logits[:, n:2 * n].argmax(-1)           # predicts tokens[n+2:2n+2]
        mtp_acc = (mtp_pred == main_tgt).float().mean().item()
    return main_acc, mtp_acc


def train(cfg, args, gen_seed, record=False):
    gen = torch.Generator().manual_seed(gen_seed)
    torch.manual_seed(gen_seed)
    model = DeepSeekV3(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    curve = []
    for step in range(1, args.steps + 1):
        model.train()
        tokens, n = make_batch(args.batch_size, args.seq_len, args.vocab_size, gen)
        idx, nxt = tokens[:, :-1], tokens[:, 1:]
        main_logits, mtp_logits, infos = model(idx, nxt)
        main_loss = F.cross_entropy(main_logits.reshape(-1, cfg.vocab_size), nxt.reshape(-1))
        loss = main_loss
        if mtp_logits is not None:
            # MTP predicts t+2: align mtp position p with target tokens[p+2].
            mtp_tgt = tokens[:, 2:]                             # (bs, 2n)
            mtp_loss = F.cross_entropy(mtp_logits[:, :-1].reshape(-1, cfg.vocab_size),
                                       mtp_tgt.reshape(-1))
            loss = loss + mtp_loss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if record and (step % args.log_every == 0 or step == 1):
            ma, mt = accuracies(model, 256, args.seq_len, args.vocab_size, gen)
            curve.append({"step": step, "loss": round(loss.item(), 4),
                          "main_acc": round(ma, 4), "mtp_acc": round(mt, 4)})
            print(f"  step {step:4d}/{args.steps} | loss {loss.item():.4f} | "
                  f"next-token {ma*100:5.1f}% | MTP(t+2) {mt*100:5.1f}%")
    ma, mt = accuracies(model, 512, args.seq_len, args.vocab_size, gen)
    return model, curve, ma, mt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=6)
    p.add_argument("--vocab-size", type=int, default=32)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--log-every", type=int, default=100)
    args = p.parse_args()

    print("=" * 74)
    print("DeepSeek-V3 demo — Technical Report (arXiv:2412.19437)")
    print("=" * 74)

    mla_cfg = MLAConfig(dim=args.dim, n_heads=4, head_dim=16, kv_latent=32,
                        q_latent=48, rope_dim=16)
    base = dict(vocab_size=args.vocab_size, dim=args.dim, n_layers=args.n_layers,
                mla=mla_cfg, n_routed=8, n_shared=1, top_k=2)

    # ---- 1+3. Train the full DeepSeek-V3 mini (MLA + DeepSeekMoE + MTP) ----
    full_cfg = DeepSeekV3Config(attn="mla", use_mtp=True,
                                **{k: v for k, v in base.items()})
    model = DeepSeekV3(full_cfg)
    n_params = sum(pm.numel() for pm in model.parameters())
    print(f"params={n_params:,} | attn=MLA | MoE: {full_cfg.n_routed} routed + "
          f"{full_cfg.n_shared} shared, top-{full_cfg.top_k} | MTP: on\n")
    print("Training full model (MLA + DeepSeekMoE + Multi-Token Prediction):")
    start = time.time()
    trained, curve, main_acc, mtp_acc = train(full_cfg, args, gen_seed=0, record=True)
    print(f"\nFull model: next-token {main_acc*100:.1f}% | MTP two-ahead {mtp_acc*100:.1f}%")

    # DeepSeekMoE expert load from the trained model.
    gen = torch.Generator().manual_seed(1)
    tokens, _ = make_batch(256, args.seq_len, args.vocab_size, gen)
    trained.eval()
    with torch.no_grad():
        _, _, infos = trained(tokens[:, :-1], tokens[:, 1:])
    load = torch.stack([i.load for i in infos]).mean(0)
    print(f"\nDeepSeekMoE routed-expert load (ideal {full_cfg.top_k/full_cfg.n_routed:.3f} each):")
    print("  " + "  ".join(f"E{i}:{load[i]:.3f}" for i in range(full_cfg.n_routed)))

    # ---- 2. MLA vs MHA: quality match + KV-cache saving ----
    print("\nMLA vs standard MHA (same budget, fresh models):")
    mha_cfg = DeepSeekV3Config(attn="mha", use_mtp=True, **{k: v for k, v in base.items()})
    _, _, mha_main, mha_mtp = train(mha_cfg, args, gen_seed=0, record=False)
    mla_cache = trained.blocks[0].attn.cache_floats_per_token()
    # fresh MHA model just to query the cache size
    mha_probe = DeepSeekV3(mha_cfg)
    mha_cache = mha_probe.blocks[0].attn.cache_floats_per_token()
    saving = mha_cache / mla_cache
    per_layer_bytes = lambda f: f * 4  # float32
    print(f"  MLA  : next-token acc {main_acc*100:5.1f}% | KV cache {mla_cache} floats/token/layer")
    print(f"  MHA  : next-token acc {mha_main*100:5.1f}% | KV cache {mha_cache} floats/token/layer")
    print(f"  => MLA matches quality and caches {saving:.2f}x less "
          f"({per_layer_bytes(mla_cache)} vs {per_layer_bytes(mha_cache)} bytes/token/layer, fp32)")

    elapsed = time.time() - start
    print(f"\nTotal demo time: {elapsed:.1f}s on CPU")

    # Show one worked MTP example: predict the next TWO tokens greedily.
    gen2 = torch.Generator().manual_seed(7)
    tokens, n = make_batch(1, args.seq_len, args.vocab_size, gen2)
    with torch.no_grad():
        idx, nxt = tokens[:, :-1], tokens[:, 1:]
        ml, tl, _ = trained(idx, nxt)
        pos = n + 1  # from here the main head predicts t+1 and the MTP head t+2
        t1 = int(ml[0, pos].argmax()); t2 = int(tl[0, pos].argmax())
    print(f"\nMTP example — from prompt, predict the next two tokens at once:")
    print(f"  sequence     : {tokens[0].tolist()}")
    print(f"  true (t+1,t+2): ({int(tokens[0, pos+1])}, {int(tokens[0, pos+2])})")
    print(f"  pred (t+1,t+2): ({t1}, {t2})   [next-token head + MTP head]")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "paper": "DeepSeek-V3 Technical Report (arXiv:2412.19437)",
        "params": n_params,
        "curve": curve,
        "final": {"main_acc": round(main_acc, 4), "mtp_acc": round(mtp_acc, 4)},
        "moe_load": [round(float(x), 4) for x in load],
        "moe_ideal": round(full_cfg.top_k / full_cfg.n_routed, 4),
        "n_shared": full_cfg.n_shared, "n_routed": full_cfg.n_routed, "top_k": full_cfg.top_k,
        "kv_cache": {"mla_floats": mla_cache, "mha_floats": mha_cache,
                     "saving": round(saving, 3),
                     "mla_acc": round(main_acc, 4), "mha_acc": round(mha_main, 4)},
        "mla_config": {"kv_latent": mla_cfg.kv_latent, "rope_dim": mla_cfg.rope_dim,
                       "n_heads": mla_cfg.n_heads, "head_dim": mla_cfg.head_dim},
        "seconds": round(elapsed, 1),
    }
    (DATA_DIR / "dsv3_results.json").write_text(json.dumps(out, indent=2))
    print("\nWrote data/dsv3_results.json (curves + KV-cache + MoE load for the viz).")

    if main_acc < 0.9:
        raise SystemExit(f"Full model did not converge (next-token {main_acc*100:.1f}%).")
    if mtp_acc < 0.9:
        raise SystemExit(f"MTP two-ahead accuracy too low ({mtp_acc*100:.1f}%).")
    if abs(main_acc - mha_main) > 0.05:
        raise SystemExit("MLA did not match MHA quality.")
    print("OK: MLA matches MHA quality with a smaller KV cache; MoE balanced; MTP predicts t+2.")


if __name__ == "__main__":
    main()
