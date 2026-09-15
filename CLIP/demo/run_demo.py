"""End-to-end CLIP demo: contrastive training, then zero-shot classification.

What it does (all on CPU, well under a minute):

  1. Generates a tiny synthetic dataset of colored shapes with captions.
  2. Trains the dual encoder with the symmetric InfoNCE loss (paper Fig. 3).
  3. Evaluates ZERO-SHOT: held-out images are classified by picking the caption
     ("a photo of a {color} {shape}") whose text embedding is most similar —
     the images' true labels are never used as training targets.
  4. Prints an image<->text similarity matrix (a bright diagonal = aligned pairs)
     and writes data/clip_demo.json for the visualization.

Run with:  python demo/run_demo.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(1)  # shared CPU box: avoid thread oversubscription

PAPER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PAPER_DIR))

from data.shapes import (  # noqa: E402
    all_classes,
    caption_for,
    make_dataset,
)
from src import CLIP, clip_contrastive_loss  # noqa: E402
from src.tokenizer import encode  # noqa: E402

DATA_DIR = PAPER_DIR / "data"
SEED = 0
CHANCE = 1.0 / len(all_classes())


def class_caption_tokens() -> torch.Tensor:
    """Tokenized 'a photo of a {color} {shape}' prompt for every class."""
    prompts = [encode(caption_for(c, s)) for c, s in all_classes()]
    return torch.tensor(prompts, dtype=torch.long)


def tokens_for_labels(labels: torch.Tensor, class_tokens: torch.Tensor) -> torch.Tensor:
    """Map each image's class label to its caption token ids."""
    return class_tokens[labels]


@torch.no_grad()
def zero_shot_accuracy(model, images, labels, class_tokens):
    """Classify each image by the nearest class caption; return accuracy + preds."""
    model.eval()
    logits, _, _ = model(images, class_tokens)  # (N_images, N_classes)
    preds = logits.argmax(dim=1)
    acc = (preds == labels).float().mean().item()
    return acc, preds


def main() -> None:
    torch.manual_seed(SEED)

    embed_dim = 64
    batch_size = 48
    steps = 700
    lr = 3e-4

    classes = all_classes()
    class_tokens = class_caption_tokens()

    # Separate train / test image sets (same 12 classes, different pixels).
    train_imgs, train_labels = make_dataset(n_per_class=40, seed=SEED)
    test_imgs, test_labels = make_dataset(n_per_class=15, seed=SEED + 999)
    train_imgs_t = torch.from_numpy(train_imgs)
    train_labels_t = torch.from_numpy(train_labels)
    test_imgs_t = torch.from_numpy(test_imgs)
    test_labels_t = torch.from_numpy(test_labels)

    model = CLIP(embed_dim=embed_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"CLIP demo | params={num_params:,} | {len(classes)} classes "
          f"| chance={CHANCE * 100:.1f}%")
    print(f"train images={len(train_labels)}  test images={len(test_labels)}\n")

    curve = []
    rng = torch.Generator().manual_seed(SEED)
    start = time.time()
    for step in range(1, steps + 1):
        model.train()
        idx = torch.randint(0, len(train_labels_t), (batch_size,), generator=rng)
        imgs = train_imgs_t[idx]
        toks = tokens_for_labels(train_labels_t[idx], class_tokens)

        logits, _, _ = model(imgs, toks)
        loss = clip_contrastive_loss(logits)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == 1:
            acc, _ = zero_shot_accuracy(model, test_imgs_t, test_labels_t, class_tokens)
            curve.append({"step": step, "loss": round(loss.item(), 4),
                          "zero_shot_acc": round(acc, 4)})
            print(f"step {step:3d}/{steps} | loss {loss.item():.4f} "
                  f"| zero-shot acc {acc * 100:5.1f}%")

    elapsed = time.time() - start
    final_acc, preds = zero_shot_accuracy(
        model, test_imgs_t, test_labels_t, class_tokens
    )
    print(f"\nTrained {steps} steps in {elapsed:.1f}s")
    print(f"Final zero-shot accuracy: {final_acc * 100:.1f}%  "
          f"(chance {CHANCE * 100:.1f}%)")

    # ---- One prototype image per class -> 12x12 image/text similarity matrix.
    proto_imgs, proto_labels = make_dataset(n_per_class=1, seed=SEED + 7)
    order = proto_labels.argsort()  # sort so row/col k == class k
    proto_imgs_t = torch.from_numpy(proto_imgs)[order]
    with torch.no_grad():
        img_e = model.encode_image(proto_imgs_t)
        txt_e = model.encode_text(class_tokens)
        sim = (img_e @ txt_e.t())  # cosine similarity, [-1, 1]
    diag = sim.diag()
    off = sim - torch.diag(diag)
    print("\nImage<->text cosine similarity (rows=images, cols=captions):")
    print("mean diagonal (matched)   =", round(diag.mean().item(), 3))
    print("mean off-diagonal (mismatched) =",
          round(off.sum().item() / (off.numel() - len(diag)), 3))

    # ---- A few concrete zero-shot predictions.
    print("\nExample zero-shot predictions (held-out images):")
    for i in range(6):
        true = classes[test_labels_t[i].item()]
        pred = classes[preds[i].item()]
        ok = "OK " if true == pred else "XX "
        print(f"  {ok} true={true[0]:>6} {true[1]:<8} -> pred={pred[0]} {pred[1]}")

    # ---- Persist everything the visualization needs.
    payload = {
        "classes": [{"color": c, "shape": s, "caption": caption_for(c, s)}
                    for c, s in classes],
        "similarity_matrix": [[round(v, 4) for v in row] for row in sim.tolist()],
        "diag_mean": round(diag.mean().item(), 4),
        "offdiag_mean": round(off.sum().item() / (off.numel() - len(diag)), 4),
        "final_zero_shot_acc": round(final_acc, 4),
        "chance": round(CHANCE, 4),
        "training_curve": curve,
        "logit_scale": round(model.logit_scale.exp().item(), 3),
        "embed_dim": embed_dim,
        "num_params": num_params,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "clip_demo.json").write_text(json.dumps(payload, indent=2))
    print("\nWrote data/clip_demo.json")

    if final_acc < 0.5:
        raise SystemExit(f"Zero-shot accuracy too low ({final_acc*100:.1f}%).")
    print("OK: zero-shot classification works far above chance.")


if __name__ == "__main__":
    main()
