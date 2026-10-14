"""CPU demo of Qwen2.5-VL-like window attention and visual grounding."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.synthesizer import make_batch, iou
from src.model import WindowViT, attention_pair_cost

def main():
    torch.manual_seed(25); rng = np.random.default_rng(25)
    model = WindowViT(); opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(650):
        x, label, box = make_batch(48, rng)
        classes, pred_box = model(torch.from_numpy(x))
        loss = F.cross_entropy(classes, torch.from_numpy(label)) + 5 * F.l1_loss(pred_box, torch.from_numpy(box))
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    x, label, box = make_batch(400, rng)
    with torch.no_grad():
        classes, pred_box = model(torch.from_numpy(x))
    pred = pred_box.numpy(); truth = box
    hybrid, full = attention_pair_cost()
    results = {
      "classification_accuracy": float((classes.argmax(-1).numpy() == label).mean()),
      "mean_iou": float(iou(pred, truth).mean()),
      "attention_pairs_hybrid": hybrid, "attention_pairs_all_full": full,
      "attention_cost_reduction": 1 - hybrid/full,
      "example": {"predicted_box": pred[0].round(3).tolist(), "target_box": truth[0].tolist(),
                  "object_color_id": int(label[0])}
    }
    out = ROOT/"data/demo_results.json"; out.write_text(json.dumps(results, indent=2))
    print("Qwen2.5-VL tiny CPU reproduction")
    print("Blocks: window, window, full (8×8 native token grid)")
    print(f"Object-color accuracy: {results['classification_accuracy']:.1%}")
    print(f"Grounding mean IoU:    {results['mean_iou']:.3f}")
    print(f"Attention score pairs: {hybrid:,} hybrid vs {full:,} all-full "
          f"({results['attention_cost_reduction']:.1%} fewer)")
    print(f"Wrote visualization data: {out}")
if __name__ == "__main__": main()
