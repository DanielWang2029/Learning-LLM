"""Compare DeepStack multi-level visual injection with final-level-only input."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from data.synthesizer import make_batch
from src.model import DeepStackClassifier

def run(deepstack: bool, seed: int):
    torch.manual_seed(seed); rng=np.random.default_rng(seed)
    model=DeepStackClassifier(deepstack); opt=torch.optim.AdamW(model.parameters(),lr=2e-3)
    for _ in range(520):
        x,y=make_batch(48,rng); loss=F.cross_entropy(model(torch.from_numpy(x)),torch.from_numpy(y))
        opt.zero_grad();loss.backward();opt.step()
    x,y=make_batch(500,rng)
    with torch.no_grad(): return float((model(torch.from_numpy(x)).argmax(-1).numpy()==y).mean())

def main():
    deep, last=run(True,31),run(False,31)
    results={"deepstack_accuracy":deep,"last_layer_only_accuracy":last,"chance_accuracy":.5,
             "injection_layers":[{"lm_layer":1,"vit_feature":"low token features"},{"lm_layer":2,"vit_feature":"mid token features"},{"lm_layer":3,"vit_feature":"high pooled feature"}]}
    out=ROOT/"data/demo_results.json";out.write_text(json.dumps(results,indent=2))
    print("Qwen3-VL tiny CPU reproduction")
    print("Fine detail: direction of a two-pixel diagonal (same global color mean)")
    print(f"DeepStack multi-level injection accuracy: {deep:.1%}")
    print(f"Final-layer-only baseline accuracy:       {last:.1%} (chance: 50.0%)")
    print(f"Wrote visualization data: {out}")
if __name__=="__main__":main()
