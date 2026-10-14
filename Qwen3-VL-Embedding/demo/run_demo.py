"""Train a tiny multimodal embedding model and a pairwise reranker."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT))
from data.synthesizer import balanced_batch,all_captions,make_images
from src.model import ImageEncoder,TextEncoder,Reranker

def main():
    torch.manual_seed(41);rng=np.random.default_rng(41)
    # A deliberately compact shared space leaves some ambiguity for reranking.
    image,text=ImageEncoder(3),TextEncoder(3);opt=torch.optim.AdamW(list(image.parameters())+list(text.parameters()),lr=3e-3)
    # Symmetric InfoNCE over a batch containing one example per caption.
    for _ in range(35):
        x,c,s=balanced_batch(rng);im=image(torch.from_numpy(x));tx=text(torch.from_numpy(c),torch.from_numpy(s))
        logits=im@tx.T/.12;target=torch.arange(6)
        loss=(F.cross_entropy(logits,target)+F.cross_entropy(logits.T,target))/2
        opt.zero_grad();loss.backward();opt.step()
    c,s,names=all_captions();caps=text(torch.from_numpy(c),torch.from_numpy(s)).detach()
    rerank=Reranker(5);ropt=torch.optim.AdamW(rerank.parameters(),lr=3e-3)
    # The reranker learns pair compatibility for every image/caption candidate.
    for _ in range(260):
        x,bc,bs=balanced_batch(rng)
        emb=F.one_hot(torch.from_numpy(c),3).float()
        emb=torch.cat((emb,F.one_hot(torch.from_numpy(s),2).float()),-1)
        scores=rerank(torch.from_numpy(x),emb.unsqueeze(0).expand(6,-1,-1))
        targets=torch.eye(6)
        loss=F.binary_cross_entropy_with_logits(scores,targets)
        ropt.zero_grad();loss.backward();ropt.step()
    x=make_images(c,s,rng)
    with torch.no_grad():
        sim=image(torch.from_numpy(x))@caps.T
        pair_text=torch.cat((F.one_hot(torch.from_numpy(c),3).float(),
                             F.one_hot(torch.from_numpy(s),2).float()),-1)
        rer=rerank(torch.from_numpy(x),pair_text.unsqueeze(0).expand(6,-1,-1))
    truth=np.arange(6);base=float((sim.argmax(-1).numpy()==truth).mean());better=float((rer.argmax(-1).numpy()==truth).mean())
    results={"embedding_top1_accuracy":base,"reranked_top1_accuracy":better,"chance_accuracy":1/6,
      "captions":names,"similarity_matrix":sim.numpy().round(3).tolist(),
      "pipeline":["synthetic image → image encoder","caption → text encoder","cosine retrieval → top candidates","joint reranker → reordered result"]}
    out=ROOT/"data/demo_results.json";out.write_text(json.dumps(results,indent=2))
    print("Qwen3-VL-Embedding tiny CPU reproduction")
    print(f"Image→caption embedding top-1: {base:.1%} (chance: 16.7%)")
    print(f"After pairwise reranking top-1: {better:.1%}")
    print(f"Wrote visualization data: {out}")
if __name__=="__main__":main()
