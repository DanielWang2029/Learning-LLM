"""Small dual encoder and pairwise cross-encoder reranker for synthetic retrieval."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class ImageEncoder(nn.Module):
    def __init__(self, dim=12):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(3*8*8,48),nn.GELU(),nn.Linear(48,dim))
    def forward(self,x): return F.normalize(self.net(x.flatten(1)),dim=-1)

class TextEncoder(nn.Module):
    def __init__(self, dim=12):
        super().__init__();self.color=nn.Embedding(3,dim);self.shape=nn.Embedding(2,dim)
    def forward(self,color,shape): return F.normalize(self.color(color)+self.shape(shape),dim=-1)

class Reranker(nn.Module):
    """Joint pair scorer: image pixels and caption embeddings interact in one MLP."""
    def __init__(self, dim=12):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(3*8*8+dim,64),nn.GELU(),nn.Linear(64,1))
    def forward(self,image, text_embedding):
        b,k,d=text_embedding.shape
        pixels=image.flatten(1).unsqueeze(1).expand(-1,k,-1)
        return self.net(torch.cat((pixels,text_embedding),-1)).squeeze(-1)
