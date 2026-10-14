"""NumPy color/shape image-caption pairs; no external images or downloads."""
from __future__ import annotations
import numpy as np

COLORS=np.array([[.96,.24,.38],[.2,.78,.98],[.42,.92,.42]],np.float32)
SHAPES=("square","diagonal")

def make_images(colors, shapes, rng):
    n=len(colors); x=rng.normal(.035,.012,(n,3,8,8)).astype(np.float32)
    for i,(c,s) in enumerate(zip(colors,shapes)):
        if s==0: x[i,:,2:6,2:6]=COLORS[c,:,None,None]
        else:
            for p in range(1,7): x[i,:,p,p]=COLORS[c]
    return x.clip(0,1)

def all_captions():
    colors=np.repeat(np.arange(3),2).astype(np.int64)
    shapes=np.tile(np.arange(2),3).astype(np.int64)
    names=[f"{('red','blue','green')[c]} {SHAPES[s]}" for c,s in zip(colors,shapes)]
    return colors,shapes,names

def balanced_batch(rng):
    colors,shapes,_=all_captions()
    return make_images(colors,shapes,rng),colors,shapes
