# Qwen-Image
**Qwen Team · 2025 · [arXiv:2508.02324](https://arxiv.org/abs/2508.02324)**

Qwen-Image uses text features to condition an MMDiT diffusion/flow model for
text-to-image generation, including precise layout and text control. This is an
intentionally tiny CPU flow-matching reproduction: a conditioned MLP learns 16×16
NumPy-generated colored squares and crosses. It is not a reproduction of the
20B MMDiT, VAE, Qwen2.5-VL encoder, or its real-image capabilities.

It trains the rectified-flow objective, samples a reverse trajectory from noise, and
uses a pixel-level attribute checker to report whether generated color and shape match
each prompt.

## Layout
```text
src/flow.py               # conditioned velocity predictor and Euler sampler
data/generate.py          # NumPy shape renderer and attribute checker
demo/run_demo.py          # flow matching train/sample/evaluation loop
visualization/index.html  # offline image/trajectory viewer
```
## Setup and run
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python demo/run_demo.py
```
It prints final flow MSE and conditional attribute accuracy, then writes
`data/image_results.json`. Open `visualization/index.html` directly for baked
results; serving the folder enables loading the new JSON.

## Code ↔ paper map
| Paper section | Mechanism | Code |
|---|---|---|
| §2.4 | text-conditioned diffusion transformer (scaled toy analogue) | `src/flow.py` |
| §4.1 Eq. 1–2 | rectified-flow interpolation and velocity MSE | `demo/run_demo.py` |
| §4.1 | reverse-time numerical sampling | `ConditionalFlow.sample` |
