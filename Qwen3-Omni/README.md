# Qwen3-Omni
**Qwen Team · 2025 · [arXiv:2509.17765](https://arxiv.org/abs/2509.17765)**

Qwen3-Omni replaces the earlier Whisper front end with a scratch-trained Audio
Transformer (AuT), uses block-wise window attention for streaming prefill, and has a
Thinker–Talker MoE design with multi-codebook, multi-token prediction (MTP) speech
generation. This is a compact CPU mechanism demo—not the 30B model, learned codecs,
or a speech system.

The demo makes a tone signal with NumPy, computes STFT/log-mel features without
audio libraries, compares local windowed encoding to full locality-biased attention,
reports sparse MoE routes, and emits all three codebook tokens for one Talker step.

## Layout
```text
src/omni.py               # window attention, top-1 MoE, MTP codebook heads
data/generate.py          # NumPy audio, STFT, and log-mel frontend
demo/run_demo.py          # streaming/full comparison and Talker output
visualization/index.html  # offline architecture and real trace view
```
## Setup and run
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python demo/run_demo.py
```
It prints windowed-vs-full cosine similarity, active experts, and three predicted
codebook IDs and writes `data/omni_results.json`. The self-contained visualization
works through `file://` with baked output and optionally loads fresh JSON over HTTP.

## Code ↔ paper map
| Paper section | Mechanism | Code |
|---|---|---|
| §1–2 | AuT block-wise window attention | `src/omni.py:window_attention` |
| §2 | Thinker/Talker sparse MoE analogue | `src/omni.py:TinyMoE` |
| §2 | multi-codebook MTP Talker | `AudioTransformerToy.predict_codebooks` |
| §2 | low-rate audio frontend demonstration | `data/generate.py` |
