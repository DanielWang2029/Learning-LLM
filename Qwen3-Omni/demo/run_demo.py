"""Exercise AuT-style streaming windows and a multi-codebook Talker MTP head."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "data")]
from generate import log_mel, synthetic_audio  # noqa: E402
from omni import AudioTransformerToy  # noqa: E402


def main():
    torch.manual_seed(31)
    mel = log_mel(synthetic_audio())
    # Normalize the synthetic frontend output before its small neural encoder.
    mel = (mel - mel.mean(0)) / (mel.std(0) + 1e-5)
    model = AudioTransformerToy(mel_bins=mel.shape[1])
    frames = torch.tensor(mel)
    with torch.no_grad():
        full, routes = model.encode(frames, window=None)
        streamed, _ = model.encode(frames, window=8)
        cosine = torch.nn.functional.cosine_similarity(full, streamed, dim=-1).mean().item()
        logits = model.predict_codebooks(streamed[-1:])
        codebook_tokens = logits.argmax(-1).squeeze(0).tolist()
    active = len(set(routes.tolist()))
    print("Qwen3-Omni — AuT streaming attention + MoE + multi-codebook MTP")
    print(f"log-mel frames: {len(mel)} at 100 Hz; simulated output rate: 12.5 Hz")
    print(f"windowed-vs-full encoder cosine similarity: {cosine:.3f} (window=±8 frames)")
    print(f"MoE active experts: {active}/4; MTP predicts 3 codebooks in one step: {codebook_tokens}")
    result = {"frames": int(len(mel)), "window": 8, "cosine_similarity": round(cosine, 4),
              "active_experts": active, "total_experts": 4, "codebook_tokens": codebook_tokens,
              "mel_preview": mel[:24, :8].round(3).tolist(),
              "routes": routes[:32].tolist()}
    (ROOT / "data" / "omni_results.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
