"""End-to-end conditional rectified-flow training on NumPy-made colored shapes."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
torch.set_num_threads(1)
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "data")]
from flow import ConditionalFlow  # noqa: E402
from generate import PROMPTS, batch, classify  # noqa: E402


def main():
    torch.manual_seed(23)
    model = ConditionalFlow()
    opt = torch.optim.Adam(model.parameters(), lr=0.002)
    for step in range(1400):
        clean_np, cond_np = batch(48, step)
        clean, cond = torch.tensor(clean_np), torch.tensor(cond_np)
        noise, t = torch.randn_like(clean), torch.rand(len(clean)) * 0.98
        x_t = t[:, None, None, None] * clean + (1 - t[:, None, None, None]) * noise
        target_velocity = clean - noise
        loss = F.mse_loss(model(x_t, t, cond), target_velocity)
        opt.zero_grad(); loss.backward(); opt.step()

    conditions = torch.tensor(list(range(4)) * 8)
    generated, trajectory = model.sample(conditions, steps=24)
    generated_np = generated.numpy()
    predicted = [classify(x) for x in generated_np]
    score = float(np.mean(np.asarray(predicted) == conditions.numpy()))
    print("Qwen-Image — text-conditioned rectified-flow image generator")
    print(f"final flow-matching MSE: {loss.item():.4f}; conditional attribute accuracy: {score:.1%}")
    for c in range(4):
        print(f"  prompt '{' '.join(PROMPTS[c])}' → predicted '{' '.join(PROMPTS[predicted[c]])}'")
    payload = {
        "prompts": [" ".join(x) for x in PROMPTS],
        "predicted": [" ".join(PROMPTS[predicted[i]]) for i in range(4)],
        "accuracy": round(score, 3), "loss": round(float(loss.item()), 4),
        "generated": generated_np[:4].round(3).tolist(),
        "trajectory": [frame.numpy().round(3).tolist() for frame in trajectory],
    }
    (ROOT / "data" / "image_results.json").write_text(json.dumps(payload))


if __name__ == "__main__":
    main()
