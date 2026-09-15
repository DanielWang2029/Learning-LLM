"""End-to-end Muon demo for DeepSeek-V4 (CPU, < 60s).

Two identical MLPs (same init, same data) are trained on a teacher-student
regression task — one with the from-scratch **Muon** optimizer, one with Adam.
We report both loss curves to show Muon converges comparably / faster, and we
also trace how the **hybrid Newton–Schulz** iterations drive a matrix's
singular values toward 1 (the heart of Muon's orthogonalization).

Run with:  python demo/run_demo.py
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import Muon, newton_schulz  # noqa: E402
from src.model import MLP  # noqa: E402


def load_data():
    path = ROOT / "data" / "regression.json"
    if not path.exists():
        sys.path.insert(0, str(ROOT / "data"))
        import generate_data  # noqa: E402

        generate_data.main()
    p = json.loads(path.read_text())
    X = torch.tensor(p["X"], dtype=torch.float32)
    Y = torch.tensor(p["Y"], dtype=torch.float32)
    return X, Y, p["d_in"], p["d_out"]


def train(model, optimizer, X, Y, steps, batch=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = X.size(0)
    losses = []
    for _ in range(steps):
        idx = torch.randint(0, n, (batch,), generator=g)
        pred = model(X[idx])
        loss = torch.nn.functional.mse_loss(pred, Y[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            losses.append(torch.nn.functional.mse_loss(model(X), Y).item())
    return losses


def ns_singular_value_trace():
    """Show Newton–Schulz pulling singular values to 1, iteration by iteration."""
    torch.manual_seed(0)
    G = torch.randn(24, 24)
    from src.muon import _NS_SCHEDULE

    x = G / (G.norm() + 1e-7)
    trace = [torch.linalg.svdvals(x).tolist()]
    for a, b, c in _NS_SCHEDULE:
        gram = x @ x.t()
        x = a * x + (b * gram + c * (gram @ gram)) @ x
        trace.append(torch.linalg.svdvals(x).tolist())
    return trace


def main() -> None:
    t0 = time.time()
    X, Y, d_in, d_out = load_data()
    steps = 300

    # Same initial weights for both runs.
    torch.manual_seed(0)
    base = MLP(d_in, 64, d_out)

    model_muon = copy.deepcopy(base)
    model_adam = copy.deepcopy(base)

    opt_muon = Muon(model_muon.parameters(), lr=0.02, momentum=0.95, weight_decay=0.0)
    opt_adam = torch.optim.Adam(model_adam.parameters(), lr=3e-3)

    print(f"Task: fit a teacher MLP | params/matrix stepped by Muon\n")
    loss_muon = train(model_muon, opt_muon, X, Y, steps, seed=0)
    loss_adam = train(model_adam, opt_adam, X, Y, steps, seed=0)

    def at(losses, frac):
        return losses[int(frac * (len(losses) - 1))]

    print(f"{'step':>6} | {'Muon':>10} | {'Adam':>10}")
    for frac in (0.0, 0.1, 0.25, 0.5, 1.0):
        s = int(frac * (steps - 1))
        print(f"{s:6d} | {at(loss_muon,frac):10.5f} | {at(loss_adam,frac):10.5f}")

    final_muon, final_adam = loss_muon[-1], loss_adam[-1]
    speedup = "-"
    # steps for Muon to reach Adam's final loss
    target = final_adam
    reached = next((i for i, l in enumerate(loss_muon) if l <= target), None)
    if reached is not None and reached > 0:
        speedup = f"{steps / (reached + 1):.1f}x fewer steps"

    print("\n=== Results ===")
    print(f"final MSE   Muon : {final_muon:.5f}")
    print(f"final MSE   Adam : {final_adam:.5f}")
    print(f"Muon reaches Adam's final loss in step "
          f"{reached if reached is not None else 'n/a'}  ({speedup})")

    ns_trace = ns_singular_value_trace()
    sv0, sv_final = ns_trace[0], ns_trace[-1]
    print(f"\nNewton–Schulz singular values: "
          f"start range [{min(sv0):.3f}, {max(sv0):.3f}] "
          f"-> final [{min(sv_final):.3f}, {max(sv_final):.3f}]  (target = 1.0)")
    print(f"elapsed: {time.time()-t0:.1f}s")

    out = {
        "steps": steps,
        "loss_muon": [round(x, 6) for x in loss_muon],
        "loss_adam": [round(x, 6) for x in loss_adam],
        "final_muon": final_muon,
        "final_adam": final_adam,
        "muon_reach_step": reached,
        "ns_singular_values": [[round(v, 4) for v in row] for row in ns_trace],
    }
    art = ROOT / "data" / "muon_result.json"
    art.write_text(json.dumps(out))
    print(f"\nWrote {art}")

    assert final_muon <= final_adam * 1.05, "Muon should match or beat Adam here"
    assert max(sv_final) < 1.2 and min(sv_final) > 0.8, "Newton–Schulz did not converge"
    print("OK: Muon converges (>=) as well as Adam; Newton–Schulz orthogonalizes the update.")


if __name__ == "__main__":
    main()
