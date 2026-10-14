"""A compact conditioned rectified-flow denoiser (not a production MMDiT)."""
import torch
from torch import nn


class ConditionalFlow(nn.Module):
    """Predicts v = x_clean - x_noise from noisy image, timestep, and text condition."""
    def __init__(self, image_size: int = 16, conditions: int = 4):
        super().__init__()
        pixels = image_size * image_size * 3
        self.image_size = image_size
        self.condition = nn.Embedding(conditions, 16)
        # At this miniature scale a learned condition prototype makes the
        # conditional clean-image endpoint explicit; the MLP learns its residual flow.
        self.prototypes = nn.Parameter(torch.full((conditions, 3, image_size, image_size), -2.0))
        self.net = nn.Sequential(
            nn.Linear(pixels + 17, 160), nn.SiLU(),
            nn.Linear(160, 160), nn.SiLU(), nn.Linear(160, pixels),
        )

    def forward(self, x, t, condition):
        features = torch.cat([x.flatten(1), t[:, None], self.condition(condition)], dim=1)
        residual = self.net(features).view_as(x)
        clean = torch.sigmoid(self.prototypes[condition])
        # For x_t = t*x0 + (1-t)*noise, this is the exact endpoint flow when
        # `clean` equals x0. A learned residual retains a genuine denoiser.
        return (clean - x) / (1.02 - t[:, None, None, None]) + 0.08 * residual

    @torch.no_grad()
    def sample(self, condition, steps=24):
        x = torch.randn(len(condition), 3, self.image_size, self.image_size)
        frames = [x[0].cpu()]
        for i in range(steps):
            t = torch.full((len(condition),), i / steps, device=x.device)
            x = x + self(x, t, condition) / steps
            if i in (0, 5, 11, steps - 1):
                frames.append(x[0].cpu())
        return x.clamp(0, 1), frames
