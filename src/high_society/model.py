"""The network (spec §4.2): a shared opponent encoder, a trunk, and two heads.

opponents ─► same small network for each ─► max + mean ─┐
me, the card, the game ─────────────────────────────────┴─► trunk ─► policy (17 actions)
                                                                  └─► value (chance to win)
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from .encoding import MAIN_FEATURES, N_ACTIONS, OPPONENT_FEATURES


@dataclass(frozen=True)
class ModelConfig:
    opponent_width: int = 64
    width: int = 256
    layers: int = 2


class PolicyValueNet(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        # The same weights read every opponent: what it learns about one applies to all.
        self.opponent = nn.Sequential(
            nn.Linear(OPPONENT_FEATURES, c.opponent_width),
            nn.ReLU(),
            nn.Linear(c.opponent_width, c.opponent_width),
            nn.ReLU(),
        )
        trunk: list[nn.Module] = []
        width_in = MAIN_FEATURES + 2 * c.opponent_width
        for _ in range(c.layers):
            trunk += [nn.Linear(width_in, c.width), nn.LayerNorm(c.width), nn.ReLU()]
            width_in = c.width
        self.trunk = nn.Sequential(*trunk)
        self.policy = nn.Linear(c.width, N_ACTIONS)
        self.value = nn.Linear(c.width, 1)
        # Start close to "no preference" and "no idea": small last-layer weights.
        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        nn.init.zeros_(self.policy.bias)

    def forward(
        self, main: Tensor, opponents: Tensor, present: Tensor, legal: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Returns the action logits (illegal ones at -inf) and the estimated chance to win."""
        encoded = self.opponent(opponents)  # (B, 4, W)
        mask = present.unsqueeze(-1)  # (B, 4, 1)
        # Absent seats can't be the max, and don't count in the mean.
        strongest = (encoded + (mask - 1) * 1e9).amax(dim=1)
        average = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        hidden = self.trunk(torch.cat([main, strongest, average], dim=-1))
        logits = self.policy(hidden).masked_fill(~legal, float("-inf"))
        value = torch.sigmoid(self.value(hidden)).squeeze(-1)
        return logits, value

    def save(self, path: str, **extra: object) -> None:
        """Written to a temporary file first: stopping mid-save never leaves a broken file."""
        temporary = f"{path}.tmp"
        torch.save({"config": asdict(self.config), "state": self.state_dict(), **extra}, temporary)
        os.replace(temporary, path)

    @classmethod
    def load(cls, path: str) -> tuple[PolicyValueNet, dict[str, object]]:
        data = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(ModelConfig(**data["config"]))
        model.load_state_dict(data["state"])
        model.eval()
        return model, data
