from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import Dataset

@dataclass(frozen=True)
class SyntheticConfig:
    n_samples: int
    input_dim: int
    num_classes: int
    seed: int = 42

class SyntheticClassificationDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, cfg: SyntheticConfig) -> None:
        g = torch.Generator().manual_seed(cfg.seed)
        self.x = torch.randn(cfg.n_samples, cfg.input_dim, generator=g, dtype=torch.float32)
        self.y = torch.randint(0, cfg.num_classes, (cfg.n_samples,), generator=g, dtype=torch.int64)
    
    def __len__(self) -> int:
        return int(self.y.shape[0])
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]