from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from .synthetic import SyntheticClassificationDataset, SyntheticConfig

@dataclass(frozen=True)
class DataConfig:
    name: str = "synthetic_classification"
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    seed: int = 42
    shuffle_train: bool = True

def build_datasets(cfg: Any) -> Tuple[Dataset, Dataset]:
    # expects cfg.data.name etc.
    if cfg.data.name == "synthetic_classification":
        train = SyntheticClassificationDataset(
            SyntheticConfig(
                n_samples=int(cfg.synthetic.n_train),
                input_dim=int(cfg.synthetic.input_dim),
                num_classes=int(cfg.synthetic.num_classes),
                seed=int(cfg.data.seed),
            )
        )
        val = SyntheticClassificationDataset(
            SyntheticConfig(
                n_samples=int(cfg.synthetic.n_val),
                input_dim=int(cfg.synthetic.input_dim),
                num_classes=int(cfg.synthetic.num_classes),
                seed=int(cfg.data.seed) + 1,
            )
        )
        return train, val

    raise ValueError(f"Unknown dataset: {cfg.data.name}")

def get_dataloaders(cfg: Any) -> Tuple[DataLoader, DataLoader]:
    train_ds, val_ds = build_datasets(cfg)

    g = torch.Generator().manual_seed(int(cfg.data.seed))
    persistent = bool(cfg.data.persistent_workers and cfg.data.num_workers > 0)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.data.batch_size),
        shuffle=bool(cfg.data.shuffle_train),
        num_workers=int(cfg.data.num_workers),
        pin_memory=bool(cfg.data.pin_memory),
        persistent_workers=persistent,
        generator=g,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.data.batch_size),
        shuffle=False,
        num_workers=int(cfg.data.num_workers),
        pin_memory=bool(cfg.data.pin_memory),
        persistent_workers=persistent,
        generator=g,
    )
    return train_loader, val_loader
