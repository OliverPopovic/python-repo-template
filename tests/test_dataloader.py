from __future__ import annotations

from types import SimpleNamespace

import torch

from python_repo_template.data.dataloader import get_dataloaders

def _cfg(seed: int = 42):
    # Minimal Hydra-like config object: cfg.data.*, cfg.synthetic.*
    return SimpleNamespace(
        data=SimpleNamespace(
            name="synthetic_classification",
            batch_size=8,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            seed=seed,
            shuffle_train=True,
        ),
        synthetic=SimpleNamespace(
            n_train=64,
            n_val=16,
            input_dim=12,
            num_classes=5,
        ),
    )


def test_get_dataloaders_smoke_and_batch_contract() -> None:
    cfg1 = _cfg(seed=123)
    train_loader_1, val_loader_1 = get_dataloaders(cfg1)

    xb, yb = next(iter(train_loader_1))

    # batch contract
    assert isinstance(xb, torch.Tensor)
    assert isinstance(yb, torch.Tensor)
    assert xb.dtype == torch.float32
    assert yb.dtype == torch.int64
    assert xb.shape == (cfg1.data.batch_size, cfg1.synthetic.input_dim)
    assert yb.shape == (cfg1.data.batch_size,)

    # val loader also yields correctly shaped batches
    xb_val, yb_val = next(iter(val_loader_1))
    assert xb_val.shape[1] == cfg1.synthetic.input_dim
    assert yb_val.ndim == 1

    # minimal determinism check (same seed => same first batch)
    cfg2 = _cfg(seed=123)
    train_loader_2, _ = get_dataloaders(cfg2)
    xb2, yb2 = next(iter(train_loader_2))

    assert torch.equal(xb, xb2)
    assert torch.equal(yb, yb2)
