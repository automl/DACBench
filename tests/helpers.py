from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset


def tiny_sgd_loader(*args, **kwargs):
    """Tiny seeded in-memory dataset replacing CIFAR-10 for fast SGD tests.

    Uses a local Generator so the global PyTorch RNG is not affected.
    """
    rng = torch.Generator().manual_seed(0)
    x = torch.rand(256, 3, 32, 32, generator=rng)
    y = torch.randint(0, 10, (256,), generator=rng)
    train_ds = TensorDataset(x[:192], y[:192])
    val_ds = TensorDataset(x[192:], y[192:])
    test_ds = TensorDataset(x[:64], y[:64])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True, generator=torch.Generator().manual_seed(0))
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(0))
    test_loader = DataLoader(test_ds, batch_size=32)
    return (train_ds, test_ds), (train_loader, val_loader, test_loader)
