import os
import random
import copy
from collections import namedtuple
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms

datasets.CIFAR10.download


DATASETS = {
    "MNIST": {
        "transform": transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
        "icgen_name": "mnist",
    },
    "CIFAR10": {
        "transform": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
        "icgen_name": "cifar10",
    },
    "FashionMNIST": {
        "transform": transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        ),
        "icgen_name": "fashion_mnist",
    },
}


def random_torchvision_loader(
    dataset_path: str,
    name: str,
    batch_size: int,
    noisy_validation_batch_size: int,
    fraction_of_dataset: float,
    train_validation_ratio: float,
    shuffle_training: bool,
    **kwargs,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, test loaders for `name` dataset."""
    transform = DATASETS[name]["transform"]
    train_dataset = getattr(datasets, name)(
        dataset_path, train=True, download=True, transform=transform
    )
    train_size = int(len(train_dataset) * fraction_of_dataset)
    classes = train_dataset.classes
    train_dataset, _ = torch.utils.data.random_split(
        train_dataset, [train_size, len(train_dataset) - train_size]
    )
    train_dataset.classes = classes
    test = getattr(datasets, name)(dataset_path, train=False, transform=transform)
    train_size = int(len(train_dataset) * train_validation_ratio)
    train_size = train_size - train_size % batch_size
    train, val = torch.utils.data.random_split(
        train_dataset, [train_size, len(train_dataset) - train_size]
    )
    train_loader = DataLoader(
        train, batch_size=batch_size, drop_last=True, shuffle=shuffle_training
    )
    val_loader = DataLoader(val, batch_size=noisy_validation_batch_size)
    test_loader = DataLoader(test, batch_size=64)
    return (train_dataset, test), (train_loader, val_loader, test_loader)


def random_architecture(
    rng: np.random.RandomState,
    input_shape: Tuple[int, int, int],
    n_classes: int,
    **kwargs,
) -> Tuple[nn.Module, int]:
    """Samples random architecture with `rng` for given `input_shape` and `n_classes`."""
    modules = [nn.Identity()]
    max_n_conv_layers = 3
    n_conv_layers = rng.randint(low=0, high=max_n_conv_layers + 1)
    prev_conv = input_shape[0]
    kernel_sizes = [3, 5, 7][: max(0, 3 - n_conv_layers + 1)]
    activation = rng.choice([nn.Identity, nn.ReLU, nn.PReLU, nn.ELU])
    batch_norm_2d = rng.choice([nn.Identity, nn.BatchNorm2d])
    bn_first = rng.choice([False, True])

    for layer_idx, layer_exp in enumerate(range(1, int(n_conv_layers * 2 + 1), 2)):
        if layer_idx > 0:
            modules.append(nn.MaxPool2d(2))
        conv = int(
            np.exp(
                rng.uniform(
                    low=np.log(2**layer_exp), high=np.log(2 ** (layer_exp + 2))
                )
            )
        )
        kernel_size = rng.choice(kernel_sizes)
        modules.append(nn.Conv2d(prev_conv, conv, kernel_size, 1))
        prev_conv = conv
        if bn_first:
            modules.append(batch_norm_2d(prev_conv))
        modules.append(activation())
        if not bn_first:
            modules.append(batch_norm_2d(prev_conv))

    feature_extractor = nn.Sequential(*modules)

    linear_layers = [nn.Flatten()]
    batch_norm_1d = rng.choice([nn.Identity, nn.BatchNorm1d])
    max_n_mlp_layers = 2
    n_mlp_layers = int(rng.randint(low=0, high=max_n_mlp_layers + 1))
    prev_l = int(
        torch.prod(
            torch.tensor(feature_extractor(torch.zeros((1, *input_shape))).shape)
        ).item()
    )
    for layer_idx in range(n_mlp_layers):
        l = 2 ** (
            2 ** (max_n_mlp_layers + 1 - layer_idx)
            - int(
                np.exp(
                    rng.uniform(
                        low=np.log(1), high=np.log(max_n_mlp_layers + 1 + layer_idx)
                    )
                )
            )
        )
        linear_layers.append(nn.Linear(prev_l, l))
        prev_l = l
        if bn_first:
            linear_layers.append(batch_norm_1d(prev_l))
        linear_layers.append(activation())
        if not bn_first:
            linear_layers.append(batch_norm_1d(prev_l))

    linear_layers.append(nn.Linear(prev_l, n_classes))
    linear_layers.append(nn.LogSoftmax(1))
    mlp = nn.Sequential(*linear_layers)
    return nn.Sequential(feature_extractor, mlp), n_conv_layers
