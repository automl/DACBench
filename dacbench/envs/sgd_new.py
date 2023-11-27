import copy
from typing import Iterator, Optional, Tuple, Union, Dict

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from dacbench import AbstractMADACEnv
from dacbench.envs.env_utils.utils import random_torchvision_loader


def optimizer_action(optimizer: torch.optim.Optimizer, action: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = action[0]
        print(f"learning rate: {g['lr']}")
    return optimizer


def test(model, loss_function, loader, device="cpu"):
    """Evaluate given `model` on `loss_function`.

    Returns:
        test_losses: Full batch validation loss per data point
    """
    model.eval()
    test_losses = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_losses.append(loss_function(output, target))
    test_losses = torch.cat(test_losses)
    return test_losses


def forward_backward(model, loss_function, loader, device="cpu"):
    """Do a forward and a backward pass for given `model` for `loss_function`.

    Returns:
        loss: Mini batch training loss per data point
    """
    model.train()
    (data, target) = next(loader)
    data, target = data.to(device), target.to(device)
    output = model(data)
    loss = loss_function(output, target)
    loss.mean().backward()
    return loss


def noisy_validate(model, loss_function, loader, device="cpu"):
    model.eval()
    with torch.no_grad():
        (data, target) = next(loader)
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = loss_function(output, target)
    return loss


class SGDEnv(AbstractMADACEnv):
    """
    The SGD DAC Environment implements the problem of dynamically configuring the learning rate hyperparameter of a
    neural network optimizer (more specifically, torch.optim.AdamW) for a supervised learning task. While training,
    the model is evaluated after every epoch and a checkpoint of the model with minimal validation loss is retained.

    Actions correspond to learning rate values in [0,+inf[
    For observation space check `observation_space` method docstring.
    For instance space check the `SGDInstance` class docstring
    Reward:
        negative loss of checkpoint on test_loader of the instance  if done and a valid checkpoint is available
        crash_penalty of the instance                               if done and no checkpoint is available
                                                                    (~ crash / divergence in first epoch)
        0                                                           otherwise
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config):
        """Init env."""
        super(SGDEnv, self).__init__(config)
        self.device = config.get("device")

        self.learning_rate = None
        self.optimizer_type = torch.optim.AdamW
        self.train_iter: Iterator[Tuple[torch.Tensor, torch.Tensor]]
        self.validation_iter: Iterator[Tuple[torch.Tensor, torch.Tensor]]
        self.optimizer_params = config.get("optimizer_params")
        self.batch_size = config.get("training_batch_size")
        self.model = config.get("model")
        self.crash_penalty = config.get("crash_penalty")
        self.loss_function = config.loss_function(**config.loss_function_kwargs)

        # Get loaders for instance
        datasets, loaders = random_torchvision_loader(
            config.get("instance_set_path"),
            "MNIST",
            self.batch_size,
            self.batch_size,
            config.get("fraction_of_dataset"),
            config.get("train_validation_ratio"),
            True,
        )
        self.train_loader, self.validation_loader, self.test_loader = loaders

    def step(self, action: float):
        """
        Update the parameters of the neural network using the given learning rate lr,
        in the direction specified by AdamW, and if not done (crashed/cutoff reached),
        performs another forward/backward pass (update only in the next step)."""
        truncated = super(SGDEnv, self).step_()
        info = {}

        # default_rng_state = torch.get_rng_state()
        # torch.set_rng_state(self.env_rng_state)

        self.optimizer = optimizer_action(self.optimizer, action)
        self.optimizer.step()
        self.optimizer.zero_grad()

        train_args = [
            self.model,
            self.loss_function,
            self.train_iter,
            self.device,
        ]
        try:
            self.loss = forward_backward(*train_args)
        except StopIteration:
            self.train_iter = iter(self.train_loader)
            train_args[2] = self.train_iter
            self.loss = forward_backward(*train_args)

        # self.env_rng_state.data = torch.get_rng_state()
        # torch.set_rng_state(default_rng_state)
        crashed = (
            not torch.isfinite(self.loss).any()
            or not torch.isfinite(
                torch.nn.utils.parameters_to_vector(self.model.parameters())
            ).any()
        )

        self._done = truncated or crashed

        if (
            self.n_steps % len(self.train_loader) == 0 or self._done
        ):  # Calculate validation loss at the end of an epoch
            validation_loss = test(
                self.model, self.loss_function, self.validation_loader, self.device
            )

            self.validation_loss_last_epoch = validation_loss.mean()
            if self.validation_loss_last_epoch <= self.min_validation_loss:
                self.min_validation_loss = self.validation_loss_last_epoch
                self.checkpoint = copy.deepcopy(self.model)

        else:
            val_args = [
                self.model,
                self.loss_function,
                self.validation_iter,
                self.device,
            ]
            try:
                validation_loss = noisy_validate(*val_args)
            except StopIteration:
                self.validation_iter = iter(self.validation_loader)
                val_args[2] = self.validation_iter
                validation_loss = noisy_validate(*val_args)

        state = {
            "step": self.n_steps,
            "loss": self.loss,
            "validation_loss": validation_loss,
            "done": self._done,
        }

        if self._done:
            if self.checkpoint is None:
                reward = -self.crash_penalty
            else:
                test_losses = test(
                    self.checkpoint, self.loss_function, self.test_loader, self.device
                )
                reward = max(
                    -test_losses.sum().item() / len(self.test_loader.dataset),
                    -self.crash_penalty,
                )
        else:
            reward = 0.0
        return state, reward, False, truncated, info

    def reset(self, seed=None, options={}):
        """Initialize the neural network, data loaders, etc. for given/random next task. Also perform a single
        forward/backward pass, not yet updating the neural network parameters."""
        super(SGDEnv, self).reset_(seed)

        self.learning_rate = 0
        self.optimizer_type = torch.optim.AdamW
        self.train_iter: Iterator[Tuple[torch.Tensor, torch.Tensor]]
        self.validation_iter: Iterator[Tuple[torch.Tensor, torch.Tensor]]
        self.info = {}

        self.train_iter = iter(self.train_loader)
        self.validation_iter = iter(self.validation_loader)
        self.model.to(self.device)
        self.optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            **self.optimizer_params, params=self.model.parameters()
        )

        self.optimizer.zero_grad()
        self.loss = forward_backward(
            self.model,
            self.loss_function,
            self.train_iter,
            self.device,
        )
        self.env_rng_state: torch.Tensor = copy.deepcopy(torch.get_rng_state())
        self.validation_loss_last_epoch = None
        self.checkpoint = None
        self.min_validation_loss = self.crash_penalty
        val_args = [
            self.model,
            self.loss_function,
            iter(self.validation_loader),
            self.device,
        ]
        validation_loss = noisy_validate(*val_args)
        return {
            "step": 0,
            "loss": self.loss,
            "validation_loss": validation_loss,
            "crashed": False,
        }, {}

    # def seed(self, seed=None):
    #     torch.backends.cudnn.benchmark = False
    #     torch.backends.cudnn.deterministic = True
    #     return super().seed(seed)

    def render(self, mode="human"):
        if mode == "human":
            epoch = 1 + self.n_steps // len(self.train_loader)
            epoch_cutoff = self.cutoff // len(self.train_loader)
            batch = 1 + self.n_steps % len(self.train_loader)
            print(
                f"prev_lr {self.optimizer.param_groups[0]['lr'] if self.n_steps > 0 else None}, "
                f"epoch {epoch}/{epoch_cutoff}, "
                f"batch {batch}/{len(self.train_loader)}, "
                f"batch_loss {self.loss.mean()}, "
                f"val_loss_last_epoch {self.validation_loss_last_epoch}"
            )
        else:
            raise NotImplementedError
