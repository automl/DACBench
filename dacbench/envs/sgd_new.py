import torch

from dacbench import AbstractMADACEnv
from dacbench.envs.env_utils.utils import random_torchvision_loader


def optimizer_action(
    optimizer: torch.optim.Optimizer, action: float, use_momentum: bool
) -> None:
    for g in optimizer.param_groups:
        g["lr"] = action[0]
        if use_momentum:
            print("Momentum")
            g["betas"] = (action[1], 0.999)
    return optimizer


def test(
    model,
    loss_function,
    loader,
    batch_size,
    batch_percentage: float = 1.0,
    device="cpu",
):
    """Evaluate given `model` on `loss_function`.

    Percentage defines how much percentage of the data shall be used. If nothing given the whole data is used.

    Returns:
        test_losses: Batch validation loss per data point
    """
    nmb_sets = batch_percentage * (len(loader.dataset) / batch_size)
    model.eval()
    test_losses = []
    i = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_losses.append(loss_function(output, target))
            i += 1
            if i >= nmb_sets:
                break
    test_losses = torch.cat(test_losses)
    return test_losses


def forward_backward(model, loss_function, loader, device="cpu"):
    """Do a forward and a backward pass for given `model` for `loss_function`.

    Returns:
        loss: Mini batch training loss per data point
    """
    model.train()
    (data, target) = next(iter(loader))
    data, target = data.to(device), target.to(device)
    output = model(data)
    loss = loss_function(output, target)
    loss.mean().backward()
    return loss


class SGDEnv(AbstractMADACEnv):
    """
    The SGD DAC Environment implements the problem of dynamically configuring the learning rate hyperparameter of a
    neural network optimizer (more specifically, torch.optim.AdamW) for a supervised learning task. While training,
    the model is evaluated after every epoch.

    Actions correspond to learning rate values in [0,+inf[
    For observation space check `observation_space` method docstring.
    For instance space check the `SGDInstance` class docstring
    Reward:
        negative loss of model on test_loader of the instance       if done
        crash_penalty of the instance                               if crashed
        0                                                           otherwise
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config):
        """Init env."""
        super(SGDEnv, self).__init__(config)
        self.device = config.get("device")

        self.learning_rate = None
        self.optimizer_type = torch.optim.AdamW
        self.optimizer_params = config.get("optimizer_params")
        self.batch_size = config.get("training_batch_size")
        self.model = config.get("model")
        self.crash_penalty = config.get("crash_penalty")
        self.loss_function = config.loss_function(**config.loss_function_kwargs)
        self.dataset_name = config.get("dataset_name")
        self.use_momentum = config.get("use_momentum")

        # Use default reward function, if no specific function is given
        if "reward_function" in config.keys():
            self.get_reward = config["reward_function"]
        else:
            self.get_reward = self.get_default_reward

        # Use default state function, if no specific function is given
        if "state_method" in config.keys():
            self.get_state = config["state_method"]
        else:
            self.get_state = self.get_default_state

        # Get loaders for instance
        self.datasets, loaders = random_torchvision_loader(
            config.get("seed"),
            config.get("instance_set_path"),
            self.dataset_name,
            self.batch_size,
            config.get("fraction_of_dataset"),
            config.get("train_validation_ratio"),
        )
        self.train_loader, self.validation_loader, self.test_loader = loaders

    def step(self, action: float):
        """
        Update the parameters of the neural network using the given learning rate lr,
        in the direction specified by AdamW, and if not done (crashed/cutoff reached),
        performs another forward/backward pass (update only in the next step)."""
        truncated = super(SGDEnv, self).step_()
        info = {}
        if isinstance(action, float):
            action = [action]

        self.optimizer = optimizer_action(self.optimizer, action, self.use_momentum)
        self.optimizer.step()
        self.optimizer.zero_grad()

        train_args = [
            self.model,
            self.loss_function,
            self.train_loader,
            self.device,
        ]
        self.loss = forward_backward(*train_args)

        crashed = (
            not torch.isfinite(self.loss).any()
            or not torch.isfinite(
                torch.nn.utils.parameters_to_vector(self.model.parameters())
            ).any()
        )

        if crashed:
            self._done = True
            return (
                self.get_state(self),
                self.crash_penalty,
                False,
                True,
                info,
            )

        self._done = truncated

        if (
            self.n_steps % len(self.train_loader) == 0 or self._done
        ):  # Calculate validation loss at the end of an epoch
            batch_percentage = 1.0
        else:
            batch_percentage = 0.1

        val_args = [
            self.model,
            self.loss_function,
            self.validation_loader,
            self.batch_size,
            batch_percentage,
            self.device,
        ]
        validation_loss = test(*val_args)

        self.validation_loss = validation_loss.mean()
        if (
            self.min_validation_loss is None
            or self.validation_loss <= self.min_validation_loss
        ):
            self.min_validation_loss = self.validation_loss

        if self._done:
            val_args = [
                self.model,
                self.loss_function,
                self.test_loader,
                self.batch_size,
                1.0,
                self.device,
            ]
            self.test_losses = test(*val_args)

        reward = self.get_reward(self)

        return self.get_state(self), reward, False, truncated, info

    def reset(self, seed=None, options={}):
        """Initialize the neural network, data loaders, etc. for given/random next task. Also perform a single
        forward/backward pass, not yet updating the neural network parameters."""
        super(SGDEnv, self).reset_(seed)

        self.learning_rate = None
        self.optimizer_type = torch.optim.AdamW
        self.info = {}
        self._done = False

        self.model.to(self.device)
        self.optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            **self.optimizer_params, params=self.model.parameters()
        )
        self.loss = 0
        self.test_losses = None

        self.validation_loss = None
        self.min_validation_loss = None

        return self.get_state(self), {}

    def get_default_reward(self, _):
        if self.test_losses is not None:
            reward = self.test_losses.sum().item() / len(self.test_loader.dataset)
        else:
            reward = 0.0
        return reward

    def get_default_state(self, _):
        return {
            "step": self.c_step,
            "loss": self.loss,
            "validation_loss": self.validation_loss,
            "done": self._done,
        }

    def render(self, mode="human"):
        if mode == "human":
            epoch = 1 + self.c_step // len(self.train_loader)
            epoch_cutoff = self.n_steps // len(self.train_loader)
            batch = 1 + self.c_step % len(self.train_loader)
            print(
                f"prev_lr {self.optimizer.param_groups[0]['lr'] if self.n_steps > 0 else None}, "
                f"epoch {epoch}/{epoch_cutoff}, "
                f"batch {batch}/{len(self.train_loader)}, "
                f"batch_loss {self.loss}, "
                f"val_loss {self.validation_loss}"
            )
        else:
            raise NotImplementedError
