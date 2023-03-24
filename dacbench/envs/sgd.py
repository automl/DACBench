import json
import math
import numbers
import random
import warnings
from enum import IntEnum, auto
from functools import reduce

import numpy as np
import torch
from backpack import backpack, extend
from backpack.extensions import BatchGrad
from torchvision import datasets, transforms

from dacbench import AbstractEnv

warnings.filterwarnings("ignore")


def reward_range(frange):
    def wrapper(f):
        f.frange = frange
        return f

    return wrapper


class Reward(IntEnum):
    TrainingLoss = auto()
    ValidationLoss = auto()
    LogTrainingLoss = auto()
    LogValidationLoss = auto()
    DiffTraining = auto()
    DiffValidation = auto()
    LogDiffTraining = auto()
    LogDiffValidation = auto()
    FullTraining = auto()

    def __call__(self, f):
        if hasattr(self, "func"):
            raise ValueError("Can not assign the same reward to a different function!")
        self.func = f
        return f


class SGDEnv(AbstractEnv):
    """
    Environment to control the learning rate of adam
    """

    def __init__(self, config):
        """
        Initialize SGD Env

        Parameters
        -------
        config : objdict
            Environment configuration
        """
        super(SGDEnv, self).__init__(config)

        self.batch_size = config.training_batch_size
        self.validation_batch_size = config.validation_batch_size
        self.no_cuda = config.no_cuda
        self.current_batch_size = config.training_batch_size
        self.on_features = config.features
        self.cd_paper_reconstruction = config.cd_paper_reconstruction
        self.cd_bias_correction = config.cd_bias_correction
        self.crashed = False
        self.terminate_on_crash = config.terminate_on_crash
        self.crash_penalty = config.crash_penalty

        if isinstance(config.reward_type, Reward):
            self.reward_type = config.reward_type
        elif isinstance(config.reward_type, str):
            try:
                self.reward_type = getattr(Reward, config.reward_type)
            except AttributeError:
                raise ValueError(f"{config.reward_type} is not a valid reward type!")
        else:
            raise ValueError(f"Type {type(config.reward_type)} is not valid!")

        self.use_cuda = not self.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.training_validation_ratio = config.train_validation_ratio
        self.dataloader_shuffle = config.dataloader_shuffle
        # self.test_dataset = None
        self.train_dataset = None
        self.validation_dataset = None
        self.train_loader = None
        # self.test_loader = None
        self.validation_loader = None
        self.train_loader_it = None
        self.validation_loader_it = None

        self.train_batch_index = 0
        self.epoch_index = 0

        self.current_training_loss = None
        self.loss_batch = None
        self.prev_training_loss = None
        self._current_validation_loss = torch.zeros(
            1, device=self.device, requires_grad=False
        )
        self._current_validation_loss.calculated = False
        self.prev_validation_loss = torch.zeros(
            1, device=self.device, requires_grad=False
        )

        self.model = None
        self.val_model = None
        # TODO:
        """
        TODO: Samuel Mueller (PhD student in our group) also uses backpack and has ran into a similar memory leak.
        He solved it calling this custom made RECURSIVE memory_cleanup function:
        # from backpack import memory_cleanup
        # def recursive_backpack_memory_cleanup(module: torch.nn.Module):
        #   memory_cleanup(module)
        #   for m in module.modules():
        #      memory_cleanup(m)
        (calling this after computing the training loss/gradients and after validation loss should suffice)
        """
        self.parameter_count = 0
        self.layer_sizes = []

        self.loss_function = config.loss_function(**config.loss_function_kwargs)
        self.loss_function = extend(self.loss_function)
        self.val_loss_function = config.loss_function(**config.val_loss_function_kwargs)

        self.initial_lr = config.lr * torch.ones(
            1, device=self.device, requires_grad=False
        )
        self.current_lr = config.lr * torch.ones(
            1, device=self.device, requires_grad=False
        )

        self.optimizer_name = config.optimizer

        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.epsilon = config.epsilon
        # RMSprop parameters
        self.beta2 = config.beta2
        self.m = 0
        self.v = 0
        # Momentum parameters
        self.sgd_momentum_v = 0
        self.sgd_rho = 0.9

        self.clip_grad = config.clip_grad

        self.t = 0
        self.step_count = torch.zeros(1, device=self.device, requires_grad=False)

        self.prev_direction = None
        self.current_direction = None

        self.predictiveChangeVarDiscountedAverage = torch.zeros(
            1, device=self.device, requires_grad=False
        )
        self.predictiveChangeVarUncertainty = torch.zeros(
            1, device=self.device, requires_grad=False
        )
        self.lossVarDiscountedAverage = torch.zeros(
            1, device=self.device, requires_grad=False
        )
        self.lossVarUncertainty = torch.zeros(
            1, device=self.device, requires_grad=False
        )
        self.discount_factor = config.discount_factor
        self.firstOrderMomentum = torch.zeros(
            1, device=self.device, requires_grad=False
        )
        self.secondOrderMomentum = torch.zeros(
            1, device=self.device, requires_grad=False
        )

        if self.optimizer_name == "adam":
            self.get_optimizer_direction = self.get_adam_direction
        elif self.optimizer_name == "rmsprop":
            self.get_optimizer_direction = self.get_rmsprop_direction
        elif self.optimizer_name == "momentum":
            self.get_optimizer_direction = self.get_momentum_direction
        else:
            raise NotImplementedError

        if "reward_function" in config.keys():
            self._get_reward = config["reward_function"]
        else:
            self._get_reward = self.reward_type.func

        if "state_method" in config.keys():
            self.get_state = config["state_method"]
        else:
            self.get_state = self.get_default_state

        self.reward_range = self.reward_type.func.frange

    def get_reward(self):
        return self._get_reward(self)

    @reward_range([-(10**9), 0])
    @Reward.TrainingLoss
    def get_training_reward(self):
        return -self.current_training_loss.item()

    @reward_range([-(10**9), 0])
    @Reward.ValidationLoss
    def get_validation_reward(self):
        return -self.current_validation_loss.item()

    @reward_range([-(10**9), (10**9)])
    @Reward.LogTrainingLoss
    def get_log_training_reward(self):
        return -torch.log(self.current_training_loss).item()

    @reward_range([-(10**9), (10**9)])
    @Reward.LogValidationLoss
    def get_log_validation_reward(self):
        return -torch.log(self.current_validation_loss).item()

    @reward_range([-(10**9), (10**9)])
    @Reward.LogDiffTraining
    def get_log_diff_training_reward(self):
        return -(
            torch.log(self.current_training_loss) - torch.log(self.prev_training_loss)
        ).item()

    @reward_range([-(10**9), (10**9)])
    @Reward.LogDiffValidation
    def get_log_diff_validation_reward(self):
        return -(
            torch.log(self.current_validation_loss)
            - torch.log(self.prev_validation_loss)
        ).item()

    @reward_range([-(10**9), (10**9)])
    @Reward.DiffTraining
    def get_diff_training_reward(self):
        return (self.current_training_loss - self.prev_training_loss).item()

    @reward_range([-(10**9), (10**9)])
    @Reward.DiffValidation
    def get_diff_validation_reward(self):
        return (self.current_validation_loss - self.prev_validation_loss).item()

    @reward_range([-(10**9), 0])
    @Reward.FullTraining
    def get_full_training_reward(self):
        return -self._get_full_training_loss(loader=self.train_loader).item()

    def get_full_training_loss(self):
        return -self.get_full_training_reward()

    @property
    def crash(self):
        self.crashed = True
        truncated = False
        terminated = False
        if self.c_step >= self.n_steps:
            truncated = True
        else:
            terminated = self.terminate_on_crash
        return self.get_state(self), self.crash_penalty, terminated, truncated, {}

    def seed(self, seed=None, seed_action_space=False):
        """
        Set rng seed

        Parameters
        ----------
        seed:
            seed for rng
        seed_action_space: bool, default False
            if to seed the action space as well
        """
        (seed,) = super().seed(seed, seed_action_space)
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        return [seed]

    def step(self, action):
        """
        Execute environment step

        Parameters
        ----------
        action : list
            action to execute

        Returns
        -------
        np.array, float, bool, bool, dict
            state, reward, terminated, truncated, info
        """
        truncated = super(SGDEnv, self).step_()

        self.step_count += 1
        index = 0

        if not isinstance(action, int) and not isinstance(action, float):
            action = action.item()
        if not isinstance(action, numbers.Number):
            action = action[0]

        if np.isnan(action):
            return self.crash

        new_lr = torch.Tensor([action]).to(self.device)
        self.current_lr = new_lr

        direction = self.get_optimizer_direction()
        if np.isnan(direction).any():
            return self.crash

        self.current_direction = direction

        delta_w = torch.mul(new_lr, direction)

        for i, p in enumerate(self.model.parameters()):
            layer_size = self.layer_sizes[i]
            p.data = p.data - delta_w[index : index + layer_size].reshape(
                shape=p.data.shape
            )
            index += layer_size

        self.model.zero_grad()

        self.prev_training_loss = self.current_training_loss
        if self._current_validation_loss.calculated:
            self.prev_validation_loss = self.current_validation_loss

        self.train_network()
        reward = self.get_reward()

        if np.isnan(reward):
            return self.crash

        state = self.get_state(self)
        for value in state.values():
            if np.isnan(value):
                return self.crash
        return state, reward, False, truncated, {}

    def _architecture_constructor(self, arch_str):
        layer_specs = []
        layer_strs = arch_str.split("-")
        for layer_str in layer_strs:
            idx = layer_str.find("(")
            if idx == -1:
                nn_module_name = layer_str
                vargs = []
            else:
                nn_module_name = layer_str[:idx]
                vargs_json_str = '{"tmp": [' + layer_str[idx + 1 : -1] + "]}"
                vargs = json.loads(vargs_json_str)["tmp"]
            layer_specs.append((getattr(torch.nn, nn_module_name), vargs))

        def model_constructor():
            layers = [cls(*vargs) for cls, vargs in layer_specs]
            return torch.nn.Sequential(*layers)

        return model_constructor

    def reset(self, seed=None, options={}):
        """
        Reset environment

        Returns
        -------
        np.array
            Environment state
        """
        super(SGDEnv, self).reset_(seed)

        dataset = self.instance[0]
        instance_seed = self.instance[1]
        construct_model = self._architecture_constructor(self.instance[2])
        self.n_steps = self.instance[3]
        dataset_size = self.instance[4]

        self.crashed = False

        self.seed(instance_seed)

        self.model = construct_model().to(self.device)
        self.val_model = construct_model().to(self.device)

        def init_weights(m):
            if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d:
                torch.nn.init.xavier_normal(m.weight)
                m.bias.data.fill_(0.0)

        if self.cd_paper_reconstruction:
            self.model.apply(init_weights)

        train_dataloader_args = {
            "batch_size": self.batch_size,
            "drop_last": True,
            "shuffle": self.dataloader_shuffle,
        }
        validation_dataloader_args = {
            "batch_size": self.validation_batch_size,
            "drop_last": True,
            "shuffle": False,
        }  # SA: shuffling empty data loader causes exception
        if self.use_cuda:
            param = {"num_workers": 1, "pin_memory": True}
            train_dataloader_args.update(param)
            validation_dataloader_args.update(param)

        if dataset == "MNIST":
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            )

            train_dataset = datasets.MNIST(
                "../data", train=True, download=True, transform=transform
            )
            # self.test_dataset = datasets.MNIST('../data', train=False, transform=transform)
        elif dataset == "CIFAR":
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )

            train_dataset = datasets.CIFAR10(
                "../data", train=True, download=True, transform=transform
            )
            # self.test_dataset = datasets.MNIST('../data', train=False, transform=transform)
        else:
            raise NotImplementedError

        if dataset_size is not None:
            train_dataset = torch.utils.data.Subset(
                train_dataset, range(0, dataset_size)
            )

        training_dataset_limit = math.floor(
            len(train_dataset) * self.training_validation_ratio
        )
        validation_dataset_limit = len(train_dataset)

        self.train_dataset = torch.utils.data.Subset(
            train_dataset, range(0, training_dataset_limit - 1)
        )
        self.validation_dataset = torch.utils.data.Subset(
            train_dataset, range(training_dataset_limit, validation_dataset_limit)
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, **train_dataloader_args
        )
        # self.test_loader = torch.utils.data.DataLoader(self.test_dataset, **train_dataloader_args)
        self.validation_loader = torch.utils.data.DataLoader(
            self.validation_dataset, **validation_dataloader_args
        )

        self.train_batch_index = 0
        self.epoch_index = 0
        self.train_loader_it = iter(self.train_loader)
        self.validation_loader_it = iter(self.validation_loader)

        self.parameter_count = 0
        self.layer_sizes = []
        for p in self.model.parameters():
            layer_size = reduce(lambda x, y: x * y, p.shape)
            self.layer_sizes.append(layer_size)
            self.parameter_count += layer_size

        self.model = extend(self.model)

        self.model.zero_grad()
        self.model.train()
        self.val_model.eval()

        self.current_training_loss = None
        self.loss_batch = None

        # Momentum parameters
        self.m = 0
        self.v = 0
        self.sgd_momentum_v = 0

        self.t = 0

        self.step_count = torch.zeros(1, device=self.device, requires_grad=False)

        self.current_lr = self.initial_lr
        self.prev_direction = torch.zeros(
            (self.parameter_count,), device=self.device, requires_grad=False
        )
        self.current_direction = torch.zeros(
            (self.parameter_count,), device=self.device, requires_grad=False
        )

        self.predictiveChangeVarDiscountedAverage = torch.zeros(
            1, device=self.device, requires_grad=False
        )
        self.predictiveChangeVarUncertainty = torch.zeros(
            1, device=self.device, requires_grad=False
        )
        self.lossVarDiscountedAverage = torch.zeros(
            1, device=self.device, requires_grad=False
        )
        self.lossVarUncertainty = torch.zeros(
            1, device=self.device, requires_grad=False
        )
        self.firstOrderMomentum = torch.zeros(
            1, device=self.device, requires_grad=False
        )
        self.secondOrderMomentum = torch.zeros(
            1, device=self.device, requires_grad=False
        )

        self._current_validation_loss = torch.zeros(
            1, device=self.device, requires_grad=False
        )
        self._current_validation_loss.calculated = False
        self.prev_validation_loss = torch.zeros(
            1, device=self.device, requires_grad=False
        )
        self.train_network()

        return self.get_state(self), {}

    def set_writer(self, writer):
        self.writer = writer

    def close(self):
        """
        No additional cleanup necessary

        Returns
        -------
        bool
            Cleanup flag
        """
        return True

    def render(self, mode: str = "human"):
        """
        Render env in human mode

        Parameters
        ----------
        mode : str
            Execution mode
        """
        if mode != "human":
            raise NotImplementedError

        pass

    def get_default_state(self, _):
        """
        Gather state description

        Returns
        -------
        dict
            Environment state

        """
        self.gradients = self._get_gradients()
        self.gradients = self.gradients.clip(*self.clip_grad)

        (
            self.firstOrderMomentum,
            self.secondOrderMomentum,
            self.sgdMomentum,
        ) = self._get_momentum(self.gradients)

        if (
            "predictiveChangeVarDiscountedAverage" in self.on_features
            or "predictiveChangeVarUncertainty" in self.on_features
        ):
            (
                predictiveChangeVarDiscountedAverage,
                predictiveChangeVarUncertainty,
            ) = self._get_predictive_change_features(self.current_lr)

        if (
            "lossVarDiscountedAverage" in self.on_features
            or "lossVarUncertainty" in self.on_features
        ):
            lossVarDiscountedAverage, lossVarUncertainty = self._get_loss_features()

        if "alignment" in self.on_features:
            alignment = self._get_alignment()

        state = {}

        if "predictiveChangeVarDiscountedAverage" in self.on_features:
            state[
                "predictiveChangeVarDiscountedAverage"
            ] = predictiveChangeVarDiscountedAverage.item()
        if "predictiveChangeVarUncertainty" in self.on_features:
            state[
                "predictiveChangeVarUncertainty"
            ] = predictiveChangeVarUncertainty.item()
        if "lossVarDiscountedAverage" in self.on_features:
            state["lossVarDiscountedAverage"] = lossVarDiscountedAverage.item()
        if "lossVarUncertainty" in self.on_features:
            state["lossVarUncertainty"] = lossVarUncertainty.item()
        if "currentLR" in self.on_features:
            state["currentLR"] = self.current_lr.item()
        if "trainingLoss" in self.on_features:
            if self.crashed:
                state["trainingLoss"] = 0.0
            else:
                state["trainingLoss"] = self.current_training_loss.item()
        if "validationLoss" in self.on_features:
            if self.crashed:
                state["validationLoss"] = 0.0
            else:
                state["validationLoss"] = self.current_validation_loss.item()
        if "step" in self.on_features:
            state["step"] = self.step_count.item()
        if "alignment" in self.on_features:
            state["alignment"] = alignment.item()
        if "crashed" in self.on_features:
            state["crashed"] = self.crashed

        return state

    def _train_batch_(self):
        (data, target) = next(self.train_loader_it)
        data, target = data.to(self.device), target.to(self.device)
        self.current_batch_size = data.size()[0]
        output = self.model(data)
        loss = self.loss_function(output, target)

        with backpack(BatchGrad()):
            loss.mean().backward()

        loss_value = loss.mean()

        self.loss_batch = loss
        self.current_training_loss = torch.unsqueeze(loss_value.detach(), dim=0)
        self.train_batch_index += 1
        self._current_validation_loss.calculated = False

    def train_network(self):
        try:
            self._train_batch_()
        except StopIteration:
            self.train_batch_index = 0
            self.epoch_index += 1
            self.train_loader_it = iter(self.train_loader)
            self._train_batch_()

    def _get_full_training_loss(self, loader):
        for target_param, param in zip(
            self.val_model.parameters(), self.model.parameters()
        ):
            target_param.data.copy_(param.data)
        loss = torch.zeros(1, device=self.device, requires_grad=False)
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.val_model(data)
                loss += self.val_loss_function(output, target).sum().detach().detach()

        loss /= len(loader.dataset)
        return loss

    @property
    def current_validation_loss(self):
        if not self._current_validation_loss.calculated:
            self._current_validation_loss = self._get_validation_loss()
            self._current_validation_loss.calculated = True
        return self._current_validation_loss

    def _get_validation_loss_(self):
        with torch.no_grad():
            (data, target) = next(self.validation_loader_it)
            data, target = data.to(self.device), target.to(self.device)
            output = self.val_model(data)
            validation_loss = self.val_loss_function(output, target).mean()
            validation_loss = torch.unsqueeze(validation_loss.detach(), dim=0)

        return validation_loss

    def _get_validation_loss(self):
        for target_param, param in zip(
            self.val_model.parameters(), self.model.parameters()
        ):
            target_param.data.copy_(param.data)
        try:
            validation_loss = self._get_validation_loss_()
        except StopIteration:
            self.validation_loader_it = iter(self.validation_loader)
            validation_loss = self._get_validation_loss_()

        return validation_loss

    def _get_gradients(self):
        gradients = []
        for p in self.model.parameters():
            if p.grad is None:
                continue
            gradients.append(p.grad.flatten())

        gradients = torch.cat(gradients, dim=0)

        return gradients

    def _get_momentum(self, gradients):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        self.v = self.beta2 * self.v + (1 - self.beta2) * torch.square(gradients)
        bias_corrected_m = self.m / (1 - self.beta1**self.t)
        bias_corrected_v = self.v / (1 - self.beta2**self.t)

        self.sgd_momentum_v = self.sgd_rho * self.sgd_momentum_v + gradients

        return bias_corrected_m, bias_corrected_v, self.sgd_momentum_v

    def get_adam_direction(self):
        return self.firstOrderMomentum / (
            torch.sqrt(self.secondOrderMomentum) + self.epsilon
        )

    def get_rmsprop_direction(self):
        return self.gradients / (torch.sqrt(self.secondOrderMomentum) + self.epsilon)

    def get_momentum_direction(self):
        return self.sgd_momentum_v

    def _get_loss_features(self):
        if self.crashed:
            return torch.tensor(0.0), torch.tensor(0.0)
        bias_correction = (
            (1 - self.discount_factor ** (self.c_step + 1))
            if self.cd_bias_correction
            else 1
        )
        with torch.no_grad():
            loss_var = torch.log(torch.var(self.loss_batch))
            self.lossVarDiscountedAverage = (
                self.discount_factor * self.lossVarDiscountedAverage
                + (1 - self.discount_factor) * loss_var
            )
            self.lossVarUncertainty = (
                self.discount_factor * self.lossVarUncertainty
                + (1 - self.discount_factor)
                * (loss_var - self.lossVarDiscountedAverage / bias_correction) ** 2
            )

        return (
            self.lossVarDiscountedAverage / bias_correction,
            self.lossVarUncertainty / bias_correction,
        )

    def _get_predictive_change_features(self, lr):
        if self.crashed:
            return torch.tensor(0.0), torch.tensor(0.0)
        bias_correction = (
            (1 - self.discount_factor ** (self.c_step + 1))
            if self.cd_bias_correction
            else 1
        )
        batch_gradients = []
        for i, (name, param) in enumerate(self.model.named_parameters()):
            grad_batch = param.grad_batch.reshape(
                self.current_batch_size, self.layer_sizes[i]
            )
            batch_gradients.append(grad_batch)

        batch_gradients = torch.cat(batch_gradients, dim=1)

        update_value = torch.mul(lr, self.get_optimizer_direction())

        predictive_change = torch.log(
            torch.var(-1 * torch.matmul(batch_gradients, update_value))
        )

        self.predictiveChangeVarDiscountedAverage = (
            self.discount_factor * self.predictiveChangeVarDiscountedAverage
            + (1 - self.discount_factor) * predictive_change
        )
        self.predictiveChangeVarUncertainty = (
            self.discount_factor * self.predictiveChangeVarUncertainty
            + (1 - self.discount_factor)
            * (
                predictive_change
                - self.predictiveChangeVarDiscountedAverage / bias_correction
            )
            ** 2
        )

        return (
            self.predictiveChangeVarDiscountedAverage / bias_correction,
            self.predictiveChangeVarUncertainty / bias_correction,
        )

    def _get_alignment(self):
        if self.crashed:
            return torch.tensor(0.0)
        alignment = torch.mean(
            torch.sign(torch.mul(self.prev_direction, self.current_direction))
        )
        alignment = torch.unsqueeze(alignment, dim=0)
        self.prev_direction = self.current_direction
        return alignment

    def generate_instance_file(self, file_name, mode="test", n=100):
        header = ["ID", "dataset", "architecture", "seed", "steps"]

        # dataset name, architecture, dataset size, sample dimension, number of max pool layers, hidden layers, test architecture convolutional layers
        architectures = [
            (
                "MNIST",
                "Conv2d(1, {0}, 3, 1, 1)-MaxPool2d(2, 2)-Conv2d({0}, {1}, 3, 1, 1)-MaxPool2d(2, 2)-Conv2d({1}, {2}, 3, 1, 1)-ReLU-Flatten-Linear({3}, 10)-LogSoftmax(1)",
                60000,
                28,
                2,
                3,
                [20, 50, 500],
            ),
            (
                "CIFAR",
                "Conv2d(3, {0}, 3, 1, 1)-MaxPool2d(2, 2)-ReLU-Conv2d({0}, {1}, 3, 1, 1)-ReLU-MaxPool2d(2, 2)-Conv2d({1}, {2}, 3, 1, 1)-ReLU-MaxPool2d(2, 2)-Conv2d({2}, {3}, 3, 1, 1)-ReLU-Flatten-Linear({4}, 10)-LogSoftmax(1)",
                60000,
                32,
                3,
                4,
                [32, 32, 64, 64],
            ),
        ]
        if mode == "test":
            seed_list = [random.randrange(start=0, stop=1e9) for _ in range(n)]

            for i in range(len(architectures)):
                fname = file_name + "_" + architectures[i][0].lower() + ".csv"

                steps = int(1e8)

                conv = architectures[i][6]
                hidden_layers = architectures[i][5]

                sample_size = architectures[i][3]
                pool_layer_count = architectures[i][4]
                linear_layer_size = conv[-1] * pow(
                    sample_size / pow(2, pool_layer_count), 2
                )
                linear_layer_size = int(round(linear_layer_size))

                dataset = architectures[i][0]

                if hidden_layers == 3:
                    architecture = architectures[i][1].format(
                        conv[0], conv[1], conv[2], linear_layer_size
                    )
                else:
                    architecture = architectures[i][1].format(
                        conv[0], conv[1], conv[2], conv[3], linear_layer_size
                    )

                # args = conv
                # args.append(linear_layer_size)
                # # architecture = architectures[i][1].format(**conv)
                # args = {0: conv[0], 1: conv[1], 2: conv[2], 3: linear_layer_size}
                # architecture = architectures[i][1].format(**args)

                with open(fname, "w", encoding="UTF8") as f:
                    for h in header:
                        f.write(h + ";")

                    f.write("\n")

                    for id in range(0, n):
                        f.write(str(id) + ";")

                        f.write(dataset + ";")
                        f.write(architecture + ";")

                        seed = seed_list[id]
                        f.write(str(seed) + ";")

                        f.write(str(steps) + ";")

                        f.write("\n")
                    f.close()

        else:
            dataset_index = 0

            dataset_size_start = 0.1
            dataset_size_stop = 0.5

            steps_start = 300
            steps_stop = 1000

            conv1_start = 2
            conv1_stop = 10
            conv2_start = 5
            conv2_stop = 25
            conv3_start = 50
            conv3_stop = 250

            dataset_list = [dataset_index for _ in range(n)]

            dataset_size_list = [
                random.uniform(dataset_size_start, dataset_size_stop) for _ in range(n)
            ]

            seed_list = [random.randrange(start=0, stop=1e9) for _ in range(n)]

            steps_list = [
                random.randrange(start=steps_start, stop=steps_stop) for _ in range(n)
            ]

            conv1_list = [
                random.randrange(start=conv1_start, stop=conv1_stop) for _ in range(n)
            ]
            conv2_list = [
                random.randrange(start=conv2_start, stop=conv2_stop) for _ in range(n)
            ]
            conv3_list = [
                random.randrange(start=conv3_start, stop=conv3_stop) for _ in range(n)
            ]

            fname = file_name + ".csv"
            with open(fname, "w", encoding="UTF8") as f:
                for h in header:
                    f.write(h + ";")

                f.write("\n")

                for id in range(0, n):
                    f.write(str(id) + ";")

                    sample_size = architectures[dataset_list[id]][3]
                    pool_layer_count = architectures[dataset_list[id]][4]
                    linear_layer_size = conv3_list[id] * pow(
                        sample_size / pow(2, pool_layer_count), 2
                    )
                    linear_layer_size = int(round(linear_layer_size))

                    dataset_size = int(
                        dataset_size_list[id] * architectures[dataset_list[id]][2]
                    )
                    dataset = (
                        architectures[dataset_list[id]][0] + "_" + str(dataset_size)
                    )
                    architecture = architectures[dataset_list[id]][1].format(
                        conv1_list[id],
                        conv2_list[id],
                        conv3_list[id],
                        linear_layer_size,
                    )

                    f.write(dataset + ";")
                    f.write(architecture + ";")

                    seed = seed_list[id]
                    f.write(str(seed) + ";")

                    steps = steps_list[id]
                    f.write(str(steps) + ";")

                    f.write("\n")
                f.close()
