import math
import numbers
import warnings
import json
from functools import reduce, singledispatchmethod
from copy import deepcopy
from contextlib import contextmanager
import random
import inspect
from dataclasses import dataclass, field


import torch
from backpack import backpack, extend
from backpack.extensions import BatchGrad
from torchvision import datasets, transforms
import numpy as np
from dacbench import AbstractEnv

warnings.filterwarnings("ignore")


@dataclass
class ExponentialMovingAverage:
    alpha: float = 0.0
    bias_correction: bool = False
    mean: float = field(default=0.0, init=False)
    variance: float = field(default=0.0, init=False)
    t: int = field(default=0, init=False)

    def update(self, value):
        self.t += 1
        bias_correction = 1 - self.alpha ** self.t if self.bias_correction else 1
        self.mean = self.alpha * self.mean + (1 - self.alpha) * value
        mean_hat = self.mean / bias_correction
        self.variance = (
            self.alpha * self.variance + (1 - self.alpha) * (value - mean_hat) ** 2
        )
        return mean_hat, self.variance / bias_correction


@contextmanager
def fake_step(model, optimizer):
    optimizer_state = deepcopy(optimizer.state_dict())
    model_state = deepcopy(model.state_dict())
    try:
        yield optimizer
    finally:
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)


@dataclass
class Reward:
    low: float
    high: float


class TrainingLoss(Reward):
    pass


class ValidationLoss(Reward):
    pass


class LogTrainingLoss(Reward):
    pass


class LogValidationLoss(Reward):
    pass


class DiffTraining(Reward):
    pass


class DiffValidation(Reward):
    pass


class LogDiffTraining(Reward):
    pass


class LogDiffValidation(Reward):
    pass


class FullTraining(Reward):
    pass


training_loss = TrainingLoss(-(10 ** 9), 0)
validation_loss = ValidationLoss(-(10 ** 9), 10 ** 9)
log_training_loss = LogTrainingLoss(-(10 ** 9), 10 ** 9)
log_validation_loss = LogTrainingLoss(-(10 ** 9), 10 ** 9)
diff_training_loss = DiffTraining(-(10 ** 9), 10 ** 9)
diff_validation_loss = DiffValidation(-(10 ** 9), 10 ** 9)
log_diff_training_loss = LogDiffTraining(-(10 ** 9), 10 ** 9)
log_diff_validation_loss = LogDiffValidation(-(10 ** 9), 10 ** 9)
full_training_loss = FullTraining(-(10 ** 9), 0)


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
        self.config = config.copy()
        sig = inspect.signature(self.config.optimizer)
        for name, _ in self.config.actions.items():
            if name not in sig.parameters:
                raise ValueError(
                    f"{name} is not a valid {self.config.optimizer} param."
                )

        ######################## Custom ###############################
        self.action_names = self.config.actions.keys()
        num_actions = [len(values) for values in config.actions.values()]
        self.config.action_space_args = [int(np.prod(num_actions))]

        # alle permutationen mappen  [[0.2, 0.9, ...], ...]
        import itertools

        s = [values for values in config.actions.values()]
        all_permutations = list(itertools.product(*s))
        self.action_mapping = [perm for perm in all_permutations]

        self.action_mapper = {}
        for idx, prod_idx in zip(
            range(len(all_permutations)),
            itertools.product(*[np.arange(val) for val in config["action_values"]]),
        ):
            self.action_mapper[idx] = prod_idx
        ########################## END #################################

        if isinstance(self.config.reward_type, str):
            try:
                self.config.reward_type = globals()[config.reward_type]
            except AttributeError:
                raise ValueError(f"{config.reward_type} is not a valid reward type!")
        assert isinstance(self.config.reward_type, Reward)
        self.config.reward_range = [
            self.config.reward_type.low,
            self.config.reward_type.high,
        ]

        super(SGDEnv, self).__init__(self.config)

        self.crashed = False

        self.use_cuda = not self.config.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.training_validation_ratio = config.train_validation_ratio
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
        self.parameter_count = (
            0  # TODO: Verify that we still need this if we use pytorch.optim
        )
        self.layer_sizes = (
            []
        )  # TODO: Verify that we still need this if we use pytorch.optim

        self.loss_function = config.loss_function(**config.loss_function_kwargs)
        self.loss_function = extend(self.loss_function)
        self.val_loss_function = config.loss_function(**config.val_loss_function_kwargs)

        # TODO: Verify that we still need this if we use pytorch.optim (initial lr is just an optimizer_kwargs)
        self.initial_lr = config.lr * torch.ones(
            1, device=self.device, requires_grad=False
        )
        self.current_lr = config.lr * torch.ones(
            1, device=self.device, requires_grad=False
        )

        self.prev_direction = None
        self.current_direction = None

        self.learning_rate = 0.001  # TODO: Yet another lr? Is this used?
        self.discount_factor = 0.9  # TODO: Make this part of the config

        if "state_method" in config.keys():
            self.get_state = config["state_method"]
        else:
            self.get_state = self.get_default_state

    @singledispatchmethod
    def get_reward(self, reward_type):
        raise NotImplementedError

    @get_reward.register
    def _(self, reward_type: TrainingLoss):
        return -self.current_training_loss.item()

    @get_reward.register
    def _(self, reward_type: ValidationLoss):
        return -self.current_validation_loss.item()

    @get_reward.register
    def _(self, reward_type: LogTrainingLoss):
        return -torch.log(self.current_training_loss).item()

    @get_reward.register
    def _(self, reward_type: LogValidationLoss):
        return -torch.log(self.current_validation_loss).item()

    @get_reward.register
    def _(self, reward_type: LogDiffTraining):
        return -(
            torch.log(self.current_training_loss) - torch.log(self.prev_training_loss)
        ).item()

    @get_reward.register
    def _(self, reward_type: LogDiffValidation):
        return -(
            torch.log(self.current_validation_loss)
            - torch.log(self.prev_validation_loss)
        ).item()

    @get_reward.register
    def _(self, reward_type: DiffTraining):
        return (self.current_training_loss - self.prev_training_loss).item()

    @get_reward.register
    def _(self, reward_type: DiffValidation):
        return (self.current_validation_loss - self.prev_validation_loss).item()

    @get_reward.register
    def _(self, reward_type: FullTraining):
        return -self._get_full_training_loss().item()

    @property
    def crash(self):
        self.crashed = True
        return (
            self.get_state(self),
            self.config.crash_penalty,
            self.config.terminate_on_crash,
            {},
        )

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
        action : int
            action to execute

        Returns
        -------
        np.array, float, bool, dict
            state, reward, done, info
        """
        done = super(SGDEnv, self).step_()

        self.step_count += 1
        index = 0
        action = self.action_mapping[action]

        if isinstance(action, numbers.Number):
            action = [action]

        if any(torch.isnan(self.update_value)):
            return self.crash

        self.current_direction = -self.update_value / self.current_lr

        for idx, action_name in enumerate(self.action_names):
            for g in self.optimizer.param_groups:
                if np.isnan(action[idx]):
                    return self.crash
                g[action_name] = action[idx]
            if action_name == "lr":
                self.current_lr = torch.Tensor([action[idx]]).to(self.device)

        self.optimizer.step()
        self.optimizer.zero_grad()

        self.prev_training_loss = self.current_training_loss
        if self._current_validation_loss.calculated:
            self.prev_validation_loss = self.current_validation_loss

        self.fake_train()
        reward = self.get_reward(self.config.reward_type)
        if np.isnan(reward):
            return self.crash

        state = self.get_state(self)
        for value in state.values():
            if np.isnan(value):
                return self.crash

        return state, reward, done, {}

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

    def reset(self):
        """
        Reset environment

        Returns
        -------
        np.array
            Environment state
        """
        super(SGDEnv, self).reset_()

        dataset = self.instance[0]
        instance_seed = self.instance[1]
        construct_model = self._architecture_constructor(self.instance[2])
        self.n_steps = min(self.instance[3], self.config.cutoff)
        dataset_size = self.instance[4]
        self.step_count = 0

        self.seed(instance_seed)

        self.model = construct_model().to(self.device)
        self.val_model = construct_model().to(self.device)

        def init_weights(m):
            if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d:
                torch.nn.init.xavier_normal(m.weight)
                m.bias.data.fill_(0.0)

        if self.config.cd_paper_reconstruction:
            self.model.apply(init_weights)

        train_dataloader_args = {
            "batch_size": self.config.training_batch_size,
            "drop_last": True,
        }
        validation_dataloader_args = {
            "batch_size": self.config.validation_batch_size,
            "drop_last": True,
        }
        if self.use_cuda:
            param = {"num_workers": 1, "pin_memory": True}
            train_dataloader_args.update(param)
            validation_dataloader_args.update(param)

        if dataset == "MNIST":
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            )

            # hot fix for https://github.com/pytorch/vision/issues/3549
            # If fix is available in stable version (0.9.1), we should update and be removed this.
            new_mirror = "https://ossci-datasets.s3.amazonaws.com/mnist"
            datasets.MNIST.resources = [
                ("/".join([new_mirror, url.split("/")[-1]]), md5)
                for url, md5 in datasets.MNIST.resources
            ]

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

        self.model.train()
        self.val_model.eval()

        self.current_training_loss = None
        self.loss_batch = None

        self.optimizer = self.config.optimizer(
            self.model.parameters(),
            lr=self.initial_lr.item(),
            **self.config.optimizer_kwargs,
        )
        self.optimizer.zero_grad()

        self.current_lr = (
            self.initial_lr
        )  # TODO: Should not need these anymore if we use torch.optim as the initial LR is specified in config.optimizer_kwargs
        self.prev_direction = torch.zeros(
            (self.parameter_count,), device=self.device, requires_grad=False
        )
        self.current_direction = torch.zeros(
            (self.parameter_count,), device=self.device, requires_grad=False
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
        self.ema_loss_var = ExponentialMovingAverage(
            self.discount_factor, self.config.cd_bias_correction
        )
        self.ema_predictive_change = ExponentialMovingAverage(
            self.discount_factor, self.config.cd_bias_correction
        )

        self.fake_train()

        return self.get_state(self)

    def fake_train(self):
        loss = self.epoch()
        self.optimizer.zero_grad()
        with backpack(BatchGrad()):
            loss.mean().backward()

        with fake_step(self.optimizer, self.model):
            prev_values = torch.nn.utils.parameters_to_vector(self.model.parameters())
            self.optimizer.step()
            current_values = torch.nn.utils.parameters_to_vector(
                self.model.parameters()
            )
            self.update_value = current_values - prev_values

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
        if (
            "predictiveChange" in self.config.features
            or "predictiveChangeVarDiscountedAverage" in self.config.features
            or "predictiveChangeVarUncertainty" in self.config.features
        ):
            predictive_change = self._get_predictive_change_feature(
                self.current_lr
            ).item()
            (
                predictive_change_average,
                predictive_change_uncertainty,
            ) = self.ema_predictive_change.update(predictive_change)

        if (
            "lossVar" in self.config.features
            or "lossVarDiscountedAverage" in self.config.features
            or "lossVarUncertainty" in self.config.features
        ):
            loss_var = self._get_loss_feature().item()
            loss_var_average, loss_var_uncertainty = self.ema_loss_var.update(loss_var)

        if "alignment" in self.config.features:
            alignment = self._get_alignment()

        state = {}

        if "predictiveChange" in self.config.features:
            state["predictiveChange"] = predictive_change
        if "predictiveChangeVarDiscountedAverage" in self.config.features:
            state["predictiveChangeVarDiscountedAverage"] = predictive_change_average
        if "predictiveChangeVarUncertainty" in self.config.features:
            state["predictiveChangeVarUncertainty"] = predictive_change_uncertainty
        if "lossVar" in self.config.features:
            state["lossVar"] = loss_var
        if "lossVarDiscountedAverage" in self.config.features:
            state["lossVarDiscountedAverage"] = loss_var_average
        if "lossVarUncertainty" in self.config.features:
            state["lossVarUncertainty"] = loss_var_uncertainty
        if "currentLR" in self.config.features:
            state["currentLR"] = self.current_lr.item()
        if "trainingLoss" in self.config.features:
            if self.crashed:
                state["trainingLoss"] = 0.0
            else:
                state["trainingLoss"] = self.current_training_loss.item()
        if "validationLoss" in self.config.features:
            if self.crashed:
                state["validationLoss"] = 0.0
            else:
                state["validationLoss"] = self.current_validation_loss.item()
        if "step" in self.config.features:
            state["step"] = self.step_count
        if "alignment" in self.config.features:
            state["alignment"] = alignment.item()
        if "crashed" in self.config.features:
            state["crashed"] = 1 if self.crashed else 0

        state = {
            k: np.array([v], dtype=np.float32)
            for k, v in state.items()
            if k != "crashed"
        }
        if "crashed" in self.config.features:
            state["crashed"] = 1 if self.crashed else 0

        return state

    def _epoch(self):
        (data, target) = self.train_loader_it.next()
        data, target = data.to(self.device), target.to(self.device)
        self.current_batch_size = data.size()[0]
        output = self.model(data)
        loss = self.loss_function(output, target)

        loss_value = loss.mean()

        self.loss_batch = loss
        self.current_training_loss = torch.unsqueeze(loss_value.detach(), dim=0)
        self.train_batch_index += 1
        self._current_validation_loss.calculated = False

        return loss

    def epoch(self):
        try:
            loss = self._epoch()
        except StopIteration:
            self.train_batch_index = 0
            self.epoch_index += 1
            self.train_loader_it = iter(self.train_loader)
            loss = self._epoch()
        return loss

    def transfer_model_parameters(
        self,
    ):  # TODO: If this is only used in validation loss calculation you can probably hide it there.
        # self.val_model.load_state_dict(self.model.state_dict())
        for target_param, param in zip(
            self.val_model.parameters(), self.model.parameters()
        ):
            target_param.data.copy_(param.data)

    def _get_full_training_loss(self):
        self.transfer_model_parameters()
        # self.model.eval()
        loss = torch.zeros(1, device=self.device, requires_grad=False)
        with torch.no_grad():
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.val_model(data)
                loss += self.val_loss_function(output, target).sum().detach().detach()

        loss /= len(self.train_loader.dataset)
        # self.model.train()
        return loss

    def _get_full_validation_loss(self):
        self.transfer_model_parameters()
        # self.model.eval()
        validation_loss = torch.zeros(1, device=self.device, requires_grad=False)
        with torch.no_grad():
            for data, target in self.validation_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.val_model(data)
                validation_loss += self.val_loss_function(output, target).sum().detach()

        validation_loss /= len(self.validation_loader.dataset)
        # self.model.train()
        return -validation_loss.item()

    @property
    def current_validation_loss(self):
        if not self._current_validation_loss.calculated:
            self._current_validation_loss = self._get_validation_loss()
            self._current_validation_loss.calculated = True
        return self._current_validation_loss

    def _get_validation_loss_(self):
        with torch.no_grad():
            (data, target) = self.validation_loader_it.next()
            data, target = data.to(self.device), target.to(self.device)
            output = self.val_model(data)
            validation_loss = self.val_loss_function(output, target).mean()
            validation_loss = torch.unsqueeze(validation_loss.detach(), dim=0)

        return validation_loss

    def _get_validation_loss(self):
        self.transfer_model_parameters()  # TODO: I would probably just inline this function (with a comment explaining why it is needed)
        try:
            validation_loss = self._get_validation_loss_()
        except StopIteration:
            self.validation_loader_it = iter(self.validation_loader)
            validation_loss = self._get_validation_loss_()

        return validation_loss

    def _get_loss_feature(self):
        if self.crashed:
            return torch.tensor(0.0)
        bias_correction = (
            (1 - self.discount_factor ** (self.c_step + 1))
            if self.config.cd_bias_correction
            else 1
        )
        with torch.no_grad():
            loss_var = torch.log(torch.var(self.loss_batch))
        return loss_var

    def _get_predictive_change_feature(self, lr):
        if self.crashed:
            return torch.tensor(0.0)
        bias_correction = (
            (1 - self.discount_factor ** (self.c_step + 1))
            if self.config.cd_bias_correction
            else 1
        )
        batch_gradients = []
        for i, param in enumerate(self.model.parameters()):
            grad_batch = param.grad_batch.reshape(
                self.current_batch_size, self.layer_sizes[i]
            )
            batch_gradients.append(grad_batch)

        batch_gradients = torch.cat(batch_gradients, dim=1)
        predictive_change = torch.log(
            torch.var(-1 * torch.matmul(batch_gradients, self.update_value))
        )
        return predictive_change

    def _get_alignment(self):
        if self.crashed:
            return torch.tensor(0.0)
        a = torch.mul(self.prev_direction, self.current_direction)
        alignment = torch.mean(
            torch.sign(torch.mul(self.prev_direction, self.current_direction))
        )
        alignment = torch.unsqueeze(alignment, dim=0)
        self.prev_direction.data.copy_(self.current_direction)
        return alignment

    def generate_instance_file(self, file_name, mode="test", n=100):
        header = ["ID", "dataset", "architecture", "seed", "steps"]

        # dataset name, architecture, dataset size, sample dimension, number of max pool layers, hidden layers, test architecture convolutional layers
        architectures = [
            (
                "MNIST",
                "Conv2d(1, {0}, 3, 1, 1)-"
                "MaxPool2d(2, 2)-"
                "Conv2d({0}, {1}, 3, 1, 1)-"
                "MaxPool2d(2, 2)-"
                "Conv2d({1}, {2}, 3, 1, 1)-"
                "ReLU-"
                "Flatten-"
                "Linear({3}, 10)-"
                "LogSoftmax(1)",
                60000,
                28,
                2,
                3,
                [20, 50, 500],
            ),
            (
                "CIFAR",
                "Conv2d(3, {0}, 3, 1, 1)-"
                "MaxPool2d(2, 2)-"
                "ReLU-"
                "Conv2d({0}, {1}, 3, 1, 1)-"
                "ReLU-"
                "MaxPool2d(2, 2)-"
                "Conv2d({1}, {2}, 3, 1, 1)-"
                "ReLU-"
                "MaxPool2d(2, 2)-"
                "Conv2d({2}, {3}, 3, 1, 1)-"
                "ReLU-"
                "Flatten-"
                "Linear({4}, 10)-"
                "LogSoftmax(1)",
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
