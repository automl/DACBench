import math
import warnings
import json
from functools import reduce

import torch
from backpack import backpack, extend
from backpack.extensions import BatchGrad
from torchvision import datasets, transforms
from dacbench import AbstractEnv

warnings.filterwarnings("ignore")


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

        self.use_cuda = not self.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.training_validation_ratio = 0.8
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

        self.model = None

        self.parameter_count = 0
        self.layer_sizes = []

        self.loss_function = torch.nn.NLLLoss(reduction="none")
        self.loss_function = extend(self.loss_function)

        self.initial_lr = config.lr * torch.ones(
            1, device=self.device, requires_grad=False
        )
        self.current_lr = config.lr * torch.ones(
            1, device=self.device, requires_grad=False
        )

        # Adam parameters
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.m = 0
        self.v = 0
        self.epsilon = 1.0e-08
        self.t = 0
        self.step_count = torch.zeros(1, device=self.device, requires_grad=False)

        self.prev_direction = None
        self.current_direction = None

        self.learning_rate = 0.001
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
        self.discount_factor = 0.9
        self.firstOrderMomentum = torch.zeros(
            1, device=self.device, requires_grad=False
        )
        self.secondOrderMomentum = torch.zeros(
            1, device=self.device, requires_grad=False
        )

        self.writer = None

        if "reward_function" in config.keys():
            self.get_reward = config["reward_function"]
        else:
            self.get_reward = self.get_default_reward

        if "state_method" in config.keys():
            self.get_state = config["state_method"]
        else:
            self.get_state = self.get_default_state

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
        np.array, float, bool, dict
            state, reward, done, info
        """
        done = super(SGDEnv, self).step_()

        self.step_count += 1
        index = 0
        if not isinstance(action, float) and not isinstance(action, int):
            action = action[0]

        new_lr = torch.Tensor([action]).to(self.device)
        # new_lr = 10 ** (-action)
        self.current_lr = new_lr

        direction = self.firstOrderMomentum / (torch.sqrt(self.secondOrderMomentum) + self.epsilon)
        self.current_direction = direction
        delta_w = torch.mul(new_lr, direction)

        for i, p in enumerate(self.model.parameters()):
            layer_size = self.layer_sizes[i]
            p.data = p.data - delta_w[index : index + layer_size].reshape(
                shape=p.data.shape
            )
            index += layer_size

        self._set_zero_grad()
        reward = self.get_reward(self)
        return self.get_state(self), reward, done, {}

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

        self.seed(instance_seed)

        self.model = construct_model().to(self.device)

        self.training_validation_ratio = 0.8

        train_dataloader_args = {"batch_size": self.batch_size}
        validation_dataloader_args = {"batch_size": self.validation_batch_size}
        if self.use_cuda:
            param = {"num_workers": 1, "pin_memory": True, "shuffle": True}
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
        else:
            raise NotImplementedError

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

        self._set_zero_grad()
        self.model.train()

        self.current_training_loss = None
        self.loss_batch = None

        # Adam parameters
        self.m = 0
        self.v = 0
        self.t = 0
        self.step_count = torch.zeros(1, device=self.device, requires_grad=False)

        self.current_lr = self.initial_lr
        self.prev_direction = torch.zeros(
            (self.parameter_count,), device=self.device, requires_grad=False
        )
        self.current_direction = torch.zeros(
            (self.parameter_count,), device=self.device, requires_grad=False
        )
        self.get_default_reward(self)

        return self.get_state(self)

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
        gradients = self._get_gradients()
        self.firstOrderMomentum, self.secondOrderMomentum = self._get_momentum(
            gradients
        )

        if 'predictiveChangeVarDiscountedAverage' in self.on_features or 'predictiveChangeVarUncertainty' in self.on_features:
            predictiveChangeVarDiscountedAverage, predictiveChangeVarUncertainty = \
                self._get_predictive_change_features(self.current_lr, self.firstOrderMomentum, self.secondOrderMomentum)

        if 'lossVarDiscountedAverage' in self.on_features or 'lossVarUncertainty' in self.on_features:
            lossVarDiscountedAverage, lossVarUncertainty = self._get_loss_features()

        if 'alignment' in self.on_features:
            alignment = self._get_alignment()

        state = {}

        if 'predictiveChangeVarDiscountedAverage' in self.on_features:
            state["predictiveChangeVarDiscountedAverage"] = predictiveChangeVarDiscountedAverage
        if 'predictiveChangeVarUncertainty' in self.on_features:
            state["predictiveChangeVarUncertainty"] = predictiveChangeVarUncertainty
        if 'lossVarDiscountedAverage' in self.on_features:
            state["lossVarDiscountedAverage"] = lossVarDiscountedAverage
        if 'lossVarUncertainty' in self.on_features:
            state["lossVarUncertainty"] = lossVarUncertainty
        if 'currentLR' in self.on_features:
            state["currentLR"] = self.current_lr
        if 'trainingLoss' in self.on_features:
            state["trainingLoss"] = self.current_training_loss
        if 'validationLoss' in self.on_features:
            state["validationLoss"] = self.current_validation_loss
        if 'step' in self.on_features:
            state["step"] = self.step_count
        if 'alignment' in self.on_features:
            state["alignment"] = alignment

        return state

    def _set_zero_grad(self):
        index = 0
        for i, p in enumerate(self.model.parameters()):
            if p.grad is None:
                continue
            layer_size = self.layer_sizes[i]
            p.grad.zero_()
            index += layer_size

    def _train_batch_(self):
        (data, target) = self.train_loader_it.next()
        data, target = data.to(self.device), target.to(self.device)
        self.current_batch_size = data.size()[0]
        output = self.model(data)
        loss = self.loss_function(output, target)

        with backpack(BatchGrad()):
            loss.mean().backward()

        loss_value = loss.mean()
        reward = self._get_validation_loss()
        self.loss_batch = loss
        self.current_training_loss = torch.unsqueeze(loss_value.detach(), dim=0)
        self.train_batch_index += 1

        return reward

    def get_default_reward(self, _):
        try:
            reward = self._train_batch_()
        except StopIteration:
            self.train_batch_index = 0
            self.epoch_index += 1
            self.train_loader_it = iter(self.train_loader)
            reward = self._train_batch_()

        return reward

    def _get_val_loss(self):
        self.model.eval()
        validation_loss = torch.zeros(1, device=self.device, requires_grad=False)
        with torch.no_grad():
            for data, target in self.validation_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                validation_loss += self.loss_function(output, target).mean()

        validation_loss /= len(self.validation_loader.dataset)
        self.model.train()
        return validation_loss

    def _get_validation_loss_(self):
        self.model.eval()
        (data, target) = self.validation_loader_it.next()
        data, target = data.to(self.device), target.to(self.device)
        output = self.model(data)
        validation_loss = self.loss_function(output, target).mean()
        validation_loss = torch.unsqueeze(validation_loss.detach(), dim=0)
        self.current_validation_loss = validation_loss
        self.model.train()

        return -validation_loss.item()  # negative because it is the reward

    def _get_validation_loss(self):
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
        bias_corrected_m = self.m / (1 - self.beta1 ** self.t)
        bias_corrected_v = self.v / (1 - self.beta2 ** self.t)

        return bias_corrected_m, bias_corrected_v

    def _get_adam_feature(self, learning_rate, m, v):
        epsilon = 1.0e-8
        return torch.mul(learning_rate, m / (torch.sqrt(v) + epsilon))

    def _get_loss_features(self):
        with torch.no_grad():
            loss_var = torch.log(torch.var(self.loss_batch))

            self.lossVarDiscountedAverage = (
                self.discount_factor * self.lossVarDiscountedAverage
                + (1 - self.discount_factor) * loss_var
            )
            self.lossVarUncertainty = (
                self.discount_factor * self.lossVarUncertainty
                + (1 - self.discount_factor)
                * (loss_var - self.lossVarDiscountedAverage) ** 2
            )

        return self.lossVarDiscountedAverage, self.lossVarUncertainty

    def _get_predictive_change_features(self, lr, m, v):
        batch_gradients = []
        for i, (name, param) in enumerate(self.model.named_parameters()):
            grad_batch = param.grad_batch.reshape(
                self.current_batch_size, self.layer_sizes[i]
            )
            batch_gradients.append(grad_batch)

        batch_gradients = torch.cat(batch_gradients, dim=1)

        update_value = self._get_adam_feature(lr, m, v)
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
            * (predictive_change - self.predictiveChangeVarDiscountedAverage) ** 2
        )

        return (
            self.predictiveChangeVarDiscountedAverage,
            self.predictiveChangeVarUncertainty,
        )

def _get_alignment(self):
    alignment = torch.mean(torch.sign(torch.mul(self.prev_direction, self.current_direction)))
    alignment = torch.unsqueeze(alignment, dim=0)
    self.prev_direction = self.current_direction
    return alignment
