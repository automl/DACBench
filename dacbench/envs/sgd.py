import math
import numbers
import warnings
import json
from functools import reduce
from enum import IntEnum, auto

import torch
from backpack import backpack, extend
from backpack.extensions import BatchGrad
from numpy import float32
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
        if hasattr(self, 'func'):
            raise ValueError('Can not assign the same reward to a different function!')
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

        self.use_cuda = not self.no_cuda and torch.cuda.is_available()
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
        self.current_validation_loss = torch.zeros(
            1, device=self.device, requires_grad=False
        )
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
        self.parameter_count = 0  # TODO: Verify that we still need this if we use pytorch.optim
        self.layer_sizes = []  # TODO: Verify that we still need this if we use pytorch.optim

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

        self.optimizer_name = config.optimizer

        # TODO: Make this part of the config (optimizer_kwargs)
        # TODO: m, v and t should not be stored here but in the param_group of the optimizer (if Adam is used)
        # Adam parameters
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.m = 0
        self.v = 0
        self.epsilon = 1.0e-06
        # RMSprop parameters
        self.beta1 = config.beta1
        self.v = 0
        # Momentum parameters
        self.sgd_momentum_v = 0
        self.sgd_rho = 0.9

        self.t = 0
        self.step_count = torch.zeros(1, device=self.device, requires_grad=False)

        self.prev_direction = None
        self.current_direction = None

        self.learning_rate = 0.001  # TODO: Yet another lr? Is this used?
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
        self.discount_factor = 0.9  # TODO: Make this part of the config
        self.firstOrderMomentum = torch.zeros(
            1, device=self.device, requires_grad=False
        )
        self.secondOrderMomentum = torch.zeros(
            1, device=self.device, requires_grad=False
        )

        self.writer = None  # TODO: Is this used?

        if self.optimizer_name=="adam":
            self.get_optimizer_direction = self.get_adam_direction
        elif self.optimizer_name=="rmsprop":
            self.get_optimizer_direction = self.get_rmsprop_direction
        elif self.optimizer_name=="momentum":
            self.get_optimizer_direction = self.get_momentum_direction
        else:
            raise NotImplementedError

        if "reward_function" in config.keys():
            self._get_reward = config["reward_function"]
        else:
            self._get_reward = config.reward_type.func

        if "state_method" in config.keys():
            self.get_state = config["state_method"]
        else:
            self.get_state = self.get_default_state

        self.reward_range = config.reward_type.func.frange

    def get_reward(self):
        return self._get_reward(self)

    @reward_range([-(10**9), 0])
    @Reward.TrainingLoss
    def get_training_reward(self):
        return -self.current_training_loss.item()

    @reward_range([-(10**9), 0])
    @Reward.ValidationLoss
    def get_validation_reward(self):
        return -self._get_validation_loss().item()

    @reward_range([-(10**9), (10**9)])
    @Reward.LogTrainingLoss
    def get_log_training_reward(self):
        return -torch.log(self.current_training_loss).item()

    @reward_range([-(10**9), (10**9)])
    @Reward.LogValidationLoss
    def get_log_validation_reward(self):
        return -torch.log(self._get_validation_loss()).item()

    @reward_range([-(10**9), (10**9)])
    @Reward.LogDiffTraining
    def get_log_diff_training_reward(self):
        return -(torch.log(self.current_training_loss) - torch.log(self.prev_training_loss)).item()

    @reward_range([-(10**9), (10**9)])
    @Reward.LogDiffValidation
    def get_log_diff_validation_reward(self):
        return -(torch.log(self._get_validation_loss()) - torch.log(self.prev_validation_loss)).item()

    @reward_range([-(10**9), (10**9)])
    @Reward.DiffTraining
    def get_diff_training_reward(self):
        return (self.current_training_loss - self.prev_training_loss).item()

    @reward_range([-(10**9), (10**9)])
    @Reward.DiffValidation
    def get_diff_validation_reward(self):
        return (self._get_validation_loss() - self.prev_validation_loss).item()

    @reward_range([-(10**9), 0])
    @Reward.FullTraining
    def get_full_training_reward(self):
        return -self._get_full_training_loss().item()

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

        if not isinstance(action, int) and not isinstance(action, float):
            action = action.item()
        if not isinstance(action, numbers.Number):
            action = action[0]

        new_lr = torch.Tensor([action]).to(self.device)
        self.current_lr = new_lr

        # TODO: (BEGIN) This update should be done by self.optimizer.step()
        direction = self.get_optimizer_direction()

        self.current_direction = direction  # TODO: See note below this todo

        delta_w = torch.mul(new_lr, direction)

        for i, p in enumerate(self.model.parameters()):
            layer_size = self.layer_sizes[i]
            p.data = p.data - delta_w[index : index + layer_size].reshape(
                shape=p.data.shape
            )
            index += layer_size

        # TODO: (END)
        # Note: Computing directions becomes more difficult and involves comparing parameters before/after the update
        # You will also need similar calculations to calculate the predictiveChange features, so probably best to write
        # this as a separate function (in fact, for every optimizer.step() you need the (direction of the) update vector
        # (in predictiveChange you must store the state before / restore it after to reverse the step, more info below)

        self._set_zero_grad()  # TODO: This could also be done by a call to self.optimizer.zero_grad?
        # TODO: Seperate the forward/backward pass on train from the caclulation of the reward (forward on val) so the following this can be (roughly) be rewritten as:
        # 1) self.compute_forward_backward()
        # 2) return self.get_state(self), self.get_reward(self), done, {}
        self.train_network()
        reward = self.get_reward()

        self.prev_training_loss = self.current_training_loss
        self.prev_validation_loss = self.current_validation_loss

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
        self.val_model = construct_model().to(self.device)

        def init_weights(m):
            if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d:
                torch.nn.init.xavier_normal(m.weight)
                m.bias.data.fill_(0.0)

        if self.cd_paper_reconstruction:
            self.model.apply(init_weights)

        train_dataloader_args = {"batch_size": self.batch_size}
        validation_dataloader_args = {"batch_size": self.validation_batch_size}
        if self.use_cuda:
            param = {"num_workers": 1, "pin_memory": True, "shuffle": True}
            train_dataloader_args.update(param)
            validation_dataloader_args.update(param)

        if dataset == "MNIST":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

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
        elif dataset == "MNISTsmall":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

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
            train_dataset = torch.utils.data.Subset(
                train_dataset, range(0, len(train_dataset) // 2)
            )
            # self.test_dataset = datasets.MNIST('../data', train=False, transform=transform)
        elif dataset == "CIFAR":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            train_dataset = datasets.CIFAR10(
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

        # TODO: Somewhere here self.optimizer should be initialised based on the benchmark config
        # config.optimizer_class: Specifying a pytorch.optim classname, e.g. 'Adam'
        # config.optimizer_kwargs: Specifying a dict of optimizer arguments, e.g. {lr: 0.01, betas=(0.5,0.5), eps=0.0000001}
        # i.e. ~ self.optimizer_class(self.model.parameters(), **self.optimizer_kwargs)

        self._set_zero_grad()  # TODO: call to self.optimizer
        self.model.train()
        self.val_model.eval()

        self.current_training_loss = None
        self.loss_batch = None

        # Adam parameters
        self.m = 0 # TODO: Should not need these anymore if we use torch.optim
        self.v = 0 # TODO: Should not need these anymore if we use torch.optim
        # RMSprop parameters
        self.v = 0
        # Momentum parameters
        self.sgd_momentum_v = 0

        self.t = 0 # TODO: Should not need these anymore if we use torch.optim

        self.step_count = torch.zeros(1, device=self.device, requires_grad=False)

        self.current_lr = self.initial_lr  # TODO: Should not need these anymore if we use torch.optim as the initial LR is specified in config.optimizer_kwargs
        self.prev_direction = torch.zeros(
            (self.parameter_count,), device=self.device, requires_grad=False
        )
        self.current_direction = torch.zeros(
            (self.parameter_count,), device=self.device, requires_grad=False
        )
        self.train_network()

        self.prev_training_loss = self.current_training_loss

        self.get_reward()

        self.prev_validation_loss = self.current_validation_loss

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

        self.gradients = self._get_gradients() # TODO: Should not need these anymore if we use torch.optim
        self.firstOrderMomentum, self.secondOrderMomentum, self.sgdMomentum = self._get_momentum(self.gradients) # TODO: Should not need these anymore if we use torch.optim

        if 'predictiveChangeVarDiscountedAverage' in self.on_features or 'predictiveChangeVarUncertainty' in self.on_features:
            predictiveChangeVarDiscountedAverage, predictiveChangeVarUncertainty = \
                self._get_predictive_change_features(self.current_lr)

        if 'lossVarDiscountedAverage' in self.on_features or 'lossVarUncertainty' in self.on_features:
            lossVarDiscountedAverage, lossVarUncertainty = self._get_loss_features()

        if 'alignment' in self.on_features:
            alignment = self._get_alignment()

        state = {}

        if 'predictiveChangeVarDiscountedAverage' in self.on_features:
            state["predictiveChangeVarDiscountedAverage"] = predictiveChangeVarDiscountedAverage.item()
        if 'predictiveChangeVarUncertainty' in self.on_features:
            state["predictiveChangeVarUncertainty"] = predictiveChangeVarUncertainty.item()
        if 'lossVarDiscountedAverage' in self.on_features:
            state["lossVarDiscountedAverage"] = lossVarDiscountedAverage.item()
        if 'lossVarUncertainty' in self.on_features:
            state["lossVarUncertainty"] = lossVarUncertainty.item()
        if 'currentLR' in self.on_features:
            state["currentLR"] = self.current_lr.item()
        if 'trainingLoss' in self.on_features:
            state["trainingLoss"] = self.current_training_loss.item()
        if 'validationLoss' in self.on_features:
            state["validationLoss"] = self.current_validation_loss.item()
        if 'step' in self.on_features:
            state["step"] = self.step_count.item()
        if 'alignment' in self.on_features:
            state["alignment"] = alignment.item()

        return state

    def _set_zero_grad(self):
        # TODO: I think this can be replaced by a self.optimizer call
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

        self.loss_batch = loss
        self.current_training_loss = torch.unsqueeze(loss_value.detach(), dim=0)
        self.train_batch_index += 1

    def train_network(self):
        try:
            self._train_batch_()
        except StopIteration:
            self.train_batch_index = 0
            self.epoch_index += 1
            self.train_loader_it = iter(self.train_loader)
            self._train_batch_()

    def transfer_model_parameters(self):  # TODO: If this is only used in validation loss calculation you can probably hide it there.
        # self.val_model.load_state_dict(self.model.state_dict())
        for target_param, param in zip(self.val_model.parameters(), self.model.parameters()):
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

    def _get_validation_loss_(self):
        # self.model.eval()
        (data, target) = self.validation_loader_it.next()
        data, target = data.to(self.device), target.to(self.device)
        output = self.val_model(data)
        validation_loss = self.val_loss_function(output, target).mean()
        validation_loss = torch.unsqueeze(validation_loss.detach(), dim=0)
        self.current_validation_loss = validation_loss
        # self.model.train()

        return validation_loss

    def _get_validation_loss(self):
        self.transfer_model_parameters()  # TODO: I would probably just inline this function (with a comment explaining why it is needed)
        try:
            validation_loss = self._get_validation_loss_()
        except StopIteration:
            self.validation_loader_it = iter(self.validation_loader)
            validation_loss = self._get_validation_loss_()

        return validation_loss

    def _get_gradients(self):  # TODO: Not needed when using pytorch.optim?
        gradients = []
        for p in self.model.parameters():
            if p.grad is None:
                continue
            gradients.append(p.grad.flatten())

        gradients = torch.cat(gradients, dim=0)

        return gradients

    def _get_momentum(self, gradients):  # TODO: Not needed when using pytorch.optim?
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        self.v = self.beta2 * self.v + (1 - self.beta2) * torch.square(gradients)
        bias_corrected_m = self.m / (1 - self.beta1 ** self.t)
        bias_corrected_v = self.v / (1 - self.beta2 ** self.t)

        self.sgd_momentum_v = self.sgd_rho * self.sgd_momentum_v + gradients

        return bias_corrected_m, bias_corrected_v, self.sgd_momentum_v

    def get_adam_direction(self): # TODO: Not needed when using pytorch.optim?
        return self.firstOrderMomentum / (torch.sqrt(self.secondOrderMomentum) + self.epsilon)

    def get_rmsprop_direction(self):
        return self.gradients / (torch.sqrt(self.secondOrderMomentum) + self.epsilon)

    def get_momentum_direction(self):
        return self.sgd_momentum_v

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

    def _get_predictive_change_features(self, lr):
        # TODO: This must be done differently/more generically when using pytorch.optim.
        # A costly but general way would
        # 1) store the full state of the model and optimizer param_groups (storing things like m, v), etc.
        # 2) perform a step to determine the update_value
        # 3) restore the full state from (1)
        # Here we have to take care that the performing 1+2+3 does not affect the (future) optimisation!
        # In particular, when using a static lr, the trajectory should be exactly the same with/without 1+2+3
        # Note: This is the way suggested here: https://discuss.pytorch.org/t/revert-optimizer-step/70692/6
        # Note: that you also need (2) for the gradient direction every step so best implement this in a separate function
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
