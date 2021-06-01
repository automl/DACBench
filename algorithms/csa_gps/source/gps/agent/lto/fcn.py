import sys
import os
import numpy as np
import tensorflow as tf
import pickle
from time import time


def printWithoutNewline(s):
    sys.stdout.write(s)
    sys.stdout.flush()


# A FcnFamily is a function template with unrealized placeholders (e.g. coefficients)
# A Fcn is a member of a FcnFamily with actual values substituted in for the placeholders

# For input to the functions "evaluate", "grad", "hess", x can be a list of variables, but each variable must be an N x 1 vector
# fcn must be a function that takes two arguments, x and params. x is a list of variables, and params is a dict, with the keys corresponding to names of placeholders and values being the substituted values
class FcnFamily(object):

    # params is a dict whose entries are (name, type)
    # hyperparams is a dict and must be the SAME as the parameters passed into the constructor of the child class - it is used for pickling
    # Options can be passed in as extra keyword arguments. Available options: disabled_hess, session, start_session_manually, gpu_id, tensor_prefix
    # Options that are for internal use only: graph_def and tensor_names - these are used when unpickling
    def __init__(self, fcn, num_dims, params, hyperparams, **kwargs):
        self.num_dims = num_dims
        self.fcn_defns = fcn
        self.param_defns = params
        self.hyperparams = hyperparams
        self.options = kwargs
        self.session = None
        self.options["disable_hess"] = True
        if "session" in self.options:
            session = self.options["session"]
            del self.options["session"]
            if (
                "start_session_manually" in self.options
                and self.options["start_session_manually"]
            ):
                print(
                    "Warning: start_session_manually is set to True even though session is passed in. Starting session anyway. "
                )
            self.start_session(session)
        else:
            if (
                "start_session_manually" not in self.options
                or not self.options["start_session_manually"]
            ):
                self.start_session()
        if "start_session_manually" in self.options:
            del self.options["start_session_manually"]

    def start_session(self, session=None):
        def construct_graph():
            if "graph_def" in self.options:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(self.options["graph_def"])
                del self.options["graph_def"]
                self.session.graph.as_default()
                tf.import_graph_def(graph_def, name="")

                if "tensor_names" in self.options:
                    tensor_names = self.options["tensor_names"]
                else:
                    prefix = (
                        "%s_" % (self.options["tensor_prefix"])
                        if "tensor_prefix" in self.options
                        else ""
                    )
                    tensor_names = dict()
                    tensor_names["params"] = {
                        param_name: "%sparam_%s:0" % (prefix, param_name)
                        for param_name in self.param_defns
                    }
                    tensor_names["x"] = [
                        "%sx_%d:0" % (prefix, i) for i in range(len(self.num_dims))
                    ]
                    tensor_names["fcn"] = "%sfcn:0" % (prefix)
                    tensor_names["grad"] = [
                        "%sgrad_%d:0" % (prefix, i) for i in range(len(self.num_dims))
                    ]
                    if ("disable_hess" not in self.options) or (
                        not self.options["disable_hess"]
                    ):
                        tensor_names["hess"] = [
                            ["%shess_%d_%d:0" % (prefix, i, j) for j in range(i + 1)]
                            for i in range(len(self.num_dims))
                        ]

                for param_name in self.param_defns:
                    self.params[param_name] = self.session.graph.get_tensor_by_name(
                        tensor_names["params"][param_name]
                    )
                    self.is_param_subsampled[param_name] = (
                        "subsampled" in self.param_defns[param_name]
                        and self.param_defns[param_name]["subsampled"]
                    )

                self.x_ = [
                    self.session.graph.get_tensor_by_name(tensor_names["x"][i])
                    for i in range(len(self.num_dims))
                ]  # A list of variable groups
                self.fcn_ = self.session.graph.get_tensor_by_name(tensor_names["fcn"])
                self.grad_ = [
                    self.session.graph.get_tensor_by_name(tensor_names["grad"][i])
                    for i in range(len(self.num_dims))
                ]

                prefix = (
                    "%s_" % (self.options["tensor_prefix"])
                    if "tensor_prefix" in self.options
                    else ""
                )
                if ("disable_hess" not in self.options) or (
                    not self.options["disable_hess"]
                ):
                    self.hess_ = []
                    for i in range(
                        len(self.num_dims)
                    ):  # Iterate over each variable group
                        #    block_cols_of_block_cells = [self.session.graph.get_tensor_by_name(tensor_names["hess"][i][j]) for j in range(i+1)]
                        #    self.hess_.append(block_cols_of_block_cells)
                        # Each element is an individual row
                        rows_of_block_cols = [
                            tf.gradients(self.grad_[i][k, :], self.x_[: i + 1])
                            for k in range(self.num_dims[i])
                        ]
                        # Each element is a block column
                        block_cols_of_block_cells = [
                            tf.transpose(
                                tf.concat([row[j] for row in rows_of_block_cols], 1),
                                name="%shess_%d_%d" % (prefix, i, j),
                            )
                            for j in range(i + 1)
                        ]
                        self.hess_.append(block_cols_of_block_cells)
            else:
                prefix = (
                    "%s_" % (self.options["tensor_prefix"])
                    if "tensor_prefix" in self.options
                    else ""
                )

                for param_name in self.param_defns:
                    self.params[param_name] = tf.placeholder(
                        self.param_defns[param_name]["type"],
                        name="%sparam_%s" % (prefix, param_name),
                    )
                    self.is_param_subsampled[param_name] = (
                        "subsampled" in self.param_defns[param_name]
                        and self.param_defns[param_name]["subsampled"]
                    )

                self.x_ = [
                    tf.placeholder(tf.float64, name="%sx_%d" % (prefix, i))
                    for i in range(len(self.num_dims))
                ]  # A list of variable groups
                fcn = self.fcn_defns(
                    self.x_, self.params
                )  # May return a tuple of functions - assume the first one is the main function which we will be differentiating
                self.fcn_ = tf.identity(fcn, name="%sfcn" % (prefix))
                self.grad_ = [
                    tf.identity(cur_grad, name="%sgrad_%d" % (prefix, i))
                    for i, cur_grad in enumerate(tf.gradients(self.fcn_, self.x_))
                ]  # A list of gradient expressions wrt each variable group, each of which is a vector

                if ("disable_hess" not in self.options) or (
                    not self.options["disable_hess"]
                ):
                    self.hess_ = []
                    for i in range(
                        len(self.num_dims)
                    ):  # Iterate over each variable group
                        # Each element is an individual row
                        rows_of_block_cols = [
                            tf.gradients(self.grad_[i][k, :], self.x_[: i + 1])
                            for k in range(self.num_dims[i])
                        ]
                        # Each element is a block column
                        block_cols_of_block_cells = [
                            tf.transpose(
                                tf.concat([row[j] for row in rows_of_block_cols], 1),
                                name="%shess_%d_%d" % (prefix, i, j),
                            )
                            for j in range(i + 1)
                        ]
                        self.hess_.append(block_cols_of_block_cells)

        if self.session is not None:
            if session is not None and self.session != session:
                print(
                    "Warning: start_session is called with a different session than the one in use. Will keep using existing session. "
                )
        else:
            if session is None:
                self.session = tf.Session()
            else:
                self.session = session

            self.params = {}
            self.is_param_subsampled = {}

            self.device_string = "/cpu:0"
            if "gpu_id" in self.options:
                self.device_string = "/gpu:%d" % (self.options["gpu_id"])

            if self.device_string == "/cpu:0":
                with tf.device(self.device_string):
                    construct_graph()
            else:
                construct_graph()

    def assign_param_vals_(self, param_vals):
        placeholder_vals = {}
        for key in self.params:
            placeholder_vals[self.params[key]] = param_vals[key]
        return placeholder_vals

    def evaluate(self, x, param_vals):
        assert self.session is not None, "start_session() must be called first. "
        placeholder_vals = {self.x_[i]: x[i] for i in range(len(self.x_))}
        placeholder_vals.update(self.assign_param_vals_(param_vals))
        with tf.device(self.device_string):
            val = self.session.run(self.fcn_, placeholder_vals)
        return val

    def grad(self, x, param_vals):
        assert self.session is not None, "start_session() must be called first. "
        placeholder_vals = {self.x_[i]: x[i] for i in range(len(self.x_))}
        placeholder_vals.update(self.assign_param_vals_(param_vals))
        with tf.device(self.device_string):
            vals = self.session.run(self.grad_, placeholder_vals)
        return vals

    # Returns a list of lists, with vals[i][j] containing the second derivative wrt self.x_[i] and self.x_[j]
    def hess(self, x, param_vals):
        assert ("disable_hess" not in self.options) or (
            not self.options["disable_hess"]
        ), "Hessian is disabled. "
        assert self.session is not None, "start_session() must be called first. "

        placeholder_vals = {self.x_[i]: x[i] for i in range(len(self.x_))}
        placeholder_vals.update(self.assign_param_vals_(param_vals))
        with tf.device(self.device_string):
            flattened_vals = self.session.run(
                [hess_elem for hess_list in self.hess_ for hess_elem in hess_list],
                placeholder_vals,
            )
        vals = []
        j = 0
        for i in range(len(self.x_)):
            vals.append(flattened_vals[j : j + (i + 1)])
            vals[-1].extend([None] * (len(self.x_) - i - 1))
            j += i + 1
        # Fill in the upper triangle of the Hessian by taking advantage of the symmetry of the Hessian
        for i in range(1, len(self.x_)):
            for j in range(i):
                vals[j][i] = vals[i][j].T
        return vals

    def get_total_num_dim(self):
        total_num_dim = 0
        for num_dim in self.num_dims:
            total_num_dim += num_dim
        return total_num_dim

    def destroy(self):
        if self.session is not None:
            self.session.close()
            self.session = None

    # For pickling
    def __getstate__(self):
        if self.session is None:
            print(
                "Warning: Session automatically started for the purposes of pickling. "
            )
            self.start_session()
        tf.train.write_graph(
            self.session.graph_def, "/tmp", "tf_graph.pb", False
        )  # proto
        with open("/tmp/tf_graph.pb", "rb") as f:
            graph_def_str = f.read()
        os.remove("/tmp/tf_graph.pb")

        tensor_names = dict()
        tensor_names["params"] = {
            param_name: self.params[param_name].name for param_name in self.params
        }
        tensor_names["x"] = [cur_x.name for cur_x in self.x_]
        tensor_names["fcn"] = self.fcn_.name
        tensor_names["grad"] = [cur_grad.name for cur_grad in self.grad_]

        if ("disable_hess" not in self.options) or (not self.options["disable_hess"]):
            tensor_names["hess"] = [
                [cur_hess_block.name for cur_hess_block in cur_hess_block_row]
                for cur_hess_block_row in self.hess_
            ]

        return {
            "hyperparams": self.hyperparams,
            "options": {
                option_name: self.options[option_name]
                for option_name in self.options
                if option_name not in ["session", "graph_def", "start_session_manually"]
            },
            "graph_def": graph_def_str,
            "tensor_names": tensor_names,
        }

    # For unpickling
    def __setstate__(self, state):
        kwargs = state["hyperparams"].copy()
        kwargs.update(state["options"])
        kwargs["graph_def"] = state["graph_def"]
        kwargs["tensor_names"] = state["tensor_names"]
        kwargs["start_session_manually"] = True
        self.__init__(**kwargs)


class Fcn(object):

    # If disable_subsampling is set to True, will never subsample regardless of what batch_size is set to be, either in the constructor or in Fcn.new_sample()
    def __init__(self, family, param_vals, batch_size="all", disable_subsampling=False):
        self.family = family
        self.param_vals = param_vals
        self.batch_size = batch_size
        self.disable_subsampling = disable_subsampling or all(
            not self.family.is_param_subsampled[key] for key in self.family.params
        )
        # If batch size is "all" or no params are subsampled, don't require calling Fcn.new_sample() before calling Fcn.evaluate/grad/hess.
        if self.disable_subsampling or batch_size == "all":
            self.subsampled_param_vals = self.param_vals
        else:
            self.subsampled_param_vals = None

    # If self.disable_subsampling is True, this is a no-op.
    # If batch_size is set, temporarily overrides self.batch_size
    # By setting batch_size to "all", can temporarily disable subsampling
    def new_sample(self, batch_size=None):
        if not self.disable_subsampling:
            if batch_size is None:
                batch_size = self.batch_size
            if batch_size != "all":
                subsampled_idx = None  # Same sampled indices are used for all params to preserve correspondence between individual entries (i.e. one row of data corresponds to one element of label)
                self.subsampled_param_vals = {}
                for key in self.family.params:
                    if (
                        not self.family.is_param_subsampled[key]
                        or batch_size >= self.param_vals[key].shape[0]
                    ):
                        self.subsampled_param_vals[key] = self.param_vals[key]
                    else:
                        if subsampled_idx is None:
                            subsampled_idx = np.random.permutation(
                                self.param_vals[key].shape[0]
                            )[:batch_size]
                        self.subsampled_param_vals[key] = self.param_vals[key][
                            subsampled_idx
                        ]
            else:
                self.subsampled_param_vals = self.param_vals

    def evaluate(self, x):
        assert (
            self.subsampled_param_vals is not None
        ), "Fcn.new_sample() must be called first. "
        return self.family.evaluate(x, self.subsampled_param_vals)

    def grad(self, x):
        assert (
            self.subsampled_param_vals is not None
        ), "Fcn.new_sample() must be called first. "
        return self.family.grad(x, self.subsampled_param_vals)

    def hess(self, x):
        assert (
            self.subsampled_param_vals is not None
        ), "Fcn.new_sample() must be called first. "
        return self.family.hess(x, self.subsampled_param_vals)


class QuadFormFcnFamily(FcnFamily):
    def __init__(self, num_dim, **kwargs):
        def fcn(x, params):
            return tf.matmul(x[0], tf.matmul(params["A"], x[0]), transpose_a=True)

        FcnFamily.__init__(
            self,
            fcn,
            [num_dim],
            {"A": {"type": tf.float64}},
            {"num_dim": num_dim},
            **kwargs,
        )


class QuadFormFcn(Fcn):
    def __init__(self, family, A, *args, **kwargs):
        Fcn.__init__(self, family, {"A": A}, *args, **kwargs)

    def evaluate(self, x):
        return Fcn.evaluate(self, x)

    def grad(self, x):
        return Fcn.grad(self, x)[0]

    def hess(self, x):
        return Fcn.hess(self, [x])[0][0]


class LogisticRegressionFcnFamily(FcnFamily):
    def __init__(self, dim, **kwargs):
        # params["data"] is N x dim, params["labels"] is N x 1
        # params["sigma_sq"] is 1 x 1 and represents the squared of the sigma parameter in the GM estimator
        # The larger sigma is, the larger the non-saturating range
        def fcn(x, params):
            weights = tf.slice(x[0], [0, 0], [dim, -1])  # dim x 1 matrix
            bias = tf.slice(x[0], [dim, 0], [1, -1])  # 1 x 1 matrix

            preds = (
                tf.matmul(params["data"], weights) + bias
            )  # N x 1 matrix, where N is the number of data points

            loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=preds, labels=params["labels"]
                )
            )

            # L2 regularization for the fully connected parameters.
            regularizers = tf.nn.l2_loss(weights)
            # Add the regularization term to the loss.
            loss += 5e-4 * regularizers

            return loss

        FcnFamily.__init__(
            self,
            fcn,
            [dim + 1],
            {
                "data": {"type": tf.float64, "subsampled": True},
                "labels": {"type": tf.float64, "subsampled": True},
            },
            {"dim": dim},
            **kwargs,
        )


class LogisticRegressionFcn(Fcn):
    def __init__(self, family, data, labels, *args, **kwargs):
        Fcn.__init__(self, family, {"data": data, "labels": labels}, *args, **kwargs)

    def evaluate(self, x):
        return Fcn.evaluate(self, x)

    def grad(self, x):
        return Fcn.grad(self, x)[0]

    def hess(self, x):
        return Fcn.hess(self, [x])[0][0]


class LogisticRegressionWithoutBiasFcnFamily(FcnFamily):
    def __init__(self, dim, **kwargs):
        # params["data"] is N x dim, params["labels"] is N x 1
        def fcn(x, params):
            weights = x[0]

            preds = tf.matmul(
                params["data"], weights
            )  # N x 1 matrix, where N is the number of data points

            loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(preds, params["labels"])
            )

            # L2 regularization for the fully connected parameters.
            regularizers = tf.nn.l2_loss(weights)
            # Add the regularization term to the loss.
            loss += 5e-4 * regularizers

            return loss

        FcnFamily.__init__(
            self,
            fcn,
            [dim],
            {
                "data": {"type": tf.float64, "subsampled": True},
                "labels": {"type": tf.float64, "subsampled": True},
            },
            {"dim": dim},
            **kwargs,
        )


class LogisticRegressionWithoutBiasFcn(Fcn):
    def __init__(self, family, data, labels, *args, **kwargs):
        Fcn.__init__(self, family, {"data": data, "labels": labels}, *args, **kwargs)

    def evaluate(self, x):
        return Fcn.evaluate(self, x)

    def grad(self, x):
        return Fcn.grad(self, x)[0]

    def hess(self, x):
        return Fcn.hess(self, [x])[0][0]


# Robust linear regresison using Geman-McLure (GM) estimator
class RobustRegressionFcnFamily(FcnFamily):
    def __init__(self, dim, **kwargs):
        # params["data"] is N x dim, params["labels"] is N x 1
        # params["sigma_sq"] is 1 x 1 and represents the squared of the sigma parameter in the GM estimator
        # The larger sigma is, the larger the non-saturating range
        def fcn(x, params):
            weights = tf.slice(x[0], [0, 0], [dim, -1])  # dim x 1 matrix
            bias = tf.slice(x[0], [dim, 0], [1, -1])  # 1 x 1 matrix

            preds = (
                tf.matmul(params["data"], weights) + bias
            )  # N x 1 matrix, where N is the number of data points
            err = params["labels"] - preds
            err_sq = tf.square(err)
            loss = tf.reduce_mean(
                tf.truediv(err_sq, tf.add(err_sq, params["sigma_sq"]))
            )

            return loss

        FcnFamily.__init__(
            self,
            fcn,
            [dim + 1],
            {
                "data": {"type": tf.float64, "subsampled": True},
                "labels": {"type": tf.float64, "subsampled": True},
                "sigma_sq": {"type": tf.float64},
            },
            {"dim": dim},
            **kwargs,
        )


class RobustRegressionFcn(Fcn):
    def __init__(self, family, data, labels, sigma_sq, *args, **kwargs):
        Fcn.__init__(
            self,
            family,
            {"data": data, "labels": labels, "sigma_sq": sigma_sq},
            *args,
            **kwargs,
        )

    def evaluate(self, x):
        return Fcn.evaluate(self, x)

    def grad(self, x):
        return Fcn.grad(self, x)[0]

    def hess(self, x):
        return Fcn.hess(self, [x])[0][0]


class NeuralNetFcnFamily(FcnFamily):
    def __init__(self, input_dim, hidden_dim, output_dim, **kwargs):
        if not isinstance(hidden_dim, list):
            hidden_dim = [hidden_dim]

        dims = [input_dim] + hidden_dim + [output_dim]

        def fcn(x, params):
            weights = []
            biases = []
            for i in range(len(dims) - 1):
                weights.append(tf.reshape(x[2 * i], [dims[i], dims[i + 1]]))
                biases.append(tf.reshape(x[2 * i + 1], [1, dims[i + 1]]))

            cur_layer = params["data"]
            for i in range(len(dims) - 1):
                if i == len(dims) - 2:
                    cur_layer = tf.matmul(cur_layer, weights[i]) + biases[i]
                else:
                    cur_layer = tf.nn.relu(tf.matmul(cur_layer, weights[i]) + biases[i])

            output = cur_layer

            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=output, labels=params["labels"]
                )
            )

            # L2 regularization for the fully connected parameters.
            regularizers = tf.nn.l2_loss(weights[0])
            for i in range(1, len(dims) - 1):
                regularizers += tf.nn.l2_loss(weights[i])
            # Add the regularization term to the loss.
            loss += params["l2_weight"] * regularizers

            return loss

        param_sizes = []
        for i in range(len(dims) - 1):
            param_sizes.append(dims[i] * dims[i + 1])
            param_sizes.append(dims[i + 1])

        FcnFamily.__init__(
            self,
            fcn,
            param_sizes,
            {
                "data": {"type": tf.float64, "subsampled": True},
                "labels": {"type": tf.int64, "subsampled": True},
                "l2_weight": {"type": tf.float64},
            },
            {
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "output_dim": output_dim,
            },
            **kwargs,
        )


class NeuralNetFcn(Fcn):

    # labels is an N x 1 array, where N is the batch size
    def __init__(self, family, data, labels, l2_weight=5e-4, *args, **kwargs):
        assert labels.shape[1] == 1
        Fcn.__init__(
            self,
            family,
            {"data": data, "labels": labels[:, 0], "l2_weight": l2_weight},
            *args,
            **kwargs,
        )

    def unpack_x(self, x):
        unpacked_x = []
        prev_dim = 0
        for num_dim in self.family.num_dims:
            unpacked_x.append(x[prev_dim : prev_dim + num_dim, :])
            prev_dim += num_dim
        return unpacked_x

    def evaluate(self, x):
        return Fcn.evaluate(self, self.unpack_x(x))

    def grad(self, x):
        return np.vstack(Fcn.grad(self, self.unpack_x(x)))

    def hess(self, x):
        return np.vstack(
            [np.hstack(block_row) for block_row in Fcn.hess(self, self.unpack_x(x))]
        )


def main(*args):

    family = QuadFormFcnFamily(2)
    fcn = QuadFormFcn(family, np.array([[2.0, 1.0], [1.0, 2.0]]))
    print(fcn.evaluate(np.array([[-1.0], [2.0]])))
    print(fcn.grad(np.array([[-1.0], [2.0]])))
    print(fcn.hess(np.array([[-1.0], [2.0]])))
    family.destroy()

    input_dim = 5
    hidden_dim = [5]
    output_dim = 5
    num_examples = 10
    family = NeuralNetFcnFamily(input_dim, hidden_dim, output_dim)
    data = np.random.randn(num_examples, input_dim)
    labels = np.random.randint(output_dim, size=(num_examples, 1))
    fcn = NeuralNetFcn(family, data, labels)
    weights1 = np.random.randn(input_dim * hidden_dim[0], 1)
    biases1 = np.random.randn(hidden_dim[0], 1)
    weights2 = np.random.randn(hidden_dim[0] * output_dim, 1)
    biases2 = np.random.randn(output_dim, 1)
    x = np.vstack((weights1, biases1, weights2, biases2))
    print("Dimensionality: %d" % (x.shape[0]))
    print(fcn.evaluate(x))
    print(fcn.grad(x))

    family.destroy()


if __name__ == "__main__":
    main(*sys.argv[1:])
