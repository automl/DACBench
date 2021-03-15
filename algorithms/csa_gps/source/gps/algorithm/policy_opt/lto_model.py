import tensorflow as tf
from gps.algorithm.policy_opt.tf_utils import TfMap
import numpy as np


def init_weights(shape, name=None):
    return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)


def init_bias(shape, name=None):
    return tf.Variable(tf.zeros(shape, dtype="float"), name=name)


def batched_matrix_vector_multiply(vector, matrix):
    """ computes x^T A in mini-batches. """
    vector_batch_as_matricies = tf.expand_dims(vector, [1])
    mult_result = tf.matmul(vector_batch_as_matricies, matrix)
    squeezed_result = tf.squeeze(mult_result, [1])
    return squeezed_result


def get_input_layer():
    """produce the placeholder inputs that are used to run ops forward and backwards.
    net_input: usually an observation.
    action: mu, the ground truth actions we're trying to learn.
    precision: precision matrix used to compute loss."""
    
    net_input = tf.placeholder("float", [None, None], name="nn_input")  # (N*T) x dO
    action = tf.placeholder("float", [None, None], name="action")  # (N*T) x dU
    precision = tf.placeholder(
        "float", [None, None, None], name="precision"
    )  # (N*T) x dU x dU
    return net_input, action, precision


def get_loss_layer(mlp_out, action, precision, batch_size):
    """The loss layer used for the MLP network is obtained through this class."""
    scale_factor = tf.constant(2 * batch_size, dtype="float")
    uP = batched_matrix_vector_multiply(action - mlp_out, precision)
    uPu = tf.reduce_sum(
        uP * (action - mlp_out)
    )  # this last dot product is then summed, so we just the sum all at once.
    return uPu / scale_factor


def fully_connected_tf_network(
    dim_input, dim_output, batch_size=25, network_config=None
):

    dim_hidden = network_config["dim_hidden"] + [dim_output]
    n_layers = len(dim_hidden)

    nn_input, action, precision = get_input_layer()

    weights = []
    biases = []
    in_shape = dim_input
    for layer_step in range(0, n_layers):
        cur_weight = init_weights(
            [in_shape, dim_hidden[layer_step]], name="w_" + str(layer_step)
        )
        cur_bias = init_bias([dim_hidden[layer_step]], name="b_" + str(layer_step))
        in_shape = dim_hidden[layer_step]
        weights.append(cur_weight)
        biases.append(cur_bias)

    cur_top = nn_input
    for layer_step in range(0, n_layers):
        if layer_step != n_layers - 1:  # final layer has no RELU
            cur_top = tf.nn.relu(
                tf.matmul(cur_top, weights[layer_step]) + biases[layer_step]
            )
        else:
            cur_top = tf.nn.relu6(
                tf.matmul(cur_top, weights[layer_step]) + biases[layer_step]
            )

    mlp_applied = cur_top
    loss_out = get_loss_layer(
        mlp_out=mlp_applied, action=action, precision=precision, batch_size=batch_size
    )

    return TfMap.init_from_lists(
        [nn_input, action, precision], [mlp_applied], [loss_out]
    )
