import math
import os

import numpy as np

import deepstruct.sparse


def generate_hessian_inverse_fc(layer, hessian_inverse_path, layer_input_train_dir):
    """
    This function calculate hessian inverse for a fully-connect layer
    :param hessian_inverse_path: the hessian inverse path you store
    :param layer: the layer weights
    :param layer_input_train_dir: layer inputs in the dir
    :return:
    """

    w_layer = layer.get_weight().data.numpy().T
    n_hidden_1 = w_layer.shape[0]

    # Here we use a recursive way to calculate hessian inverse
    hessian_inverse = 1000000 * np.eye(n_hidden_1)

    dataset_size = 0
    for input_index, input_file in enumerate(os.listdir(layer_input_train_dir)):
        layer2_input_train = np.load(layer_input_train_dir + "/" + input_file)

        if input_index == 0:
            dataset_size = layer2_input_train.shape[0] * len(
                os.listdir(layer_input_train_dir)
            )

        for i in range(layer2_input_train.shape[0]):
            # vect_w_b = np.vstack((np.array([layer2_input_train[i]]).T, np.array([[1.0]])))
            vect_w = np.array([layer2_input_train[i]]).T
            denominator = dataset_size + np.dot(
                np.dot(vect_w.T, hessian_inverse), vect_w
            )
            numerator = np.dot(
                np.dot(hessian_inverse, vect_w), np.dot(vect_w.T, hessian_inverse)
            )
            hessian_inverse = hessian_inverse - numerator * (1.00 / denominator)

    np.save(hessian_inverse_path, hessian_inverse)


def get_filtered_saliency(saliency, mask):
    s = list(saliency)
    m = list(mask)

    _, filtered_w = zip(
        *(
            (masked_val, weight_val)
            for masked_val, weight_val in zip(m, s)
            if masked_val == 1
        )
    )
    return filtered_w


def get_layer_count(network):
    i = 0
    for _ in deepstruct.sparse.prunable_layers_with_name(network):
        i += 1
    return i


def get_weight_distribution(network):
    all_weights = []
    for layer in deepstruct.sparse.prunable_layers(network):
        mask = list(layer.get_mask().numpy().flatten())
        weights = list(layer.get_weight().data.numpy().flatten())

        masked_val, filtered_weights = zip(
            *(
                (masked_val, weight_val)
                for masked_val, weight_val in zip(mask, weights)
                if masked_val == 1
            )
        )

        all_weights += list(filtered_weights)

    # return all the weights, that are not masked as a numpy array
    return np.array(all_weights)


def entropy(labels, base=None):
    value, counts = np.unique(labels, return_counts=True)
    norm_counts = counts / counts.sum()
    base = math.e if base is None else base
    return -(norm_counts * np.log(norm_counts) / np.log(base)).sum()


def kullback_leibler(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))
