import os
import math
import torch
import torch.nn as nn
import networkx as nx
import numpy as np
import torchvision

import pypaddle.pruning
import pypaddle.sparse

from torch.utils.data import SubsetRandomSampler


def get_cifar10_loaders(batch_size:int = 100, possible_dataset_roots = 'data/set/cifar10'):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if type(possible_dataset_roots) is str:
        possible_dataset_roots = [possible_dataset_roots]
    assert type(possible_dataset_roots) is list
    assert len(possible_dataset_roots) > 0

    selected_root = possible_dataset_roots[0]
    for possible_root in possible_dataset_roots:
        if os.path.exists(possible_root):
            selected_root = possible_root
    if not os.path.exists(selected_root):
        os.makedirs(selected_root)

    train_set = torchvision.datasets.CIFAR10(root=selected_root, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root=selected_root, train=False, download=True, transform=transform)
    n_training_samples = 20000
    train_sampler = SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))
    n_val_samples = 5000
    val_sampler = SubsetRandomSampler(np.arange(n_training_samples, n_training_samples + n_val_samples, dtype=np.int64))
    n_test_samples = 5000
    test_sampler = SubsetRandomSampler(np.arange(n_test_samples, dtype=np.int64))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, sampler=test_sampler, num_workers=2)
    val_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=val_sampler, num_workers=2)

    return train_loader, test_loader, val_loader, selected_root


def get_mnist_loaders(batch_size:int = 100, possible_dataset_roots = 'data/set/mnist'):
    custom_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),  # first, convert image to PyTorch tensor
        torchvision.transforms.Normalize((0.1307,), (0.3081,))  # normalize inputs
    ])

    if type(possible_dataset_roots) is str:
        possible_dataset_roots = [possible_dataset_roots]
    assert type(possible_dataset_roots) is list
    assert len(possible_dataset_roots) > 0

    selected_root = possible_dataset_roots[0]
    for possible_root in possible_dataset_roots:
        if os.path.exists(possible_root):
            selected_root = possible_root
    if not os.path.exists(selected_root):
        os.makedirs(selected_root)

    train_set = torchvision.datasets.MNIST(root=selected_root, download=True, train=True, transform=custom_transform)
    test_set = torchvision.datasets.MNIST(root=selected_root, download=True, train=False, transform=custom_transform)

    mnist_train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    mnist_test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)

    return mnist_train_loader, mnist_test_loader, None, selected_root


def build_layer_index(graph : nx.DiGraph, layer_index=None):
    """

    :param graph:
    :type graph igraph.Graph
    :param layer_index:
    :return:
    """
    if layer_index is None:
        layer_index = {}

    recursion_call = {'count': 0}

    def get_layer_index(vertex, graph):
        try:
            vertex = int(vertex)
        except TypeError:
            raise ValueError('You have to pass vertex indices to this function.')
        # print('get_layer_index(%s, roots, graph)' % vertex)
        if vertex is None:
            raise ValueError('Given vertex was none.')
        if vertex not in layer_index:
            recursion_call['count'] += 1
            layer_index[vertex] = max(
                [get_layer_index(v, graph) for v in nx.algorithms.dag.ancestors(graph, vertex)] + [-1]) + 1
        return layer_index[vertex]

    for v in graph:
        get_layer_index(v, graph)

    # print('Recursion call: %s' % recursion_call['count'])

    vertices_by_layer = {}
    for v in layer_index:
        idx = layer_index[v]
        if idx not in vertices_by_layer:
            vertices_by_layer[idx] = []
        vertices_by_layer[idx].append(v)

    return layer_index, vertices_by_layer



"""
This file contains some utility functions to calculate hessian matrix and its inverse.

Adapted from https://github.com/csyhhu/L-OBS/blob/master/PyTorch/ImageNet/util.py
Author: Chen Shangyu (schen025@e.ntu.edu.sg)
"""


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
        layer2_input_train = np.load(layer_input_train_dir + '/' + input_file)

        if input_index == 0:
            dataset_size = layer2_input_train.shape[0] * len(os.listdir(layer_input_train_dir))

        for i in range(layer2_input_train.shape[0]):
            # vect_w_b = np.vstack((np.array([layer2_input_train[i]]).T, np.array([[1.0]])))
            vect_w = np.array([layer2_input_train[i]]).T
            denominator = dataset_size + np.dot(np.dot(vect_w.T, hessian_inverse), vect_w)
            numerator = np.dot(np.dot(hessian_inverse, vect_w), np.dot(vect_w.T, hessian_inverse))
            hessian_inverse = hessian_inverse - numerator * (1.00 / denominator)

    np.save(hessian_inverse_path, hessian_inverse)


def edge_cut(layer, hessian_inverse_path, value, strategy=pypaddle.pruning.PruningStrategy.PERCENTAGE):
    """
    This function prune weights of biases based on given hessian inverse and cut ratio
    :param hessian_inverse_path:
    :param layer:
    :param value: The zeros percentage of weights and biases, or, 1 - compression ratio
    :param strategy:
    :return:
    """
    # dataset_size = layer2_input_train.shape[0]
    w_layer = layer.get_weight().data.numpy().T

    # biases = layer.bias.data.numpy()
    n_hidden_1 = w_layer.shape[0]
    n_hidden_2 = w_layer.shape[1]

    sensitivity = np.array([])

    hessian_inverse = np.load(hessian_inverse_path)

    gate_w = layer.get_mask().data.numpy().T
    # gate_b = np.ones([n_hidden_2])

    # calculate number of pruneable elements
    if strategy is pypaddle.pruning.PruningStrategy.PERCENTAGE:
        cut_ratio = value / 100  # transfer percentage from full value to floating point
        max_pruned_num = math.floor(layer.get_weight_count() * cut_ratio)
    elif strategy is pypaddle.pruning.PruningStrategy.BUCKET:
        max_pruned_num = value
    else:
        raise ValueError('Currently not implemented')

    # Calculate sensitivity score. Refer to Eq.5.
    for i in range(n_hidden_2):
        sensitivity = np.hstack(
            (sensitivity, 0.5 * ((w_layer.T[i] ** 2) / np.diag(hessian_inverse))))
    sorted_index = np.argsort(sensitivity)
    hessian_inverseT = hessian_inverse.T

    prune_count = 0
    for i in range(n_hidden_1 * n_hidden_2):
        prune_index = [sorted_index[i]]
        x_index = math.floor(prune_index[0] / n_hidden_1)  # next layer num
        y_index = prune_index[0] % n_hidden_1  # this layer num

        if gate_w[y_index][x_index] == 1:
            delta_w = (-w_layer[y_index][x_index] / hessian_inverse[y_index][y_index]) * hessian_inverseT[y_index]
            gate_w[y_index][x_index] = 0

            if strategy is strategy.PERCENTAGE:
                prune_count += 1
            elif strategy is strategy.BUCKET:
                prune_count += sensitivity[prune_index]  # todo: evaluate here probably a bit wrong :(

            # Parameters update, refer to Eq.5
            w_layer.T[x_index] = w_layer.T[x_index] + delta_w
            # b_layer[x_index] = b_layer[x_index] + delta_w[-1]

        w_layer = w_layer * gate_w
        # b_layer = b_layer * gate_b

        if prune_count == max_pruned_num:
            break

    # set created mask to network again and update the weights
    layer.set_mask(torch.from_numpy(gate_w.T))
    layer.weight = torch.nn.Parameter(torch.from_numpy(w_layer.T))

    # if not os.path.exists(prune_save_path):
    #    os.makedirs(prune_save_path)

    # np.save("%s/weights" % prune_save_path, w_layer)
    # np.save("%s/biases" % prune_save_path, b_layer)


# -----------

def find_network_threshold(network, value, strategy):
    all_sal = []
    for layer in pypaddle.sparse.prunable_layers(network):
        # flatten both weights and mask
        mask = list(layer.get_mask().abs().numpy().flatten())
        saliency = list(layer.get_saliency().numpy().flatten())

        # zip, filter, unzip the two lists
        _, filtered_saliency = zip(
            *((masked_val, weight_val) for masked_val, weight_val in zip(mask, saliency) if masked_val == 1))
        # add all saliencies to list
        all_sal += filtered_saliency

    # calculate percentile
    if strategy is pypaddle.pruning.PruningStrategy.PERCENTAGE:
        return np.percentile(np.array(all_sal), value)
    elif strategy is pypaddle.pruning.PruningStrategy.ABSOLUTE:
        # check if there are enough elements to prune
        if value >= len(all_sal):
            return np.argmax(np.array(all_sal)).item() + 1
        else:
            # determine threshold
            index = np.argsort(np.array(all_sal))[value]
            return np.array(all_sal)[index].item()
    elif strategy is pypaddle.pruning.PruningStrategy.BUCKET:
        sorted_array = np.sort(all_sal)
        sum_array = 0

        # sum up the elements from the sorted array
        for i in sorted_array:
            sum_array += i
            if sum_array > value:
                # return the last element that has been added to the array
                return i

        # return last element if bucket not reached
        return sorted_array[-1] + 1



def set_random_saliency(network):
    # set saliency to random values
    for layer in pypaddle.sparse.prunable_layers(network):
        layer.set_saliency(torch.rand_like(layer.get_weight()) * layer.get_mask())

def set_random_masks(module : nn.Module):
    if isinstance(module, pypaddle.sparse.MaskedLinearLayer):
        module.set_mask(torch.round(torch.rand_like(module.get_weight())))

def set_distributed_saliency(network):
    # prune from each layer the according number of elements
    for layer in pypaddle.sparse.prunable_layers(network):
        # calculate standard deviation for the layer
        w = layer.get_weight().data
        st_v = 1 / w.std()
        # set the saliency in the layer = weight/st.deviation
        layer.set_saliency(st_v * w.abs())


def get_filtered_saliency(saliency, mask):
    s = list(saliency)
    m = list(mask)

    _, filtered_w = zip(
        *((masked_val, weight_val) for masked_val, weight_val in zip(m, s) if masked_val == 1))
    return filtered_w


def get_layer_count(network):
    i = 0
    for _ in pypaddle.sparse.prunable_layers_with_name(network):
        i += 1
    return i


def get_weight_distribution(network):
    all_weights = []
    for layer in pypaddle.sparse.prunable_layers(network):
        mask = list(layer.get_mask().numpy().flatten())
        weights = list(layer.get_weight().data.numpy().flatten())

        masked_val, filtered_weights = zip(
            *((masked_val, weight_val) for masked_val, weight_val in zip(mask, weights) if masked_val == 1))

        all_weights += list(filtered_weights)

    # return all the weights, that are not masked as a numpy array
    return np.array(all_weights)


def get_network_weight_count(network):
    total_weights = 0
    for layer in pypaddle.sparse.prunable_layers(network):
        total_weights += layer.get_weight_count()
    return total_weights


def reset_pruned_network(network):
    for layer in pypaddle.sparse.prunable_layers(network):
        layer.reset_parameters(keep_mask=True)


def keep_input_layerwise(network):
    for layer in pypaddle.sparse.prunable_layers(network):
        layer.keep_layer_input = True
