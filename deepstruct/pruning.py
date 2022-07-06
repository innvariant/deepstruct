import math
import os

from enum import Enum
from typing import Union

import numpy as np
import torch
import torch.nn as nn

from deprecated import deprecated
from torch.autograd import grad

import deepstruct.learning
import deepstruct.sparse
import deepstruct.util


@deprecated(
    reason="This class is not of use anymore as it gives no benefit over the functions in this package.",
    version="0.8.0",
)
class PruneNeuralNetMethod:
    """
    Strategy pattern for the selection of the currently used pruning method.
    Methods can be set during creation of the pruning method object.
    Valid methods are:

    <ul>
        <li>Random Pruning</li>
        <li>Magnitude Pruning Blinded</li>
        <li>Magnitude Pruning Uniform</li>
        <li>Optimal Brain Damage</li>
    </ul>

    The following algorithms are currently under construction:
    <ul>
        <li>Layer-wise Optimal Brain Surgeon</li>
    </ul>

    Method that were considered but due to inefficiency not implemented:
    <ul>
        <li>Optimal Brain Surgeon</li>
        <li>Net-Trim</li>
    </ul>

    All methods except of the random pruning and magnitude based pruning require the loss argument. In order to
    calculate the weight saliency in a top-down approach.
    If no Strategy is specified random pruning will be used as a fallback.
    """

    def __init__(self, method):
        """
        Creates a new PruneNeuralNetMethod object. There are a number of pruning methods supported.

        :param method:      The selected strategy for pruning If no pruning strategy is provided random pruning will be
                            selected as the standard pruning method.
        """
        self.prune_method = method

        # dataset and loss function for error calculation
        self.criterion = None
        self.valid_dataset = None

    def prune(self, network, value):
        """
        Wrapper method which calls the actual pruning strategy and computes how long it takes to complete the pruning
        step.

        :param network:     The network that should be pruned
        :param value:       The percentage of elements that should be pruned
        """
        self.prune_method(self, network, value)

    def requires_loss(self):
        """
        Check if the current pruning method needs the network's loss as an argument.
        :return: True iff a gradient of the network is required.
        """
        return self.prune_method in [
            optimal_brain_damage,
            optimal_brain_damage_absolute,
            optimal_brain_damage_bucket,
            optimal_brain_surgeon_layer_wise,
            optimal_brain_surgeon_layer_wise_bucket,
        ]

    def require_retraining(self):
        """
        Check if the current pruning strategy requires a retraining after the pruning is done
        :return: True iff the retraining is required.
        """
        # todo: does obs-l need retraining?
        # return self.prune_method not in [optimal_brain_surgeon_layer_wise, optimal_brain_surgeon_layer_wise_bucket]
        return True


#
# Top-Down Pruning Approaches
#


def optimal_brain_damage(self, network, percentage):
    """
    Implementation of the optimal brain damage algorithm.
    Requires the gradient to be set in the network.

    :param self:        The strategy pattern object for the pruning method.
    :param network:     The network where the calculations should be done.
    :param percentage:  The percentage of weights that should be pruned.
    """
    # calculate the saliencies for the weights
    calculate_obd_saliency(self, network)
    # prune the elements with the lowest saliency in the network
    prune_network_by_saliency(network, percentage)


def optimal_brain_damage_absolute(self, network, number):
    calculate_obd_saliency(self, network)
    prune_network_by_saliency(network, number, strategy=PruningStrategy.ABSOLUTE)


def optimal_brain_damage_bucket(self, network, bucket_size):
    calculate_obd_saliency(self, network)
    prune_network_by_saliency(network, bucket_size, strategy=PruningStrategy.BUCKET)


#
# Layer-wise approaches
#
def optimal_brain_surgeon_layer_wise(self, network, percentage):
    """
    Layer-wise calculation of the inverse of the hessian matrix. Then the weights are ranked similar to the original
    optimal brian surgeon algorithm.

    :param network:     The network that should be pruned.
    :param percentage:  What percentage of the weights should be pruned.
    :param self:        The strategy pattern object the method is attached to.
    """
    hessian_inverse_path = calculate_obsl_saliency(self, network)

    # prune the elements from the matrix
    for name, layer in deepstruct.sparse.prunable_layers_with_name(network):
        edge_cut(layer, hessian_inverse_path + name + ".npy", value=percentage)


def optimal_brain_surgeon_layer_wise_bucket(self, network, bucket_size):
    hessian_inverse_path = calculate_obsl_saliency(self, network)
    for name, layer in deepstruct.sparse.prunable_layers_with_name(network):
        edge_cut(
            layer,
            hessian_inverse_path + name + ".npy",
            value=bucket_size,
            strategy=PruningStrategy.BUCKET,
        )


#
# Random pruning
#
def random_pruning(self, network, percentage):
    set_random_saliency(network)
    # prune the percentage% weights with the smallest random saliency
    prune_network_by_saliency(network, percentage)


def random_pruning_absolute(self, network, number):
    set_random_saliency(network)
    prune_network_by_saliency(network, number, strategy=PruningStrategy.ABSOLUTE)


#
# Magnitude based approaches
#
def magnitude_class_blinded(network, percentage):
    """
    Implementation of weight based pruning. In each step the percentage of not yet pruned weights will be eliminated
    starting with the smallest element in the network.

    The here used method is the class blinded method mentioned in the paper by See et.al from 2016 (DOI: 1606.09274v1).
    The method is also known from the paper by Bachor et.al from 2018 where it was named the PruNet pruning technique
    (DOI: 10.1109/IJCNN.2018.8489764)

    :param network:     The network where the pruning should be done.
    :param percentage:  The percentage of not yet pruned weights that should be deleted.
    """
    prune_network_by_saliency(network, percentage)


def magnitude_class_blinded_absolute(network, number):
    prune_network_by_saliency(network, number, strategy=PruningStrategy.ABSOLUTE)


def magnitude_class_uniform(network, percentage):
    prune_layer_by_saliency(network, percentage)


def magnitude_class_uniform_absolute(network, number):
    prune_layer_by_saliency(network, number, strategy=PruningStrategy.ABSOLUTE)


def magnitude_class_distributed(network, percentage):
    """
    This idea comes from the paper 'Learning both Weights and Connections for Efficient Neural Networks'
    (arXiv:1506.02626v3). The main idea is that in each layer respectively to the standard derivation many elements
    should be deleted.
    For each layer prune the weights w for which the following holds:

    std(layer weights) * t > w      This is equal to the following
    t > w/std(layer_weights)        Since std is e(x - e(x))^2 and as square number positive.

    So all elements for which the wright divided by the std. derivation is smaller than some threshold will get deleted

    :param network:     The network that should be pruned.
    :param percentage:  The number of elements that should be pruned.
    :return:
    """
    # set saliency
    set_distributed_saliency(network)
    # prune network
    prune_network_by_saliency(network, percentage)


def magnitude_class_distributed_absolute(network, number):
    set_distributed_saliency(network)
    prune_network_by_saliency(network, number, strategy=PruningStrategy.ABSOLUTE)


class PruningStrategy(Enum):
    """
    Enum to represent the different prunuing strategies that can be used.
    Note: not every pruning strategy can be used with every pruning method.
    """

    PERCENTAGE = 0
    ABSOLUTE = 1
    BUCKET = 2


def prune_network_by_saliency(
    network: nn.Module,
    value: Union[float, torch.Tensor],
    strategy: PruningStrategy = PruningStrategy.PERCENTAGE,
):
    """
    Prune the number of percentage weights from the network. The elements are pruned according to the saliency that is
    set in the network. By default the saliency is the actual weight of the connections. The number of elements are
    pruned blinded. Meaning in each layer a different percentage of elements might get pruned but overall the given one
    is removed from the network. This is a different approach than used in the method below where we prune exactly the
    given percentage from each layer.

    :param network:     The layers of the network.
    :param value:       The number/percentage of elements that should be pruned.
    :param strategy:  If percentage pruning or number of elements pruning is used.
    """
    # calculate the network's threshold
    th = find_network_threshold(network, value, strategy=strategy)

    # set the mask
    for layer in deepstruct.sparse.maskable_layers(network):
        # All deleted weights should be set to zero so they should definetly be less than the threshold since this is
        # positive.
        layer.mask = torch.ge(layer.saliency, th).float() * layer.mask


def prune_layer_by_saliency(
    network: nn.Module, value, strategy=PruningStrategy.PERCENTAGE
):
    pre_pruned_weight_count = get_network_weight_count(network).item()

    for layer in deepstruct.sparse.maskable_layers(network):
        mask = list(layer.mask.numpy().flatten())
        saliency = list(layer.saliency.numpy().flatten())
        _, filtered_saliency = zip(
            *(
                (masked_val, weight_val)
                for masked_val, weight_val in zip(mask, saliency)
                if masked_val == 1
            )
        )

        filtered_saliency = np.array(filtered_saliency)

        # calculate threshold
        # percentage pruning
        if strategy is PruningStrategy.PERCENTAGE:
            th = np.percentile(filtered_saliency, value)
        # absolute pruning
        elif strategy is PruningStrategy.ABSOLUTE:
            # due to floating point operations this is not 100 percent exact a few more or less weights might get
            # deleted
            count_weight = layer.mask.sum()
            add_val = round(
                (np.divide(count_weight, pre_pruned_weight_count) * value).item()
            )

            th = (
                np.argsort(filtered_saliency)[add_val]
                if add_val <= count_weight
                else np.argmax(filtered_saliency).item()
            )
        else:
            raise ValueError(f"Chosen pruning strategy {strategy} is not supported.")

        layer.mask = torch.ge(layer.saliency, th).float() * layer.mask


def calculate_obd_saliency(self, network):
    # the loss of the network on the cross validation set
    loss = deepstruct.learning.cross_validation_error(
        self.valid_dataset, network, self.criterion
    )

    # calculate the first order gradients for all weights from the pruning layers.
    weight_params = map(lambda x: x.weight, deepstruct.sparse.maskable_layers(network))
    loss_grads = grad(loss, weight_params, create_graph=True)

    # iterate over all layers and zip them with their corrosponding first gradient
    for grd, layer in zip(loss_grads, deepstruct.sparse.maskable_layers(network)):
        all_grads = []
        mask = layer.mask.view(-1)
        weight = layer.weight

        if torch.cuda.is_available():
            weight.cuda()

        # zip gradient and mask of the network in a lineared fashion
        for num, (g, m) in enumerate(zip(grd.view(-1), mask)):
            if torch.cuda.is_available():
                g.cuda()

            if m.item() == 0:
                # if the element is pruned i.e. if mask == 0 then the second order derivative should e zero as well
                # so no computations are needed
                all_grads += [0]
            else:
                # create the second order derivative and add it to the list which contains all gradients
                drv = grad(g, weight, retain_graph=True)
                all_grads += [drv[0].view(-1)[num].item()]

        # rearrange calculated value to their normal form and set saliency
        layer.saliency = (
            torch.tensor(all_grads).view(weight.size()) * layer.weight.data.pow(2) * 0.5
        )


def calculate_obsl_saliency(self, network):
    out_dir = "./out/hessian"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        os.mkdir(out_dir + "/layerinput")
        os.mkdir(out_dir + "/inverse")

    # where to put the cached layer inputs
    layer_input_path = out_dir + "/layerinput/"
    # where to save the hessian matricies
    hessian_inverse_path = out_dir + "/inverse/"

    # generate the input in the layers and save it for every batch
    keep_input_layerwise(network)

    for i, (images, labels) in enumerate(self.valid_dataset):
        images = images.reshape(-1, 28 * 28)
        network(images)
        for name, layer in deepstruct.sparse.prunable_layers_with_name(network):
            layer_input = layer.layer_input.data.numpy()
            path = layer_input_path + name + "/"
            if not os.path.exists(path):
                os.mkdir(path)

            np.save(path + "layerinput-" + str(i), layer_input)

    # generate the hessian matrix for each layer
    for name, layer in deepstruct.sparse.prunable_layers_with_name(network):
        hessian_inverse_location = hessian_inverse_path + name
        deepstruct.util.generate_hessian_inverse_fc(
            layer, hessian_inverse_location, layer_input_path + name
        )

    return hessian_inverse_path


def edge_cut(
    layer,
    hessian_inverse_path,
    value,
    strategy=PruningStrategy.PERCENTAGE,
):
    """
    This function prune weights of biases based on given hessian inverse and cut ratio
    :param hessian_inverse_path:
    :param layer:
    :param value: The zeros percentage of weights and biases, or, 1 - compression ratio
    :param strategy:
    :return:
    """
    # dataset_size = layer2_input_train.shape[0]
    w_layer = layer.weight.data.numpy().T

    # biases = layer.bias.data.numpy()
    n_hidden_1 = w_layer.shape[0]
    n_hidden_2 = w_layer.shape[1]

    sensitivity = np.array([])

    hessian_inverse = np.load(hessian_inverse_path)

    gate_w = layer.mask.data.numpy().T
    # gate_b = np.ones([n_hidden_2])

    # calculate number of pruneable elements
    if strategy is PruningStrategy.PERCENTAGE:
        cut_ratio = value / 100  # transfer percentage from full value to floating point
        count_weight = layer.mask.sum()
        max_pruned_num = math.floor(count_weight * cut_ratio)
    elif strategy is PruningStrategy.BUCKET:
        max_pruned_num = value
    else:
        raise ValueError("Currently not implemented")

    # Calculate sensitivity score. Refer to Eq.5.
    for i in range(n_hidden_2):
        sensitivity = np.hstack(
            (sensitivity, 0.5 * ((w_layer.T[i] ** 2) / np.diag(hessian_inverse)))
        )
    sorted_index = np.argsort(sensitivity)
    hessian_inverseT = hessian_inverse.T

    prune_count = 0
    for i in range(n_hidden_1 * n_hidden_2):
        prune_index = [sorted_index[i]]
        x_index = math.floor(prune_index[0] / n_hidden_1)  # next layer num
        y_index = prune_index[0] % n_hidden_1  # this layer num

        if gate_w[y_index][x_index] == 1:
            delta_w = (
                -w_layer[y_index][x_index] / hessian_inverse[y_index][y_index]
            ) * hessian_inverseT[y_index]
            gate_w[y_index][x_index] = 0

            if strategy is strategy.PERCENTAGE:
                prune_count += 1
            elif strategy is strategy.BUCKET:
                prune_count += sensitivity[
                    prune_index
                ]  # todo: evaluate here probably a bit wrong :(

            # Parameters update, refer to Eq.5
            w_layer.T[x_index] = w_layer.T[x_index] + delta_w
            # b_layer[x_index] = b_layer[x_index] + delta_w[-1]

        w_layer = w_layer * gate_w
        # b_layer = b_layer * gate_b

        if prune_count == max_pruned_num:
            break

    # set created mask to network again and update the weights
    layer.mask = torch.from_numpy(gate_w.T)
    layer.weight = torch.nn.Parameter(torch.from_numpy(w_layer.T))

    # if not os.path.exists(prune_save_path):
    #    os.makedirs(prune_save_path)

    # np.save("%s/weights" % prune_save_path, w_layer)
    # np.save("%s/biases" % prune_save_path, b_layer)


def find_network_threshold(
    network: nn.Module, value: Union[float, torch.Tensor], strategy: PruningStrategy
):
    all_sal = []
    for layer in deepstruct.sparse.maskable_layers(network):
        # flatten both weights and mask
        mask = list(layer.mask.numpy().flatten())
        saliency = list(layer.saliency.numpy().flatten())

        # zip, filter, unzip the two lists
        _, filtered_saliency = zip(
            *(
                (masked_val, weight_val)
                for masked_val, weight_val in zip(mask, saliency)
                if masked_val == 1
            )
        )
        # add all saliencies to list
        all_sal += filtered_saliency

    all_sal = np.array(all_sal)
    if strategy is PruningStrategy.PERCENTAGE:
        # calculate percentile
        # e.g. np.percentile(all_sal, 90) will return the saliency value of the upper 90%
        return np.percentile(all_sal, value)
    elif strategy is PruningStrategy.ABSOLUTE:
        # Retrieve the index of the weight at the position of the given threshold value
        index = (
            np.argsort(all_sal)[value]
            if value < len(all_sal)
            else np.argmax(all_sal).item()
        )
        return all_sal[index].item()
    elif strategy is PruningStrategy.BUCKET:
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


def set_random_saliency(model: nn.Module):
    # set saliency to random values
    for layer in deepstruct.sparse.maskable_layers(model):
        layer.saliency = torch.rand_like(layer.weight) * layer.mask


def set_random_masks(module: nn.Module):
    if isinstance(module, deepstruct.sparse.MaskedLinearLayer):
        module.mask = torch.round(torch.rand_like(module.weight))


def set_distributed_saliency(network):
    # prune from each layer the according number of elements
    for layer in deepstruct.sparse.maskable_layers(network):
        # calculate standard deviation for the layer
        w = layer.weight.data
        st_v = 1 / w.std()
        # set the saliency in the layer = weight/st.deviation
        layer.saliency = st_v * w.abs()


def reset_pruned_network(network):
    for layer in deepstruct.sparse.maskable_layers(network):
        layer.reset_parameters(keep_mask=True)


def keep_input_layerwise(network):
    for layer in deepstruct.sparse.maskable_layers(network):
        layer.keep_layer_input = True


def get_network_weight_count(network: nn.Module):
    total_weights = 0
    for layer in deepstruct.sparse.maskable_layers(network):
        total_weights += layer.get_weight_count()
    return total_weights
