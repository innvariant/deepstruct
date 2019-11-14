import torch
import os
import numpy as np
import pypaddle.util
import pypaddle.learning
import pypaddle.sparse
from torch.autograd import grad
from enum import Enum


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
        return self.prune_method in [optimal_brain_damage, optimal_brain_damage_absolute, optimal_brain_damage_bucket,
                                     optimal_brain_surgeon_layer_wise, optimal_brain_surgeon_layer_wise_bucket]

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
    for name, layer in pypaddle.sparse.prunable_layers_with_name(network):
        pypaddle.util.edge_cut(layer, hessian_inverse_path + name + '.npy', value=percentage)


def optimal_brain_surgeon_layer_wise_bucket(self, network, bucket_size):
    hessian_inverse_path = calculate_obsl_saliency(self, network)
    for name, layer in pypaddle.sparse.prunable_layers_with_name(network):
        pypaddle.util.edge_cut(layer, hessian_inverse_path + name + '.npy', value=bucket_size, strategy=PruningStrategy.BUCKET)


#
# Random pruning
#
def random_pruning(self, network, percentage):
    pypaddle.util.set_random_saliency(network)
    # prune the percentage% weights with the smallest random saliency
    prune_network_by_saliency(network, percentage)


def random_pruning_absolute(self, network, number):
    pypaddle.util.set_random_saliency(network)
    prune_network_by_saliency(network, number, strategy=PruningStrategy.ABSOLUTE)


#
# Magnitude based approaches
#
def magnitude_class_blinded(self, network, percentage):
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


def magnitude_class_blinded_absolute(self, network, number):
    prune_network_by_saliency(network, number, strategy=PruningStrategy.ABSOLUTE)


def magnitude_class_uniform(self, network, percentage):
    prune_layer_by_saliency(network, percentage)


def magnitude_class_uniform_absolute(self, network, number):
    prune_layer_by_saliency(network, number, strategy=PruningStrategy.ABSOLUTE)


def magnitude_class_distributed(self, network, percentage):
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
    pypaddle.util.set_distributed_saliency(network)
    # prune network
    prune_network_by_saliency(network, percentage)


def magnitude_class_distributed_absolute(self, network, number):
    pypaddle.util.set_distributed_saliency(network)
    prune_network_by_saliency(network, number, strategy=PruningStrategy.ABSOLUTE)


class PruningStrategy(Enum):
    """
    Enum to represent the different prunuing strategies that can be used.
    Note: not every pruning strategy can be used with every pruning method.
    """
    PERCENTAGE = 0
    ABSOLUTE = 1
    BUCKET = 2


def prune_network_by_saliency(network, value, strategy=PruningStrategy.PERCENTAGE):
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
    th = pypaddle.util.find_network_threshold(network, value, strategy=strategy)

    # set the mask
    for layer in pypaddle.sparse.prunable_layers(network):
        # All deleted weights should be set to zero so they should definetly be less than the threshold since this is
        # positive.
        layer.set_mask(torch.ge(layer.get_saliency(), th).float() * layer.get_mask())


def prune_layer_by_saliency(network, value, strategy=PruningStrategy.PERCENTAGE):
    pre_pruned_weight_count = pypaddle.util.get_network_weight_count(network).item()

    for layer in pypaddle.sparse.prunable_layers(network):
        mask = list(layer.get_mask().abs().numpy().flatten())
        saliency = list(layer.get_saliency().numpy().flatten())
        _, filtered_saliency = zip(
            *((masked_val, weight_val) for masked_val, weight_val in zip(mask, saliency) if masked_val == 1))

        # calculate threshold
        # percentage pruning
        if strategy is PruningStrategy.PERCENTAGE:
            th = np.percentile(np.array(filtered_saliency), value)
        # absolute pruning
        elif strategy is PruningStrategy.ABSOLUTE:
            # due to floating point operations this is not 100 percent exact a few more or less weights might get
            # deleted
            add_val = round((layer.get_weight_count() / pre_pruned_weight_count * value).item())

            # check if there are enough elements to prune select the highest element and add some penalty to it so the
            if add_val > layer.get_weight_count():
                th = np.argmax(filtered_saliency).item() + 1
            else:
                index = np.argsort(np.array(filtered_saliency))[add_val]
                th = np.array(filtered_saliency)[index].item()
        else:
            raise ValueError('Action is not supported!!!')

        # set mask
        layer.set_mask(torch.ge(layer.get_saliency(), th).float() * layer.get_mask())


def calculate_obd_saliency(self, network):
    # the loss of the network on the cross validation set
    loss = pypaddle.learning.cross_validation_error(self.valid_dataset, network, self.criterion)

    # calculate the first order gradients for all weights from the pruning layers.
    weight_params = map(lambda x: x.get_weight(), pypaddle.sparse.prunable_layers(network))
    loss_grads = grad(loss, weight_params, create_graph=True)

    # iterate over all layers and zip them with their corrosponding first gradient
    for grd, layer in zip(loss_grads, pypaddle.sparse.prunable_layers(network)):
        all_grads = []
        mask = layer.get_mask().view(-1)
        weight = layer.get_weight()

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
        layer.set_saliency(
            torch.tensor(all_grads).view(weight.size()) * layer.get_weight().data.pow(2) * 0.5)


def calculate_obsl_saliency(self, network):
    out_dir = './out/hessian'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        os.mkdir(out_dir + '/layerinput')
        os.mkdir(out_dir + '/inverse')

    # where to put the cached layer inputs
    layer_input_path = out_dir + '/layerinput/'
    # where to save the hessian matricies
    hessian_inverse_path = out_dir + '/inverse/'

    # generate the input in the layers and save it for every batch
    pypaddle.util.keep_input_layerwise(network)

    for i, (images, labels) in enumerate(self.valid_dataset):
        images = images.reshape(-1, 28 * 28)
        network(images)
        for name, layer in pypaddle.sparse.prunable_layers_with_name(network):
            layer_input = layer.layer_input.data.numpy()
            path = layer_input_path + name + '/'
            if not os.path.exists(path):
                os.mkdir(path)

            np.save(path + 'layerinput-' + str(i), layer_input)

    # generate the hessian matrix for each layer
    for name, layer in pypaddle.sparse.prunable_layers_with_name(network):
        hessian_inverse_location = hessian_inverse_path + name
        pypaddle.util.generate_hessian_inverse_fc(layer, hessian_inverse_location, layer_input_path + name)

    return hessian_inverse_path
