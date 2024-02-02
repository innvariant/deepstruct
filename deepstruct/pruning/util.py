import torch

import deepstruct.sparse


def set_random_saliency(model: torch.nn.Module):
    # set saliency to random values
    for layer in deepstruct.sparse.maskable_layers(model):
        layer.saliency = torch.rand_like(layer.weight) * layer.mask


def set_random_masks(module: torch.nn.Module):
    if isinstance(module, deepstruct.sparse.MaskedLinearLayer):
        module.mask = torch.round(torch.rand_like(module.weight))


def set_distributed_saliency(module: torch.nn.Module):
    # prune from each layer the according number of elements
    for layer in deepstruct.sparse.maskable_layers(module):
        # calculate standard deviation for the layer
        w = layer.weight.data
        st_v = 1 / w.std()
        # set the saliency in the layer = weight/st.deviation
        layer.saliency = st_v * w.abs()


def reset_pruned_network(module: torch.nn.Module):
    for layer in deepstruct.sparse.maskable_layers(module):
        layer.reset_parameters(keep_mask=True)


def keep_input_layerwise(module: torch.nn.Module):
    for layer in deepstruct.sparse.maskable_layers(module):
        layer.keep_layer_input = True


def get_network_weight_count(module: torch.nn.Module):
    total_weights = 0
    for layer in deepstruct.sparse.maskable_layers(module):
        total_weights += layer.get_weight_count()
    return total_weights
