import os

import numpy as np
import torch

import deepstruct.graph
import deepstruct.scalable
import deepstruct.sparse


def test_skip_connections():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ct1_nx_flat = deepstruct.graph.CachedLayeredGraph()
    ct1_nx_flat.add_nodes_from([1, 2, 3, 4, 5])
    ct1_nx_flat.add_edges_from([(1, 3), (2, 4), (3, 4), (4, 5)])

    architecture = deepstruct.scalable.ScalableDAN(
        ct1_nx_flat, deepstruct.graph.uniform_proportions(ct1_nx_flat)
    )

    size_batch = 50
    size_input = 20
    features_random = torch.tensor(
        np.random.random((size_batch, size_input)), device=device, requires_grad=False
    )
    fn_model = architecture.build(size_input, 2, 50)

    prediction = fn_model(features_random)
    print(prediction.shape)


def test_load_reload_model(tmp_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ct1_nx_flat = deepstruct.graph.CachedLayeredGraph()
    ct1_nx_flat.add_nodes_from([1, 2, 3, 4, 5])
    ct1_nx_flat.add_edges_from([(1, 3), (2, 4), (3, 4), (4, 5)])

    size_batch = 50
    size_input = 20
    size_output = 2

    # Init
    architecture = deepstruct.scalable.ScalableDAN(
        ct1_nx_flat, deepstruct.graph.uniform_proportions(ct1_nx_flat)
    )
    fn_model, structure_init = architecture.build(
        size_input, size_output, 50, return_graph=True
    )
    fn_model.to(device)

    # Store
    path_checkpoint = os.path.join(tmp_path, "init.pth")
    path_structure = os.path.join(tmp_path, "init.graphml")
    torch.save({"model_state": fn_model.state_dict()}, path_checkpoint)
    structure_init.save(path_structure)

    # Reload
    structure_reloaded = deepstruct.graph.CachedLayeredGraph.load(path_structure)
    checkpoint = torch.load(path_checkpoint)
    fn_reloaded = deepstruct.scalable.ScalableDAN.model(
        size_input, size_output, structure_reloaded, use_layer_norm=True
    )
    fn_reloaded.load_state_dict(checkpoint["model_state"])
    fn_reloaded.to(device)

    # Inference
    fn_reloaded.train()
    optimizer = torch.optim.Adam(fn_reloaded.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    features_random = torch.tensor(
        np.random.random((size_batch, size_input)), device=device, requires_grad=False
    )
    features_random.to(device)
    targets_random = torch.tensor(
        np.random.randint(0, 2, size_batch), device=device, requires_grad=False
    )
    targets_random.to(device)
    prediction = fn_reloaded(features_random)
    loss = criterion(prediction, targets_random)
    loss.backward()
    optimizer.step()
