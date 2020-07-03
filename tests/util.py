import os
import torch
import torchvision


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
