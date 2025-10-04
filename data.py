import torch
from torchvision import datasets, transforms

def get_mnist_loader(batch_size=64, train=True):
    """
    Returns a DataLoader for the MNIST dataset.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(
        './data',
        train=train,
        download=True,
        transform=transform
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train
    )
    return loader