from torchvision import datasets
from torchvision import transforms


def load_data(train=True):
    if train == True:
        dataset = datasets.CIFAR10(
            root="../data/raw/",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )

        return dataset
    else:
        dataset = datasets.CIFAR10(
            root="../data/raw/",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )

        return dataset
