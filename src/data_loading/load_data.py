from torchvision import datasets
from torchvision import transforms


def load_data(train=True):
    '''
    Загрузка датасета из зоопарка pytroch

    Parameters
    ----------
    train : bool
        Метка, показывающая тренировочный ли датасет или тестовый

    Returns
    -------
    dataset : torchvision.datasets
        Набор данных CIFAR10
    '''
    if train:
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
