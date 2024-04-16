import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    '''
    Модель глубокого обучения, основанная на линейных слоях

    Parameters
    ----------
    input_size : tuple
        Размер входного тенсора

    output_size : int
        Размер выходного теснора (количество классов)

    Returns
    -------
    list
        Вероятности принадлежности к n-му классу

    
    '''
    def __init__(self, input_size=(32, 32, 3), output_size=10):
        super().__init__()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size[0] * input_size[1] * input_size[2], 512)
        self.fc2 = nn.Linear(512, 512)
        self.relu = nn.ReLU(inplace=True)
        self.output = nn.Linear(512, output_size)

    def forward(self, x):
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)

        return self.output(x)
    
class NeuralNetworkMoreLayers(nn.Module):
    '''
    Модель глубокого обучения, основанная на линейных слоях

    Parameters
    ----------
    input_size : tuple
        Размер входного тенсора

    output_size : int
        Размер выходного теснора (количество классов)

    Returns
    -------
    list
        Вероятности принадлежности к n-му классу

    
    '''
    def __init__(self, input_size=(32, 32, 3), output_size=10):
        super().__init__()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size[0] * input_size[1] * input_size[2], 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.relu = nn.ReLU(inplace=True)
        self.output = nn.Linear(512, output_size)

    def forward(self, x):
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)

        return self.output(x)


class CNN(nn.Module):
    def __init__(self, input_size=(32, 32, 3), out_size=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)    
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(12544, 128)         
        self.fc2 = nn.Linear(128, out_size)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return x
