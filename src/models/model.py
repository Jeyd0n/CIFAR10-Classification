import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
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

# class CNN(nn.Module):
#     def __init__(self, input_size=(32, 32, 3), out_size=10):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, 1)   
#         self.conv2 = nn.Conv2d(32, 64, 3, 1)

#         self.dropout1 = nn.Dropout(0.25)
#         self.dropout2 = nn.Dropout(0.5)

#         self.fc1 = nn.Linear((input_size[0] * input_size[1] * input_size[2]) / 9, 128)         
#         self.fc2 = nn.Linear(128, out_size)

#         self.bn1 = nn.BatchNorm2d(32)
#         self.bn2 = nn.BatchNorm2d(64)

#         def forward(self, x):
#             x = self.conv1(x)
#             x = self.bn1(x)
#             x = F.relu(x)

#             x = self.conv2(x)
#             x = self.bn2(x)
#             x = F.relu(x)

#             x = F.max_pool2d(x, 2)
#             x = self.dropout1(x)
            
#             x = torch.flatten(x, 1)
            
#             x = self.fc1(x)
#             x = F.relu(x)
#             x = self.dropout2(x)
#             x = self.fc2(x)
            
#             return x