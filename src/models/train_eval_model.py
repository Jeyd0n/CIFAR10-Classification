import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def train(model, dataloader, optimizer, loss_function):
    '''
    Тренировка модели

    Parameters
    ----------
    model : nn.Module
        Модель глубокого обучения

    dataloader : torch.utils.data.DataLoader
        Даталоудер, подающий данные в модель

    optimizer : torch.optim
        Метод оптимизации модели

    loss_function : 
        Функция потерь, которую мы будем оптимизировать
    
    Outputs
    -------
    None


    '''
    model.train()

    for X, y in tqdm(dataloader):
        output = model(X)

        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(model, dataloader, loss_fn):
    '''
    Тестирование модели

    Parameters
    ----------
    model : nn.Module
        Модель глубокого обучения

    dataloader : torch.utils.data.DataLoader
        Даталоудер, подающий данные в модель

    optimizer : torch.optim
        Метод оптимизации модели

    loss_function : 
        Функция потерь, которую мы будем оптимизировать
    
    Returns
    -------
    accuracy : int
        Точность модели 

    avg_loss : int
        Средняя точность (по функции потерь)


    '''
    model.eval()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in tqdm(dataloader):
            pred = model(X)

            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    accuracy = round(correct, 3)
    avg_loss = round(test_loss, 3)

    return accuracy, avg_loss


def predict(model, X):
    '''
    Предсказание модели на тенсоре X, а так же вывод картинки с предсказанным классом

    Parameters
    ----------
    model : nn.Module
        Модель глубокого обучения

    X : torch.Tensor
        Тенсор изображения

    Returns
    -------
    None


    '''
    def imshow(img, title):
        '''
        Вывод изображения

        Parameters
        ----------
        img : torch.Tensor
            Тенсор изображения

        title : str
            Надпись над картинкой

        Returns
        -------
        None


        '''
        npimg = img.numpy()

        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.title(title)
        plt.show()

    LABELS_MAP = [
        "airplanes",
        "cars",
        "birds",
        "cats",
        "deer",
        "dogs",
        "frogs",
        "horses",
        "ships",
        "trucks",
    ]

    model.eval()
    with torch.no_grad():
        logits = model(np.transpose(X.flatten().unsqueeze(1), (1, 0)))

    pred_prob = nn.Softmax(dim=1)(logits)
    Y_pred = pred_prob.argmax(1)

    imshow(X, LABELS_MAP[Y_pred.item()])
