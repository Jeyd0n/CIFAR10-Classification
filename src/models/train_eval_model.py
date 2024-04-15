import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def train(model, dataloader, optimizer, loss_function):
    model.train()

    for X, y in tqdm(dataloader):
        output = model(X)

        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(model, dataloader, loss_fn):
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

    print(
        f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


def predict(model, X):
    def imshow(img, title):
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
