from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from nnkek.utils.common import get_device


class Autoencoder(nn.Module):
    """Autoencoder for compressing image features obtained from feeding forward
    an image to InceptionV3. Bottleneck layer space is of 400 dimensions
    """

    def __init__(self):
        super(Autoencoder, self).__init__()

        self.input_size = 2048
        self.hidden_size = 400

        # encoder
        self.enc1 = nn.Linear(in_features=2048, out_features=1024)
        self.enc2 = nn.Linear(in_features=1024, out_features=512)
        self.enc3 = nn.Linear(in_features=512, out_features=400)

        # decoder
        self.dec1 = nn.Linear(in_features=400, out_features=512)
        self.dec2 = nn.Linear(in_features=512, out_features=1024)
        self.dec3 = nn.Linear(in_features=1024, out_features=2048)

        self.to(get_device())

    def encode(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        # здесь убираем релу, потому что хотим, чтобы наши эмбеддинги имели и позитивные и отрицательные значения
        # мы же их будем потом скалярно перемножать с эмбеддингами фасттекста -> знак - это сигнал
        x = self.enc3(x)
        return x

    def decode(self, x):
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        # здесь убираем релу, т.к. входные эмбеддинги имеют отрицательные значения, значит МСЕ не способное их восстановить после релу
        # будет пытаться компенсировать в другом месте (разберись, как..)
        x = self.dec3(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def fit(
        self,
        trainloader,
        valloader=None,
        optimizer=None,
        criterion=nn.MSELoss(),
        num_epochs=100,
        device=None,
        debug=True,
    ):
        """
        :param trainloader:
        :param optimizer: if None, than torch.optim.Adam is used as default
        :param criterion:
        :param num_epochs:
        :param device:
        :param debug:
        :return:
        """

        optimizer = optimizer or optim.Adam(self.parameters())

        device = device or get_device()
        self.to(device)

        train_loss = []
        val_loss = []

        for epoch in range(num_epochs):
            cumulative_loss = 0.0

            for batch in trainloader:
                batch = batch.to(device)
                optimizer.zero_grad()

                outputs = self(batch)

                loss = criterion(outputs, batch)
                loss.backward()
                optimizer.step()

                cumulative_loss += loss.item()

            epoch_loss = cumulative_loss / len(trainloader)
            train_loss.append(epoch_loss)

            if debug:
                print("Epoch {} of {}, train Loss: {:.3f}".format(epoch + 1, num_epochs, epoch_loss), end=", ")

            if valloader:
                cumulative_val_loss = 0.0

                for batch in valloader:
                    batch = batch.to(device)

                    outputs = self(batch)
                    loss = criterion(outputs, batch)
                    cumulative_val_loss += loss.item()

                epoch_val_loss = cumulative_val_loss / len(valloader)
                val_loss.append(epoch_val_loss)

            if debug and valloader:
                print("Val Loss: {:.3f}, timestamp: {}".format(epoch_val_loss, datetime.now()))

        return train_loss, val_loss

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))
