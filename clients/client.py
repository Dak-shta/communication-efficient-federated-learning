import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class Client:

    def __init__(self, dataset, batch_size=64, device="cpu"):

        self.dataset = dataset
        self.device = device

        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )

    def train(self, model, epochs=1, lr=0.01):

        model = model.to(self.device)
        model.train()

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.SGD(
            model.parameters(),
            lr=lr
        )

        total_loss = 0

        for _ in range(epochs):

            for images, labels in self.loader:

                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                outputs = model(images)

                loss = criterion(outputs, labels)

                loss.backward()

                optimizer.step()

                total_loss += loss.item()

        return model.state_dict(), total_loss