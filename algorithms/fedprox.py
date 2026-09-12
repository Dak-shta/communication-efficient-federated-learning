import torch
import torch.nn as nn
import torch.optim as optim


def fedprox_train(
    model,
    train_loader,
    global_state,
    device="cpu",
    epochs=1,
    lr=0.01,
    mu=0.01
):
    """
    Train a client model using the FedProx objective.

    global_state contains the parameters of the global model
    before local training begins.
    """

    model = model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=lr
    )

    # Keep a copy of the global parameters fixed during
    # local training.
    global_params = {
        name: param.detach().clone().to(device)
        for name, param in model.named_parameters()
    }

    total_loss = 0

    for _ in range(epochs):

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            classification_loss = criterion(
                outputs,
                labels
            )

            proximal_term = 0.0

            for name, param in model.named_parameters():

                proximal_term += torch.sum(
                    (param - global_params[name]) ** 2
                )

            loss = (
                classification_loss
                + (mu / 2) * proximal_term
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

    return model.state_dict(), total_loss