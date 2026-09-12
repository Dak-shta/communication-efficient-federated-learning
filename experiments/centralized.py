import torch
import torch.nn as nn
import torch.optim as optim

from models.cnn import SimpleCNN
from data.dataset import get_mnist


def train(model, train_loader, device, epochs=5):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01
    )

    model.train()

    for epoch in range(epochs):

        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        accuracy = 100 * correct / total

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {total_loss / len(train_loader):.4f} "
            f"Accuracy: {accuracy:.2f}%"
        )


def evaluate(model, test_loader, device):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total

    print(f"Test Accuracy: {accuracy:.2f}%")

    return accuracy


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    train_loader, test_loader = get_mnist()

    model = SimpleCNN().to(device)

    train(
        model,
        train_loader,
        device,
        epochs=5
    )

    evaluate(
        model,
        test_loader,
        device
    )


if __name__ == "__main__":
    main()