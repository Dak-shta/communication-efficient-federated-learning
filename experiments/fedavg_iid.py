import torch
from torch.utils.data import DataLoader
from utils.seed import set_seed

from models.cnn import SimpleCNN
from data.dataset import get_mnist
from clients.client import Client
from server.server import Server
from utils.data_partition import create_iid_partitions


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

    return 100 * correct / total


def main():
    set_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    # Load MNIST
    train_dataset, test_dataset, _, _ = get_mnist()

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False
    )

    # Number of simulated clients
    num_clients = 5

    # Create IID client datasets
    client_datasets = create_iid_partitions(
        train_dataset,
        num_clients=num_clients
    )

    # Create global model
    global_model = SimpleCNN().to(device)

    # Create server
    server = Server(
        global_model,
        device=device
    )

    # Create clients
    clients = [
        Client(
            dataset,
            batch_size=64,
            device=device
        )
        for dataset in client_datasets
    ]

    # Federated training
    num_rounds = 5
    accuracies=[]
    for round_num in range(num_rounds):

        print(f"\n--- Round {round_num + 1}/{num_rounds} ---")

        client_states = []
        client_sizes = []
       
        for client_id, client in enumerate(clients):

            print(f"Training Client {client_id + 1}")

            # Give client a copy of global model
            local_model = server.distribute_model()

            # Local training
            state, loss = client.train(
                local_model,
                epochs=1,
                lr=0.01
            )

            client_states.append(state)
            client_sizes.append(len(client.dataset))

        # FedAvg aggregation
        global_model = server.aggregate(
            client_states,
            client_sizes
        )

        # Evaluate global model
        accuracy = evaluate(
            global_model,
            test_loader,
            device
        )
        accuracies.append(accuracy)

        print(
            f"Global Model Accuracy: {accuracy:.2f}%"
        )
    print("\nFinal Accuracy Results:")
    for i, accuracy in enumerate(accuracies, start=1):
                    print(f"Round {i}: {accuracy:.2f}%")
        


if __name__ == "__main__":
    main()