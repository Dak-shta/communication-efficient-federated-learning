from torch.utils.data import Subset
import numpy as np


def create_iid_partitions(dataset, num_clients=5):
    """
    Split the dataset randomly and equally among clients.
    Each client receives approximately the same distribution.
    """

    num_samples = len(dataset)
    indices = np.random.permutation(num_samples)

    samples_per_client = num_samples // num_clients

    client_datasets = []

    for i in range(num_clients):
        start = i * samples_per_client

        if i == num_clients - 1:
            end = num_samples
        else:
            end = (i + 1) * samples_per_client

        client_indices = indices[start:end]

        client_datasets.append(
            Subset(dataset, client_indices)
        )

    return client_datasets

def create_non_iid_partitions(dataset, num_clients=5):
    """
    Create a simple class-based non-IID partition.

    Each client receives data primarily from two digit classes.
    """

    targets = np.array(dataset.targets)

    # Sort indices according to digit labels
    sorted_indices = np.argsort(targets)

    client_datasets = []

    # Divide the 10 classes into pairs:
    # Client 1 -> 0,1
    # Client 2 -> 2,3
    # Client 3 -> 4,5
    # Client 4 -> 6,7
    # Client 5 -> 8,9

    classes_per_client = 2

    for client_id in range(num_clients):

        selected_classes = range(
            client_id * classes_per_client,
            (client_id + 1) * classes_per_client
        )

        client_indices = []

        for class_id in selected_classes:

            class_indices = np.where(
                targets == class_id
            )[0]

            client_indices.extend(class_indices)

        client_datasets.append(
            Subset(dataset, client_indices)
        )

    return client_datasets