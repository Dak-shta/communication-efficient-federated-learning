def get_model_size_bytes(state_dict):
    """
    Calculate the size of a model state_dict in bytes.
    Assumes tensors are transmitted in their current dtype.
    """
    total_bytes = 0

    for tensor in state_dict.values():
        total_bytes += tensor.numel() * tensor.element_size()

    return total_bytes


def bytes_to_mb(num_bytes):
    return num_bytes / (1024 ** 2)


def calculate_communication_cost(
    model_size_bytes,
    num_clients,
    num_rounds,
    both_directions=True
):
    """
    Calculate total communication cost.

    both_directions=True:
        Server -> clients + clients -> server

    both_directions=False:
        Only clients -> server
    """

    transmissions_per_round = num_clients

    if both_directions:
        transmissions_per_round *= 2

    total_bytes = (
        model_size_bytes
        * transmissions_per_round
        * num_rounds
    )

    return total_bytes