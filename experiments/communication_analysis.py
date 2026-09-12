from models.cnn import SimpleCNN
from utils.communication import (
    get_model_size_bytes,
    bytes_to_mb,
    calculate_communication_cost
)


def main():
    num_clients = 5
    num_rounds = 5

    model = SimpleCNN()

    model_size_bytes = get_model_size_bytes(model.state_dict())
    model_size_mb = bytes_to_mb(model_size_bytes)

    print("===== Communication Analysis =====")
    print(f"Number of clients: {num_clients}")
    print(f"Number of rounds: {num_rounds}")
    print(f"Model size: {model_size_mb:.2f} MB")

    communication_per_round = calculate_communication_cost(
        model_size_bytes,
        num_clients,
        1,
        both_directions=True
    )

    print(
        f"Communication per round: "
        f"{bytes_to_mb(communication_per_round):.2f} MB"
    )

    print("\nCumulative Communication:")

    for round_number in range(1, num_rounds + 1):
        total_communication = calculate_communication_cost(
            model_size_bytes,
            num_clients,
            round_number,
            both_directions=True
        )

        print(
            f"Round {round_number}: "
            f"{bytes_to_mb(total_communication):.2f} MB"
        )


if __name__ == "__main__":
    main()