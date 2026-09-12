import copy
import torch


class Server:

    def __init__(self, global_model, device="cpu"):
        self.global_model = global_model
        self.device = device

    def distribute_model(self):
        """
        Create independent copies of the global model
        for each selected client.
        """
        return copy.deepcopy(self.global_model)

    def aggregate(self, client_states, client_sizes):
        """
        FedAvg aggregation.

        Each client's model is weighted according to
        the number of training samples it owns.
        """

        global_state = self.global_model.state_dict()

        total_samples = sum(client_sizes)

        for key in global_state.keys():

            global_state[key] = sum(
                client_states[i][key].float()
                * (client_sizes[i] / total_samples)
                for i in range(len(client_states))
            )

        self.global_model.load_state_dict(global_state)

        return self.global_model