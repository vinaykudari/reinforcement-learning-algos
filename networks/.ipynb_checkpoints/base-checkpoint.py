import torch
import torch.nn as nn


class BaseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

    # def _init_net(self, ip_dim, op_dim, hidden_layers):
    #     self.layers = []
    #     n = len(hidden_layers)
    #     for i, n_nodes in enumerate(hidden_layers):
    #         if i == 0:
    #             self.layers.append(nn.Linear(ip_dim, n_nodes))
    #         elif i < n-1:
    #             self.layers.append(nn.Linear(n_nodes, n_nodes))
    #         else:
    #             self.layers.append(nn.Linear(n_nodes, op_dim))

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))