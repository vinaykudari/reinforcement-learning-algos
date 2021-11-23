import torch.nn as nn
import torch.nn.functional as F

from networks.base import BaseNetwork


class ValueNetwork(BaseNetwork):
    def __init__(
        self,
        state_dim,
        op_dim=1,
        hidden_layers=[256],
    ):
        super().__init__()
        # evaluates only the state
        # self._init_net(
        #     ip_dim=state_dim,
        #     op_dim=op_dim,
        #     hidden_layers=hidden_layers,
        # )

        self.fc1 = nn.Linear(state_dim, hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[0])
        self.out = nn.Linear(hidden_layers[0], 1)

    def forward(self, state):
        x = state
        x.to(self.device)

#         n = len(self.layers)

#         for i, layer in enumerate(self.layers):
#             if i < n:
#                 x = F.relu(layer(x))
#             else:
#                 x = layer(x)

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        # return state value
        return x