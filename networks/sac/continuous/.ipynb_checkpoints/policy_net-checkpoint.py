import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from networks.base import BaseNetwork


class PolicyNetwork(BaseNetwork):
    def __init__(
        self,
        state_dim,
        action_dim,
        eps,
        max_act,
        hidden_layers=[64],
        sigma_min=-20,
        sigma_max=2,
    ):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.eps = eps
        self.max_act = max_act

        self.fc1 = nn.Linear(state_dim, hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[0])
        self.out = nn.Linear(hidden_layers[0], 2*action_dim)

    def forward(self, state):
        x = state
        x.to(self.device)

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        mu, sigma = torch.chunk(x, 2, dim=-1)
        sigma = torch.clamp(
            sigma,
            min=self.sigma_min,
            max=self.sigma_max,
        )

        return mu, sigma

    def sample(self, state, add_noise=True):
        mu, log_sigma = self.forward(state)
        sigma = log_sigma.exp()
        probs = Normal(mu, sigma)

        if add_noise:
            actions = probs.rsample()
        else:
            actions = probs.sample()

        action = (torch.tanh(actions) * torch.tensor(self.max_act)).to(self.device)
        log_probs = probs.log_prob(actions)
        log_probs -= torch.log(1 - action.pow(2) + self.eps)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs