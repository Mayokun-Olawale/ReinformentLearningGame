"""Categorical actor and scalar critic for the engine's structured state."""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, observation_size, action_size, hidden_size=128):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(observation_size, hidden_size), nn.Tanh(),
                                   nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.actor = nn.Linear(hidden_size, action_size)
        self.critic = nn.Linear(hidden_size, 1)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.orthogonal_(self.critic.weight, 1.0)

    def forward(self, obs):
        features = self.trunk(obs)
        return self.actor(features), self.critic(features).squeeze(-1)

    def distribution(self, obs):
        logits, values = self(obs)
        return Categorical(logits=logits), values
