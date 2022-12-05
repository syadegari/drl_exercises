import torch
import torch.nn as nn


class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        super().__init__()
        torch.manual_seed(seed=seed)
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, fc1_units),
            nn.ReLU(),
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, action_size)
        )

    def forward(self, state):
        return self.feature_layer(state)


class QNetwork_Dueling(nn.Module):

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        super().__init__()
        torch.manual_seed(seed=seed)
        self.feature = nn.Sequential(
            nn.Linear(state_size, fc1_units),
            nn.ReLU(),
            nn.Linear(fc1_units, fc2_units),
        )
        self.value = nn.Sequential(
            nn.Linear(fc2_units, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(fc2_units, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, action_size)
        )

    def forward(self, state):
        feature_vector = self.feature(state)
        value_vector = self.value(feature_vector)
        advantage_vector = self.advantage(feature_vector)
        return value_vector + advantage_vector - advantage_vector.mean(dim=1).unsqueeze(1)

                
