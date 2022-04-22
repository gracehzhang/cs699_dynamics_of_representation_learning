from importlib.metadata import requires
from turtle import forward
import numpy as np
import torch.nn as nn
import torch


class UCBPolicy(nn.Module):
    def __init__(self, arms, epsilon_greedy=False) -> None:
        super().__init__()
        self.arms = arms
        self.experimental_means = nn.parameter.Parameter(
            torch.zeros(len(arms)), requires_grad=False
        )
        self.num_times_picked = nn.parameter.Parameter(
            torch.zeros(len(arms)), requires_grad=False
        )
        self.upper_bounds = nn.parameter.Parameter(
            torch.zeros(len(arms)), requires_grad=False
        )
        self.currently_picked_arm = 0
        self.epsilon_greedy = epsilon_greedy

    def forward(self, observation):
        return self.arms[self.currently_picked_arm].compute_action(observation)

    def pick_new_arm(self, t):
        if not self.epsilon_greedy:
            if torch.any(self.num_times_picked == 0):
                self.currently_picked_arm = torch.argmax(self.num_times_picked == 0)
            else:
                self.upper_bounds = 2 * torch.log(t) / self.num_times_picked
                self.currently_picked_arm = torch.argmax(
                    self.experimental_means + self.upper_bounds
                )
        else:
            if np.random.rand() < 0.1:
                self.currently_picked_arm = np.random.randint(0, len(self.arms))
            else:
                self.currently_picked_arm = torch.argmax(self.experimental_means)
        self.num_times_picked[self.currently_picked_arm] += 1

    def update_experimental_means(self, reward):
        self.experimental_means[self.currently_picked_arm] += (
            reward - self.experimental_means[self.currently_picked_arm]
        ) / self.num_times_picked[self.currently_picked_arm]
