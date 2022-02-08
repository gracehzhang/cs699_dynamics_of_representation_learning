import torch.nn as nn
import torch.nn.functional as F
import torch
import gym
import numpy as np
# model definition
class MLP(torch.nn.Module):
    # define model elements
    def __init__(self, env):
        super(MLP, self).__init__()
        self.env = env
        # state_dim = env.observation_space.shape[0]
        # check if environment action space is discrete
        if isinstance(env.action_space, gym.spaces.discrete.Discrete):
            action_dim = env.action_space.n
            self.action_type = "discrete"
        else:
            action_dim = env.action_space.shape[0]
            self.action_type = "continuous"


        if env.observation_space.shape == None:
            # print()
            state_dim = np.prod(env.observation_space['image'].shape) 
        else:
            state_dim = env.observation_space.shape[0]
        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = nn.Linear(256, action_dim)
        self.activation = nn.ReLU()
 
    # forward propagate input
    def forward(self, X):
        if self.action_type == "discrete":
            X = torch.flatten(X, start_dim=1)
        X = self.layer1(X)
        X = self.activation(X)
        X = self.layer2(X)
        return X

    def compute_action(self, X):
        if self.action_type =="discrete":
            out = self.forward(X)
            return torch.argmax(out, dim=1)
        else:
            out = self.forward(X)
            return out

    def compute_loss(self, inputs):
        out = self.forward(inputs['observations'])
        total = 0
        correct = 0
        if self.action_type =="discrete":
            loss = nn.CrossEntropyLoss()
            _, predicted = torch.max(out.data, 1)
            total += inputs['observations'].size(0)
            correct += (predicted == inputs['actions']).sum().item()
            # print(type(inputs['actions'][0]))
            return loss(out, inputs['actions'].long()), 100 * correct / total
        else:
            loss = nn.MSELoss()
            return loss(out, inputs['actions']), None




def get_model(model_string):
    if model_string == "BC":
        return MLP



