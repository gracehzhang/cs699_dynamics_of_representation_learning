import torch.nn as nn
import torch.nn.functional as F
import torch
import gym
import numpy as np
import os
import glob

# model definition
class MLP(nn.Module):
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
            state_dim = np.prod(env.observation_space["image"].shape)
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
        if self.action_type == "discrete":
            out = self.forward(X)
            return torch.argmax(out, dim=1)
        else:
            out = self.forward(X)
            return out

    def compute_loss(self, inputs):
        out = self.forward(inputs["observations"])
        total = 0
        correct = 0
        if self.action_type == "discrete":
            loss = nn.CrossEntropyLoss()
            _, predicted = torch.max(out.data, 1)
            total += inputs["observations"].size(0)
            correct += (predicted == inputs["actions"]).sum().item()
            # print(type(inputs['actions'][0]))
            return loss(out, inputs["actions"].long()), 100 * correct / total
        else:
            loss = nn.MSELoss()
            return loss(out, inputs["actions"]), None


def get_model(model_string):
    if model_string == "BC":
        return MLP


class MLPEnsemble(nn.Module):
    def __init__(self, env, ensemble_save_name):
        super(MLPEnsemble, self).__init__()
        self.env = env
        num_models = len(glob.glob(ensemble_save_name + "*model*"))
        self.models = nn.ModuleList()
        for i in range(num_models):
            self.models.add_module(str(i), MLP(env))
        self.ensemble_save_name = ensemble_save_name
        # now load the actual models
        for i in range(1, num_models + 1):
            model_name = ensemble_save_name + "-model-" + str(i)
            if os.path.isdir(model_name):
                self.models[i].load_state_dict(
                    torch.load(os.path.join(model_name, "ckpt", "99_model.pt"))
                )
            else:
                print("Model " + str(i) + " not found")

    def forward(self, X):
        avg_out = torch.zeros(X.size(0), self.models[0].layer2.out_features)
        for model in self.models:
            out = model.forward(X)
            avg_out += out
        avg_out /= len(self.models)
        return avg_out

    def compute_action(self, X):
        out = self.forward(X)
        if self.env.action_space.shape == None:
            return torch.argmax(out, dim=1)
        else:
            return out

    def compute_loss(self, inputs):
        out = self.forward(inputs["observations"])
        total = 0
        correct = 0
        if self.env.action_space.shape == None:
            loss = nn.CrossEntropyLoss()
            _, predicted = torch.max(out.data, 1)
            total += inputs["observations"].size(0)
            correct += (predicted == inputs["actions"]).sum().item()
            # print(type(inputs['actions'][0]))
            return loss(out, inputs["actions"].long()), 100 * correct / total
        else:
            loss = nn.MSELoss()
            return loss(out, inputs["actions"]), None

    def forward_one(self, X, i):
        return self.models[i].forward(X)

    def __len__(self):
        return len(self.models)
