import torch.nn as nn
import torch.nn.functional as F
import torch
# model definition
class MLP(torch.nn.Module):
    # define model elements
    def __init__(self, env, n_inputs, discrete=False):
        super(MLP, self).__init__()
        self.env = env
        state_dim = env.observation_space.shape[0]
        # check if environment action space is discrete
        if isinstance(env.action_space, gym.spaces.discrete.Discrete):
            action_dim = env.action_space.n
            self.action_type = "discrete"
        else:
            action_dim = env.action_space.shape[0]
            self.action_type = "continuous"

        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = Linear(256, action_dim)
        self.activation = nn.ReLU()
 
    # forward propagate input
    def forward(self, X):
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

    def compute_loss(self, inputs):
        out = self.forward(inputs['obs'])
        if self.action_type =="discrete":
            loss = nn.BCEWithLogitsLoss()
            return loss(out, inputs['actions'])
        else:
            loss = nn.MSELoss()
            return loss(out, inputs['actions'])




def get_model(model_string):
    if model_string == "BC":
        return MLP