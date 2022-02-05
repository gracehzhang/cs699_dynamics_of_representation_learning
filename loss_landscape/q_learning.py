import torch
import torch.nn as nn
import gym
import numpy as np

class Q_Learning(nn.Module):
	def __init__(self, env, gamma=0.99):
		super(Q_Learning, self).__init__()
		# check if observation space is image or vector
		if len(env.observation_space.shape) == None:
			state_dim = np.prod(env.observation_space.shape) 
		else:
			state_dim = env.observation_space.shape[0]
		if isinstance(env.action_space, gym.spaces.discrete.Discrete):
			action_dim = env.action_space.n
			self.action_type = "discrete"
		else:
			action_dim = env.action_space.shape[0]
			self.action_type = "continuous"
		self.action_space = env.action_space
		self.action_dim = action_dim
		self.gamma = gamma
		self.fc1 = nn.Linear(state_dim + action_dim, 256)
		self.fc2 = nn.Linear(256, 256)
		self.fc3 = nn.Linear(256, 1)	
		self.relu = nn.ReLU()
		self.loss_fn = nn.MSELoss()

	def forward(self, x):
		x = self.relu(self.fc1(x))
		x = self.relu(self.fc2(x))
		x = self.fc3(x)
		return x

	def sample_continuous_actions(self, next_obs):
		a_samples = []
		for i in range(100):
			a_samples.append(self.action_space.sample())
		a_samples = torch.tensor(a_samples).float()
		q_sa_samples = self.forward(torch.cat((next_obs, a_samples), dim=1))
		return q_sa_samples

	def compute_action(self, obs):
		if self.action_type == "discrete":
			action = self.forward(obs).argmax(dim=1)
		else:
			action = self.forward(obs).max(dim=1)[1]
		return action

	def compute_loss(self, input_batch: dict):
		obs = input_batch['observations']
		action = input_batch['actions']
		reward = input_batch['rewards']
		next_obs = input_batch['next_observations']
		done = input_batch['terminals']

		# compute Q(s,a)
		s_a = torch.cat((obs, action), dim=1)
		q_sa = self.forward(s_a)
		if self.action_type == "discrete":
			q_sa_target = reward + self.gamma * torch.max(self.forward(next_obs), dim=1)[0] * (1-done)
		else:
			q_sa_target = reward + self.gamma * torch.max(self.sample_continuous_actions(next_obs), dim=1)[0] * (1-done)

		return self.loss_fn(q_sa, q_sa_target)	
