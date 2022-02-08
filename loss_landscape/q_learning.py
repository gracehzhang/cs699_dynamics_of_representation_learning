from timeit import repeat
import torch
import torch.nn as nn
import gym
import numpy as np

class Q_Learning(nn.Module):
	def __init__(self, env, gamma=0.99):
		super(Q_Learning, self).__init__()
		# check if observation space is image or vector
		if env.observation_space.shape == None:
			state_dim = np.prod(env.observation_space['image'].shape)
		else:
			state_dim = env.observation_space.shape[0]
		if isinstance(env.action_space, gym.spaces.discrete.Discrete):
			action_dim = env.action_space.n
			self.action_type = "discrete"
		else:
			action_dim = env.action_space.shape[0]
			self.action_type = "continuous"
		self.state_dim = state_dim
		self.action_space = env.action_space
		self.action_dim = action_dim
		self.gamma = gamma
		if self.action_type == "discrete":
			self.fc1 = nn.Linear(state_dim, 256)
			self.fc2 = nn.Linear(256, 256)
			self.fc3 = nn.Linear(256, action_dim)	
		else:
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
		repeat_factor = 100
		batch_size = next_obs.shape[0]
		for _ in range(repeat_factor):
			a_samples.append(np.repeat(np.expand_dims(self.action_space.sample(), 0), (batch_size, 1)))
		a_samples = torch.tensor(a_samples).float().unsqueeze(0).to(next_obs.device).unsqueeze(0)
		next_obs_repeated = next_obs.repeat_interleave(repeat_factor, 1)
		a_samples = a_samples.reshape(repeat_factor * batch_size, -1)
		q_sa_samples = self.forward(torch.cat((next_obs_repeated, a_samples), dim=1))
		q_sa_samples = q_sa_samples.reshape(batch_size, repeat_factor)
		return q_sa_samples

	def compute_action(self, obs):
		if self.action_type == "discrete":
			action = self.forward(obs).argmax(dim=1)
		else:
			action = torch.max(self.sample_continuous_actions(obs), dim=1)[0]
		return action

	def compute_loss(self, input_batch: dict):
		obs = input_batch['observations']
		action = input_batch['actions']
		reward = input_batch['rewards']
		next_obs = input_batch['next_observations']
		done = input_batch['terminals'].float()

		# compute Q(s,a)
		if len(obs.shape) > 2: 
			obs = obs.reshape(obs.shape[0], self.state_dim)
			next_obs = next_obs.reshape(next_obs.shape[0], self.state_dim)
		if len(action.shape) == 1:
			action = action.unsqueeze(-1)
		if self.action_type == "discrete":
			q_sa = self.forward(obs).gather(1, action.long())
			q_sa_target = reward + self.gamma * torch.max(self.forward(next_obs), dim=1)[0] * (1-done)
		else:
			s_a = torch.cat((obs, action), dim=1)
			q_sa = self.forward(s_a)
			q_sa_target = reward + self.gamma * torch.max(self.sample_continuous_actions(next_obs), dim=1)[0] * (1-done)

		return self.loss_fn(q_sa, q_sa_target), None