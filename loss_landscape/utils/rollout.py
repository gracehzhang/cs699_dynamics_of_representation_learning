import gym
import torch
import numpy as np

def rollout(env, policy, epsilon=0.0):
	obs = env.reset()
	done = False
	reward = 0
	while not done:
		if np.random.rand() < epsilon:
			action = env.action_space.sample()
		else:
			action = policy.compute_action(obs)
		obs, r, done, _ = env.step(action)
		reward += r
	return reward