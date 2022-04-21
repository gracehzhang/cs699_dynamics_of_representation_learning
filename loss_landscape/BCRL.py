# from stable_baselines import A2C
# from stable_baselines.common.cmd_util import make_atari_env
# from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines3.common.utils import set_random_seed
import numpy as np
import os
import torch
import gym
import pickle
import matplotlib.pyplot as plt
import os.path as osp
from stable_baselines3 import DQN
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
import random
from tqdm import tqdm
import argparse
# import d4rl
import d4rl_atari
# from generateQValsAtari import getDict, genQ


from typing import Callable
import torch
from torch import nn
from stable_baselines3 import PPO
from utils.BCModel import MLP
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gym
import d4rl



pretrained = True
envName = "halfcheetah-expert-v1"
backbonePath = "/lab/ssontakk/cs699_dynamics_of_representation_learning/loss_landscape/results_hc_expert/ckpt/99_model.pt"

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True
# def _make_env(rank, log_relative_path):

#     def _init():
#         env = gym.make("CartPole-v1")
#         env.seed(rank)
# #             env = Monitor(env, log_dir)
# #             task = task_generator(task_generator_id=task_name, dense_reward_weights=np.array([0,750, 0]),tool_block_mass=0.7, tool_block_size=0.03) #####Check reward weights np.array([250, 750, 100]) for pushing, np.array([750, 0, 100, 0, 250, 0, 0, 0]) for picking
# #             env = CausalWorld(task=task,
# #                               skip_frame=skip_frame,
# #                               enable_visualization=False,
# #                               seed=seed_num + rank,
# #                               max_episode_length=maximum_episode_length)
#         env = Monitor(env, osp.join(log_relative_path, '{}'.format(rank)))
#         return env

#     set_random_seed(rank)
#     return _init



class CustomBase(BaseFeaturesExtractor):
    # define model elements
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomBase, self).__init__(observation_space, features_dim)
        self.backboneNet = MLP(gym.make(envName))
        # if pretrained:
        #     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1")
        #     self.backboneNet.load_state_dict(torch.load(backbonePath))
        # self.backboneNet.layer2 = Identity()
    def forward(self, X):
        X = self.backboneNet(X)
        return X

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

# def my_loss(output, target):
#   if outp
#     loss = torch.mean((output - target)**2)
#     return loss


# def createData(eps, gamma):

#   exp = gym.make('pong-expert-v4', stack=True)
#   exp.reset() # (4, 84, 84)
#   exp = exp.get_dataset()
#   a = getDict(exp, eps, True)
#   b1 = genQ(a, eps, gamma)

#   exp = gym.make('pong-mixed-v4', stack=True)
#   exp.reset() # (4, 84, 84)
#   exp = exp.get_dataset()
#   a = getDict(exp, eps, True)
#   b2 = genQ(a, eps, gamma)

#   returnDic = {}
#   # print(len(b1['q_values']), len(b2['q_values']))
#   # print(b1['q_values'].shape, b2['q_values'].shape)

#   for key in b2.keys():
#     if isinstance(b1[key],(list)):
#       returnDic[key] = b1[key] + b2[key]
#     else:
#       returnDic[key] = np.concatenate((b1[key], b2[key]), axis=0)

#   return returnDic
#   # return b1

def main(args):
  for i in range(1):
    set_random_seed(i)
    log_dir = f"data/{envName}/BCBackboneRL/{i}/"
    os.makedirs(log_dir, exist_ok=True)
    # env = SubprocVecEnv([_make_env(rank=i, log_relative_path=log_dir) for i in range(10)])
    # Create and wrap the environment
    env = gym.make(envName)
    env = Monitor(env, log_dir)
    env.seed(i)
    
    # env = SubprocVecEnv([_make_env(rank=i, log_relative_path=log_dir) for i in range(10)])
    # Create and wrap the environment
    policy_kwargs = dict(
    features_extractor_class=CustomBase,
    features_extractor_kwargs=dict(features_dim=256),)
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    # Train the agent
    # time_steps = 100000
    # 
    if pretrained:
      model.policy.features_extractor.backboneNet.load_state_dict(torch.load(backbonePath))
    model.policy.features_extractor.backboneNet.layer2 = Identity()
    # print(model.policy.features_extractor.backboneNet)
    # print(model.get_parameters()['policy']['features_extractor.backboneNet.layer1.weight'])
    model.learn(total_timesteps=args.n_timesteps, callback=callback)
    # print()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # parser.add_argument("--batch_size", required=False, type=int, default=32)
  # parser.add_argument("--buffer_size", required=False, type=int, default=10000)
  # parser.add_argument("--exploration_final_eps", required=False, type=int, default=0.01)
  # parser.add_argument("--exploration_fraction", required=False, type=int, default=0.1)
  # parser.add_argument("--pretrained", required=False, type=bool, default=False)
  # parser.add_argument("--gradient_steps", required=False, type=int, default=1)
  # parser.add_argument("--learning_rate", required=False, type=int, default=0.0001)
  # parser.add_argument("--learning_starts", required=False, type=int, default=100000)
  parser.add_argument("--n_timesteps", required=False, type=int, default=2000000)
  # parser.add_argument("--optimize_memory_usage", required=False, type=bool, default=True)
  # parser.add_argument("--policy", required=False, type=str, default='CnnPolicy')
  # parser.add_argument("--target_update_interval", required=False, type=int, default=1000)
  # parser.add_argument("--train_freq", required=False, type=int, default=4)
  # # parser.add_argument("--num_eval_episodes", required=False, type=int, default=10)
  # parser.add_argument(
  #     "--save_strategy", required=False, nargs="+", choices=["epoch", "init"],
  #     default=["epoch", "init"])
  # parser.add_argument("--envName", type=str, default="halfcheetah-expert-v1")
  # parser.add_argument("--backbonePath", required=False, type=str, default="/lab/ssontakk/cs699_dynamics_of_representation_learning/loss_landscape/")
  args = parser.parse_args()
  main(args)