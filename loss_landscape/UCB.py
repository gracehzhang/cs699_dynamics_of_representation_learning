from copyreg import pickle
import numpy as np
import torch.nn as nn
import torch
import argparse
from utils.BCModel import MLPEnsemble
import d4rl
import tqdm
import gym
import os


def simple_evaluate_policy(model, env, num_episodes, device):
    rews = []

    pbar = tqdm(range(num_episodes))
    for i in pbar:
        pbar.set_description("Evaluating ...")
        eval_env = env
        ob = eval_env.reset()

        while True:
            ### ASDF make sure this works for BC
            if isinstance(ob, dict):
                ob = ob["image"]
            ob = torch.Tensor(np.expand_dims(ob, axis=0)).to(device)
            ac = model.compute_action(ob)[0].cpu().detach().numpy()
            ob, rew, done, _ = eval_env.step(ac)
            rews.append(rew)

            if done:
                break
    return np.sum(rews) / num_episodes


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


def main(args):
    cum_rewards = []
    env = gym.make(args.env)
    arms = MLPEnsemble(env, os.path.join(args.load_dir, args.env))
    UCB = UCBPolicy(arms, args.epsilon_greedy).to(args.device)
    for episode in args.n_episodes:
        UCB.pick_new_arm(episode)
        reward = simple_evaluate_policy(UCB, env, 1, args.device)
        if episode % 10 == 0:
            eval_reward = simple_evaluate_policy(UCB, env, 20, args.device)
            print(f"Episode: {episode}, Eval reward: {eval_reward}")
            cum_rewards.append((episode, eval_reward))
        UCB.update_experimental_means(reward)
    save_dir = os.path.join(
        args.save_dir, args.env, f"UCB_eps-{args.epsilon_greedy}_run-{args.run_number}"
    )
    os.makedirs(save_dir, exist_ok=True)
    torch.save(UCB.state_dict(), os.path.join(save_dir, "model.pt"))
    pickle.dump(cum_rewards, open(os.path.join(save_dir, "cum_rewards.pkl"), "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes", required=False, type=int, default=4000)
    parser.add_argument("--load_dir", required=True, type=str, default=None)
    parser.add_argument("--env", required=True, type=str, default=None)
    parser.add_argument("--epsilon_greedy", required=False, type=bool, default=False)
    parser.add_argument("--save_dir", required=True, type=str, default=None)
    parser.add_argument("--run_number", required=True, type=int, default=None)
    parser.add_argument("--device", required=False, type=str, default="cuda:0")
    args = parser.parse_args()
    main(args)
