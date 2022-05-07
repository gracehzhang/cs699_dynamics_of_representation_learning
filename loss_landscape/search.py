"""
Script to train a neural network:
    currently supports training resnet for CIFAR-10 with and w/o skip connections

    Also does additional things that we may need for visualizing loss landscapes, such as using
      frequent directions or storing models during the executions etc.
   This has limited functionality or options, e.g., you do not have options to switch datasets
     or architecture too much.
"""

import argparse
import logging
import os
import pprint
import time
from tqdm import tqdm
import pickle


import dill
import numpy.random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter

# from utils.evaluations import get_loss_value
from utils.linear_algebra import FrequentDirectionAccountant
from utils.nn_manipulation import count_params, flatten_grads, flatten_params, apply_params
from utils.reproducibility import set_seed

from utils.SWAModel import AveragedModel
from torch.optim.swa_utils import SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR
# from utils.resnet import get_resnet

### RL Stuff
import gym
import d4rl
from utils.BCModel import MLP
from q_learning import Q_Learning

# "Fixed" hyperparameters
# In the resnet paper they train for ~90 epoch before reducing LR, then 45 and 45 epochs.
# We use 100-50-50 schedule here.
LR = 1e-3
DATA_FOLDER = "../data/"

class RLDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, obs, acs, next_obs, rews, dones):
        'Initialization'
        self.obs= obs
        self.acs = acs
        self.next_obs = next_obs
        self.rews = rews
        self.dones = dones

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.obs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ob = self.obs[index]
        ac = self.acs[index]
        next_ob = self.next_obs[index]
        rew = self.rews[index]
        done = self.dones[index]

        return ob,ac,next_ob,rew,done

def get_dataloader(batch_size, env, split=0.9, cap=100):
    """
        returns: cifar dataloader

    Arguments:
        batch_size:
        train_size: How many samples to use of train dataset?
        test_size: How many samples to use from test dataset?
        transform_train_data: If we should transform (random crop/flip etc) or not
    """

    dataset = d4rl.qlearning_dataset(env)
    # bc_dataset = env.get_dataset()
    ### door v0 dataset 50000
    end = min(len(dataset["observations"]), cap)
    split = int(end * split)
    train_dataset = RLDataset(dataset["observations"][:split], dataset["actions"][:split], dataset["next_observations"][:split], dataset["rewards"][:split], dataset["terminals"][:split])
    test_dataset = RLDataset(dataset["observations"][split:end], dataset["actions"][split:end], dataset["next_observations"][split:end], dataset["rewards"][split:end], dataset["terminals"][split:end])

    # Data loader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, test_loader

def evaluate_policy(model, env, num_episodes, epoch, save_dir, device):
    rews = []

    pbar = tqdm(range(num_episodes))
    for i in pbar:
        pbar.set_description("Evaluating ...")
        if i == num_episodes - 1:
            eval_env = gym.wrappers.record_video.RecordVideo(
                env,
                video_folder=f"{save_dir}/videos/",
                name_prefix=f"epoch_{epoch}",
            )
        else:
            eval_env = env
        # eval_env = env
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

    eval_env.close_video_recorder()
    return np.sum(rews) / num_episodes, len(rews)

def get_model_loss(model, test_loader, device):
    losses = []
    accs = []
    for i, (obs, acs, next_obs, rews, dones) in enumerate(test_loader):
        obs = obs.to(device)
        acs = acs.to(device)
        next_obs = next_obs.to(device)
        rews = rews.to(device)
        dones = dones.to(device)
        batch = {"observations": obs, "actions": acs, "next_observations": next_obs, "rewards": rews, "terminals": dones}
        loss, acc = model.compute_loss(batch)
        losses.append(loss.item())
        if acc is not None:
            accs.append(acc)

    if len(accs) > 0:
        acc = np.mean(accs)
    else:
        acc = np.nan
    return np.mean(losses), acc

def set_weights_by_directions(model, deltas, directions, weights, skip_bn_bias=False):
    changes = torch.zeros_like(directions[0])
    for x, direction in zip(deltas, directions):
        changes += direction * x

    apply_params(model, weights + changes, skip_bn_bias=skip_bn_bias)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--debug", action='store_true')
    parser.add_argument("--env", type=str, default="minigrid-fourrooms-v0")
    parser.add_argument("--seed", required=False, type=int, default=0)
    parser.add_argument(
        "--device", required=False, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--result_folder", "-r", required=True)
    parser.add_argument(
        "--mode", required=False, nargs="+", choices=["test", "train"], default=["test", "train"]
    )
    parser.add_argument(
        "--model", required=True, choices=["BC"]
    )

    # model related arguments
    parser.add_argument("--statefile", "-s", required=True)
    parser.add_argument("--direction_file", required=True)
    parser.add_argument("--num_directions", type=int, default=2)

    parser.add_argument("--batch_size", required=False, type=int, default=128)
    parser.add_argument("--num_eval_episodes", required=False, type=int, default=20)
    parser.add_argument(
        "--save_strategy", required=False, nargs="+", choices=["epoch", "init"],
        default=["epoch", "init"]
    )

    # cem arguments
    parser.add_argument("--num_iterations", required=False, type=int, default=10)
    parser.add_argument("--population", required=False, type=int, default=10)
    parser.add_argument("--sigma", required=False, type=float, default=1.0)

    args = parser.parse_args()

    # set up logging
    os.makedirs(f"{args.result_folder}/ckpt", exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    summary_writer = SummaryWriter(log_dir=args.result_folder)

    logger.info("Config:")
    logger.info(pprint.pformat(vars(args), indent=4))

    set_seed(args.seed)

    env = gym.make(args.env)
    env.seed(args.seed)

    # get dataset, just in case
    # train_loader, test_loader = get_dataloader(args.batch_size, env)

    # get model
    model = MLP(env)
    model.to(args.device)
    logger.info(f"using {args.model} with {count_params(model)} parameters")

    logger.debug(model)

    ## TO DO: Load policy

    logger.info(f"Loading model from {args.statefile}")
    state_dict = torch.load(args.statefile, pickle_module=dill, map_location=args.device)
    model.load_state_dict(state_dict)

    total_params = count_params(model)
    pretrained_weights = flatten_params(
        model, num_params=total_params
    ).to(args.device)

    ## TO DO: Load frequent directions

    logger.info(f"Loading directions from {args.direction_file}")
    temp = numpy.load(args.direction_file)
    directions = [torch.tensor(temp["direction{}".format(i+1)], device=args.device).float() for i in range(args.num_directions)]


    if "init" in args.save_strategy:
        torch.save(
            model.state_dict(), f"{args.result_folder}/ckpt/init_model.pt", pickle_module=dill
        )

    # training loop
    xcoords, ycoords  = [0], [0]
    searched_xcoords, searched_ycoords = [], []
    # total_x, total_y = 0, 0

    # coords = [[0] for _ in range(args.num_directions)]
    # searched_coords = [[] for _ in range(args.num_directions)]
    total_coords = np.zeros(args.num_directions)
    step = 0
    total_samples = 0
    total_episodes = 0
    stats = []
    # import ipdb; ipdb.set_trace()
    for epoch in range(args.num_iterations):
        population = []
        rewards = []
        # losses = []
        for i in range(args.population):
            ### To Do: Generate Population

            deltas = np.random.normal(loc=0, scale=args.sigma, size=args.num_directions)
            if i == 0:
                deltas = np.zeros_like(deltas)
            set_weights_by_directions(
                model, deltas, directions, pretrained_weights,
            )
            population.append(deltas)
            x, y = total_coords[:2] + deltas[:2]
            searched_xcoords.append(x)
            searched_ycoords.append(y)

            ## To Do: Evaluate fitness
            rew, num_samples = evaluate_policy(model, env, 20, f"iteration-{epoch}_x-{x}_y-{y}", f"{args.result_folder}/search-videos/", args.device)
            total_samples += num_samples
            total_episodes += 10
            rewards.append(rew)
            # losses.append(get_model_loss(model, train_loader, args.device)[0])

        ### To Do: Pick Elite

        elite_idx = np.argmax(rewards)
        deltas = population[elite_idx]
        set_weights_by_directions(
            model, deltas, directions, pretrained_weights,
        )
        total_params = count_params(model)
        pretrained_weights = flatten_params(
            model, num_params=total_params
        ).to(args.device)
        total_coords = total_coords + deltas
        xcoords.append(total_coords[0])
        ycoords.append(total_coords[1])

        ### To Do: Logging
        logger.info(f"CEM Iteration {epoch}, Best Fitness: {rewards[elite_idx]}")
        summary_writer.add_scalar("best_fitness", rewards[elite_idx], step)
        summary_writer.add_scalar("num_samples", total_samples, step)
        summary_writer.add_scalar("num_episodes", total_episodes, step)
        step += 1

        evalModel = model
        # Save the model checkpoint
        torch.save(
            evalModel.state_dict(), f'{args.result_folder}/ckpt/{epoch + 1}_model.pt',
            pickle_module=dill
        )



        eval, _ = evaluate_policy(evalModel, env, args.num_eval_episodes, epoch, args.result_folder, args.device)
        logger.info(f'Evaluating model on {args.num_eval_episodes} episodes: {eval}')
        summary_writer.add_scalar("test/return", eval, step)
        stats.append((total_episodes, eval))

    # save losses and accuracies evaluations
    logger.info("Saving results")
    numpy.savez(
        f"{args.result_folder}/CEM_proj.npz", xcoordinates=xcoords,
        ycoordinates=ycoords
    )
    numpy.savez(
        f"{args.result_folder}/CEM_sampled_proj.npz", xcoordinates=searched_xcoords,
        ycoordinates=searched_ycoords
    )

    # open a file, where you ant to store the data
    file = open(f"{args.result_folder}/data.pkl", 'wb')

    # dump information to that file
    pickle.dump(stats, file)

    # close the file
    file.close()
