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
from utils.nn_manipulation import count_params, flatten_grads
from utils.reproducibility import set_seed
# from utils.resnet import get_resnet

### RL Stuff
import gym
import d4rl
from utils.BCModel import MLP
from q_learning import Q_Learning

# "Fixed" hyperparameters
NUM_EPOCHS = 100
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

def get_dataloader(batch_size, env, split=0.9):
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
    split = int(len(dataset["observations"]) * split)
    train_dataset = RLDataset(dataset["observations"][:split], dataset["actions"][:split], dataset["next_observations"][:split], dataset["rewards"][:split], dataset["terminals"][:split])
    test_dataset = RLDataset(dataset["observations"][split:], dataset["actions"][split:], dataset["next_observations"][split:], dataset["rewards"][split:], dataset["terminals"][split:])

    # Data loader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, test_loader

def evaluate_policy(model, env, num_episodes, epoch, save_dir):
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
        ob = eval_env.reset()

        while True:
            ### ASDF make sure this works for BC
            if isinstance(ob, dict):
                ob = ob["image"]
            ob = torch.Tensor(np.expand_dims(ob, axis=0)).to(model.device)
            ac = model.compute_action(ob)[0].cpu().detach().numpy()
            ob, rew, done, _ = eval_env.step(ac)
            rews.append(rew)

            if done:
                break
    eval_env.close_video_recorder()
    return np.sum(rews) / num_episodes

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

    # model related arguments
    parser.add_argument("--statefile", "-s", required=False, default=None)
    parser.add_argument(
        "--model", required=True, choices=["BC", "Q"]
    )
    parser.add_argument("--remove_skip_connections", action="store_true", default=False)
    parser.add_argument(
        "--skip_bn_bias", action="store_true",
        help="whether to skip considering bias and batch norm params or not, Li et al do not consider bias and batch norm params"
    )

    parser.add_argument("--batch_size", required=False, type=int, default=128)
    parser.add_argument("--num_eval_episodes", required=False, type=int, default=10)
    parser.add_argument(
        "--save_strategy", required=False, nargs="+", choices=["epoch", "init"],
        default=["epoch", "init"]
    )

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

    # get dataset
    train_loader, test_loader = get_dataloader(args.batch_size, env)

    # get model
    if args.model == "BC":
        model = MLP(env)
    elif args.model == "Q":
        model = Q_Learning(env)

    model.to(args.device)
    logger.info(f"using {args.model} with {count_params(model)} parameters")

    logger.debug(model)

    # we can try computing principal directions from some specific training rounds only
    total_params = count_params(model, skip_bn_bias=args.skip_bn_bias)
    fd = FrequentDirectionAccountant(k=2, l=10, n=total_params, device=args.device)
    # frequent direction for last 10 epoch
    fd_last_10 = FrequentDirectionAccountant(k=2, l=10, n=total_params, device=args.device)
    # frequent direction for last 1 epoch
    fd_last_1 = FrequentDirectionAccountant(k=2, l=10, n=total_params, device=args.device)

    # use the same setup as He et al., 2015 (resnet)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer, lr_lambda=lambda x: 1 if x < 100 else (0.1 if x < 150 else 0.01)
    )

    if "init" in args.save_strategy:
        torch.save(
            model.state_dict(), f"{args.result_folder}/ckpt/init_model.pt", pickle_module=dill
        )

    # training loop
    # we pass flattened gradients to the FrequentDirectionAccountant before clearing the grad buffer
    total_step = len(train_loader) * NUM_EPOCHS
    step = 0
    direction_time = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        for i, (obs,acs,next_obs,rews,dones) in enumerate(train_loader):
            obs = obs.to(args.device)
            acs = acs.to(args.device)
            next_obs = next_obs.to(args.device)
            rews = rews.to(args.device)
            dones = dones.to(args.device)

            batch = {"observations": obs, "actions": acs, "next_observations": next_obs, "rewards": rews, "terminals": dones}
            # Forward pass
            loss, _ = model.compute_loss(batch)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # get gradient and send it to the accountant
            start = time.time()
            fd.update(flatten_grads(model, total_params, skip_bn_bias=args.skip_bn_bias))
            direction_time += time.time() - start

            if epoch >= NUM_EPOCHS - 10:
                fd_last_10.update(
                    flatten_grads(model, total_params, skip_bn_bias=args.skip_bn_bias)
                )
            if epoch >= NUM_EPOCHS - 1:
                fd_last_1.update(
                    flatten_grads(model, total_params, skip_bn_bias=args.skip_bn_bias)
                )

            summary_writer.add_scalar("train/loss", loss.item(), step)
            step += 1

            if step % 100 == 0:
                logger.info(
                    f"Epoch [{epoch}/{NUM_EPOCHS}], Step [{step}/{total_step}] Loss: {loss.item():.4f}"
                )

        scheduler.step()

        # Save the model checkpoint
        if "epoch" in args.save_strategy:
            torch.save(
                model.state_dict(), f'{args.result_folder}/ckpt/{epoch + 1}_model.pt',
                pickle_module=dill
            )

        loss, acc = get_model_loss(model, test_loader, args.device)

        logger.info(f'Loss of the model on the test data: {loss}')
        summary_writer.add_scalar("test/loss", loss, step)
        logger.info(f'Accuracy of the model on the test data: {acc}%')
        summary_writer.add_scalar("test/acc", acc, step)

        eval = evaluate_policy(model, env, args.num_eval_episodes, epoch, args.result_folder)
        logger.info(f'Evaluating model on {args.num_eval_episodes} episodes: {eval}')
        summary_writer.add_scalar("test/return", eval, step)

    logger.info(f"Time to computer frequent directions {direction_time} s")

    logger.info(f"fd was updated for {fd.step} steps")
    logger.info(f"fd_last_10 was updated for {fd_last_10.step} steps")
    logger.info(f"fd_last_1 was updated for {fd_last_1.step} steps")

    # save the frequent_direction buffers and principal directions
    buffer = fd.get_current_buffer()
    directions = fd.get_current_directions()
    directions = directions.cpu().data.numpy()

    numpy.savez(
        f"{args.result_folder}/buffer.npy",
        buffer=buffer.cpu().data.numpy(), direction1=directions[0], direction2=directions[1]
    )

    # save the frequent_direction buffer
    buffer = fd_last_10.get_current_buffer()
    directions = fd_last_10.get_current_directions()
    directions = directions.cpu().data.numpy()

    numpy.savez(
        f"{args.result_folder}/buffer_last_10.npy",
        buffer=buffer.cpu().data.numpy(), direction1=directions[0], direction2=directions[1]
    )

    # save the frequent_direction buffer
    buffer = fd_last_1.get_current_buffer()
    directions = fd_last_1.get_current_directions()
    directions = directions.cpu().data.numpy()

    numpy.savez(
        f"{args.result_folder}/buffer_last_1.npy",
        buffer=buffer.cpu().data.numpy(), direction1=directions[0], direction2=directions[1]
    )
