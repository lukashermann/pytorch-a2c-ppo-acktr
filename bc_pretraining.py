import copy
import glob
import os
import time
from collections import deque
import sys
import cv2
import json
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime
from PIL import Image, ImageEnhance
import matplotlib.colors
from a2c_ppo_acktr import algo
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.combi_policy import CombiPolicy
from a2c_ppo_acktr.storage import RolloutStorage, CombiRolloutStorage
from a2c_ppo_acktr.utils import get_vec_normalize, update_linear_schedule, update_linear_schedule_half, update_linear_schedule_less, update_sr_schedule
from a2c_ppo_acktr.visualize import visdom_plot
from gym_grasping.envs.grasping_env import GraspingEnv
from tensorboardX import SummaryWriter

device = torch.device("cuda:0")


def load_dataset(filename):
    data = np.load(filename)
    actions = torch.from_numpy(data['actions']).float().to(device)
    img_obs = data['img_obs']
    robot_state_obs = torch.from_numpy(data['robot_state_obs']).float().to(device)
    task_state_obs = torch.from_numpy(data['task_state_obs']).float().to(device)
    return actions, img_obs, robot_state_obs, task_state_obs


def process_imgs(img_obs):
    num_imgs = img_obs.shape[0]
    con = np.random.uniform(0.9, 1.1, size=num_imgs)
    bright = np.random.uniform(0.9, 1.1, size=num_imgs)
    col = np.random.uniform(0.8, 1.2, size=num_imgs)
    sharp = np.random.uniform(0.9, 1.1, size=num_imgs)
    blur = np.random.uniform(0, 1, size=num_imgs)
    hue = np.random.uniform(-0.03, 0.03, size=num_imgs)
    processed_imgs = []
    for i in range(num_imgs):
        img = Image.fromarray(img_obs[i])
        contrast = ImageEnhance.Contrast(img)
        img = contrast.enhance(con[i])
        brightness = ImageEnhance.Brightness(img)
        img = brightness.enhance(bright[i])
        color = ImageEnhance.Color(img)
        img = color.enhance(col[i])
        for i in range(5):
            sharpness = ImageEnhance.Sharpness(img)
            img = sharpness.enhance(sharp[i])
        blur_fac = int(blur[i]) * 2 + 1
        img = cv2.GaussianBlur(np.array(img), ksize=(blur_fac, blur_fac), sigmaX=0)
        img = img / 255
        img_hsv = matplotlib.colors.rgb_to_hsv(img)
        img_hsv[:, :, 0] += hue[i]
        img = matplotlib.colors.hsv_to_rgb(np.clip(img_hsv, 0, 1))
        img *= 255
        img = np.array(img, dtype=np.uint8)
        img = img.transpose(2, 0, 1)
        processed_imgs.append(img)
    processed_imgs = np.array(processed_imgs)
    img_obs_torch = torch.from_numpy(processed_imgs).float().to(device)
    return img_obs_torch


def pretrain(sysargs):
    args = get_args(sysargs[1:])
    training_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = os.path.join("/home/kuka/lang/robot/behavior_cloning/", training_name)
    train_filename = '/home/kuka/lang/robot/gym_grasping/gym_grasping/behavior_cloning/data/bc_100_train.npz'
    eval_filename = '/home/kuka/lang/robot/gym_grasping/gym_grasping/behavior_cloning/data/bc_100_eval.npz'
    os.makedirs(log_dir)
    tb_writer = SummaryWriter(log_dir=os.path.join(log_dir, "tb"))
    torch.set_num_threads(1)
    np.random.seed(1)
    num_epochs = args.num_bc_epochs
    lr = args.lr
    eps = args.eps
    minibatch_size = 10
    save_interval = args.save_interval
    recurrent_hidden_states = torch.zeros(minibatch_size).to(device)
    masks = torch.ones(minibatch_size).to(device)

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, None, args.add_timestep, device, False,
                         num_frame_stack=args.num_framestack, dont_normalize_obs=args.dont_normalize_obs)

    actor_critic = CombiPolicy(envs.observation_space, envs.action_space,
                               base_kwargs={'recurrent': args.recurrent_policy,
                                            'cnn_architecture': args.cnn_architecture},
                               network_architecture=args.network_architecture, share_layers=False)
    actor_critic.to(device)
    optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    train_actions, train_img_obs_np, train_robot_state_obs, train_task_state_obs = load_dataset(train_filename)
    eval_actions, eval_img_obs_np, eval_robot_state_obs, eval_task_state_obs = load_dataset(eval_filename)

    eval_recurrent_hidden_states = torch.zeros(eval_actions.size()[0]).to(device)
    eval_masks = torch.ones(eval_actions.size()[0]).to(device)
    mse = torch.nn.MSELoss()
    batch_size = len(train_actions)
    assert batch_size % minibatch_size == 0
    num_minibatches = batch_size // minibatch_size

    indices = np.random.permutation(batch_size)
    for epoch_idx in range(num_epochs):
        train_loss = []
        np.random.shuffle(indices)
        img_obs = process_imgs(train_img_obs_np)
        for minibatch_ids in np.split(indices, num_minibatches):
            _, pred_action, _, _ = actor_critic.act(
                {'img': img_obs[minibatch_ids],
                 'robot_state': train_robot_state_obs[minibatch_ids],
                 'task_state': train_task_state_obs[minibatch_ids]},
                recurrent_hidden_states,
                masks, deterministic=True)
            # print(pred_action)
            # print(actions[minibatch_ids])
            optimizer.zero_grad()
            loss = mse(pred_action, train_actions[minibatch_ids])
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                train_loss.append(loss.cpu().numpy())
        # eval
        with torch.no_grad():
            eval_img_obs = process_imgs(eval_img_obs_np)
            _, eval_pred_action, _, _ = actor_critic.act(
                {'img': eval_img_obs,
                 'robot_state': eval_robot_state_obs,
                 'task_state': eval_task_state_obs},
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)
            eval_loss = mse(eval_pred_action, eval_actions)
        tb_writer.add_scalar("eval_loss_step", eval_loss, epoch_idx * batch_size)
        tb_writer.add_scalar("eval_loss_epoch", eval_loss, epoch_idx)
        tb_writer.add_scalar("train_loss_step", np.mean(train_loss), epoch_idx * batch_size)
        tb_writer.add_scalar("train_loss_epoch", np.mean(train_loss), epoch_idx)
        print("iteration: {}, train_loss: {}, eval_loss: {}".format(epoch_idx, np.mean(train_loss), eval_loss))
        if epoch_idx != 0 and epoch_idx % save_interval == 0:
            save_model = [copy.deepcopy(actor_critic).cpu(), None, None]
            torch.save(save_model, os.path.join(log_dir, args.env_name + "_{}".format(epoch_idx) + ".pt"))


if __name__ == '__main__':
    pretrain(sys.argv)
