import copy
import glob
import os
import time
from collections import deque
import sys
import cv2

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime

from a2c_ppo_acktr import algo
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.utils import get_vec_normalize, update_linear_schedule, update_linear_schedule_half, update_linear_schedule_less
from a2c_ppo_acktr.visualize import visdom_plot
from gym_grasping.envs.grasping_env import GraspingEnv
from tensorboardX import SummaryWriter


def train(sysargs):
    args = get_args(sysargs[1:])

    assert args.algo in ['a2c', 'ppo', 'acktr']
    if args.recurrent_policy:
        assert args.algo in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'

    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # args.root_dir = "/home/kuka/lang/robot/training_logs"
    args.training_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    args.log_dir = os.path.join(args.root_dir, args.training_name)
    args.save_dir = os.path.join(args.log_dir, "save")
    args.tensorboard = True
    if args.tensorboard:
        tb_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, "tb"))

    try:
        os.makedirs(args.log_dir)
    except OSError:
        files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)

    eval_log_dir = os.path.join(args.log_dir, "_eval")

    try:
        os.makedirs(eval_log_dir)
    except OSError:
        files = glob.glob(os.path.join(eval_log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)

    with open(os.path.join(args.log_dir, "hyperparams.txt"), "w") as file:
        file.write("python " + " ".join(sysargs) + "\n")
        for arg in vars(args):
            file.write(str(arg) + ' ' + str(getattr(args, arg)) + '\n')

    log_file = open(os.path.join(args.log_dir, "log.txt"), "wt")

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    if args.vis:
        from visdom import Visdom
        viz = Visdom(port=args.port)
        win = None

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, args.add_timestep, device, False)

    actor_critic = Policy(envs.observation_space.shape, envs.action_space,
                          base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr,
                               eps=args.eps, alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                         args.value_loss_coef, args.entropy_coef, lr=args.lr,
                         eps=args.eps,
                         max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, acktr=True)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=20)
    curr_episode_rewards = deque(maxlen=20)
    reg_episode_rewards = deque(maxlen=20)
    eval_reg_episode_rewards = deque(maxlen=32)
    difficulty = 0
    desired_rew_region = args.desired_rew_region
    incr = args.incr
    eval_episode_rewards = []

    start = time.time()
    for j in range(num_updates):
        num_regular_resets = 0
        num_resets = 0
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            if args.algo == "acktr":
                # use optimizer's learning rate since it's hard-coded in kfac.py
                update_linear_schedule(agent.optimizer, j, num_updates, agent.optimizer.lr)
            else:
                update_linear_schedule(agent.optimizer, j, num_updates, args.lr)
        elif args.use_linear_lr_decay_less and args.algo == "ppo":
            update_linear_schedule_less(agent.optimizer, j, num_updates, args.lr)
        elif args.use_linear_lr_decay_half and args.algo == "ppo":
            update_linear_schedule_half(agent.optimizer, j, num_updates, args.lr)
        if args.algo == 'ppo' and args.use_linear_clip_decay:
            agent.clip_param = args.clip_param * (1 - j / float(num_updates))
        elif args.algo == 'ppo' and args.use_linear_clip_decay_less:
            agent.clip_param = args.clip_param * (1 - j / float(2 * num_updates))

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step],
                    rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            data = {'update_step': j,
                    'num_updates': num_updates,
                    'eprewmean': np.mean(episode_rewards) if len(episode_rewards) > 1 else None,
                    'curr_eprewmean': np.mean(curr_episode_rewards) if len(curr_episode_rewards) > 1 else 0,
                    'eval_eprewmean': np.mean(eval_episode_rewards) if len(eval_episode_rewards) > 1 else 0,
                    'reg_eprewmean': np.mean(reg_episode_rewards) if len(reg_episode_rewards) > 1 else 0,
                    'eval_reg_eprewmean': np.mean(eval_reg_episode_rewards) if len(eval_reg_episode_rewards) > 1 else 0,
                    'difficulty': difficulty}
            # Obser reward and next obs
            obs, reward, done, infos = envs.step_with_curriculum_reset(action, data)

            # visualize env 0
            # img = obs.cpu().numpy()[0, ::-1, :, :].transpose((1, 2, 0)).astype(np.uint8)
            # cv2.imshow("win", cv2.resize(img, (300,300)))
            # cv2.waitKey(1)
            # if done[0]:
            #     print(step)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    if 'reset_info' in info.keys() and info['reset_info'] == 'curriculum':
                        num_resets += 1
                        curr_episode_rewards.append(info['episode']['r'])
                    elif 'reset_info' in info.keys() and info['reset_info'] == 'regular':
                        num_regular_resets += 1
                        num_resets += 1
                        reg_episode_rewards.append(info['episode']['r'])
                        eval_reg_episode_rewards.append(info['episode']['r'])
            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                       for done_ in done])
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks)

        if args.adaptive_curriculum and len(curr_episode_rewards) > 1:
            if np.mean(curr_episode_rewards) > desired_rew_region[1]:
                difficulty += incr
            elif np.mean(curr_episode_rewards) < desired_rew_region[0]:
                difficulty -= incr
            difficulty = np.clip(difficulty, 0, 1)
        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1],
                                                rollouts.recurrent_hidden_states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0 or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_model,
                          getattr(get_vec_normalize(envs), 'ob_rms', None)]

            torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))

        total_num_steps = (j + 1) * args.num_processes * args.num_steps

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()
            log_output = "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward " \
                         "{:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".format(j,
                                                                                total_num_steps,
                                                                                int(total_num_steps / (end - start)),
                                                                                len(episode_rewards),
                                                                                np.mean(episode_rewards),
                                                                                np.median(episode_rewards),
                                                                                np.min(episode_rewards),
                                                                                np.max(episode_rewards),
                                                                                dist_entropy,
                                                                                value_loss,
                                                                                action_loss)
            print(log_output)
            log_file.write(log_output)
            log_file.flush()
        if (args.eval_interval is not None
                and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            eval_steps = 32 if j < num_updates - 1 else 100

            eval_envs = make_vec_envs(
                args.env_name, args.seed + args.num_processes, args.num_processes,
                args.gamma, eval_log_dir, args.add_timestep, device, True)

            vec_norm = get_vec_normalize(eval_envs)
            if vec_norm is not None:
                vec_norm.eval()
                vec_norm.ob_rms = get_vec_normalize(envs).ob_rms

            eval_episode_rewards = []

            obs = eval_envs.reset()
            eval_recurrent_hidden_states = torch.zeros(args.num_processes,
                                                       actor_critic.recurrent_hidden_state_size, device=device)
            eval_masks = torch.zeros(args.num_processes, 1, device=device)

            while len(eval_episode_rewards) <= eval_steps:
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

                # Obser reward and next obs
                obs, reward, done, infos = eval_envs.step(action)

                eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                                for done_ in done])
                for info in infos:
                    if 'episode' in info.keys():
                        eval_episode_rewards.append(info['episode']['r'])
                        eval_reg_episode_rewards.append(info['episode']['r'])

            eval_envs.close()
            if args.tensorboard:
                tb_writer.add_scalar("eval_eprewmean_updates", np.mean(eval_episode_rewards), j)
                tb_writer.add_scalar("eval_eprewmean_steps", np.mean(eval_episode_rewards), total_num_steps)

            eval_log_output = "\nEvaluation using {} episodes: mean reward {:.5f}\n\n".format(len(eval_episode_rewards),
                                                                                              np.mean(
                                                                                                  eval_episode_rewards))
            print()
            print(eval_log_output)
            print()
            log_file.write(eval_log_output)
            log_file.flush()

        if args.vis and j % args.vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                win = visdom_plot(viz, win, args.log_dir, args.training_name,
                                  args.algo, args.num_env_steps)
            except IOError:
                pass

        if args.tensorboard and len(episode_rewards) > 1:
            tb_writer.add_scalar("eprewmean_updates", np.mean(episode_rewards), j)
            tb_writer.add_scalar("eprewmean_steps", np.mean(episode_rewards), total_num_steps)
            tb_writer.add_scalar("eprewmedian_steps", np.median(episode_rewards), total_num_steps)
            tb_writer.add_scalar("difficulty", difficulty, total_num_steps)
            tb_writer.add_scalar("dist_entropy", dist_entropy, total_num_steps)
            tb_writer.add_scalar("action_loss", action_loss, total_num_steps)
            tb_writer.add_scalar("value_loss", value_loss, total_num_steps)
        if args.tensorboard and len(curr_episode_rewards) > 1:
            tb_writer.add_scalar("curr_eprewmean_steps", np.mean(curr_episode_rewards), total_num_steps)
            tb_writer.add_scalar("regular_resets_ratio", num_regular_resets / num_resets if num_resets > 0 else 0, total_num_steps)
        if args.tensorboard and len(reg_episode_rewards) > 1:
            tb_writer.add_scalar("reg_eprewmean_steps", np.mean(reg_episode_rewards), total_num_steps)

    if args.tensorboard:
        tb_writer.close()


if __name__ == "__main__":
    train(sys.argv)
