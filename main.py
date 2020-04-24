import copy
import datetime
import functools
import glob
import json
import os
import sys
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
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

import a2c_ppo_acktr
from a2c_ppo_acktr import algo
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.augmentation.augmenters import TransformsAugmenter
from a2c_ppo_acktr.combi_policy import CombiPolicy
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.combi_policy import CombiPolicy
from a2c_ppo_acktr.storage import RolloutStorage, CombiRolloutStorage
from a2c_ppo_acktr.utils import get_vec_normalize, update_linear_schedule, \
    update_linear_schedule_half, \
    update_linear_schedule_less, update_sr_schedule
from a2c_ppo_acktr.visualize import visdom_plot

from a2c_ppo_acktr.augmentation import augmenters
from a2c_ppo_acktr.augmentation.datasets import ObsDataset

from gym_grasping.envs.grasping_env import GraspingEnv
from tensorboardX import SummaryWriter
from a2c_ppo_acktr.augmentation.datasets import ObsDataset


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if len(obj) == 1:
                return obj[0]
            else:
                return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


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
    args.training_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M") + "_seed-{}".format(
        args.seed)
    if args.tag is not None:
        args.log_dir = os.path.join(args.root_dir, args.tag, args.training_name)
    else:
        args.log_dir = os.path.join(args.root_dir, args.training_name)

    args.save_dir = os.path.join(args.log_dir, "save")
    args.tensorboard = True
    if args.tensorboard:
        tb_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, "tb"))
        if args.save_train_images or args.save_eval_images:
            tb_writer_img = SummaryWriter(log_dir=os.path.join(args.log_dir, "tb"),
                                          filename_suffix="_img")

    ###
    # date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") + "_seed-" + str(args.seed)
    # run_name = date + "_exps-{}".format(num_augmentation_steps) + "_runs-{}".format(
    #     num_runs_per_experiment) + "_eps-{}".format(num_episodes_per_run) + tag
    # log_dir = os.path.join(log_base_dir, experiment_name, run_name)
    #
    # if use_tensorboard:
    #     tb_writer = SummaryWriter(log_dir=os.path.join(log_dir, "tb"))
    #     tb_writer.flush()
    #
    #     with open(os.path.join(log_dir, "hyperparams.txt"), "w") as file:
    #         file.write("python " + " ".join(sys.argv) + "\n")
    #         for arg in vars(args):
    #             file.write(str(arg) + ' ' + str(getattr(args, arg)) + '\n')

    ##

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

    curriculum_log_file = open(os.path.join(args.log_dir, "curriculum_log.json"), 'w')

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    if args.vis:
        from visdom import Visdom
        viz = Visdom(port=args.port)
        win = None
    # if args.no_curriculum:
    #     curr_args = None
    # else:
    #     curr_args = {"num_updates": num_updates,
    #                  "num_update_steps" : args.num_steps,
    #                  "desired_rew_region": args.desired_rew_region,
    #                  "incr": args.incr}
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, args.add_timestep, device,
                         allow_early_resets=False, curr_args=None,
                         num_frame_stack=args.num_framestack,
                         dont_normalize_obs=args.dont_normalize_obs,
                         env_params_sampler_dict=args.env_params_file)

    if args.snapshot is None:
        if args.combi_policy:
            base_kwargs = {'recurrent': args.recurrent_policy,
                           'cnn_architecture': args.cnn_architecture}
            if args.augmentation_use_cnn_loss:
                base_kwargs["return_cnn_output"] = True
            actor_critic = CombiPolicy(envs.observation_space, envs.action_space,
                                       base_kwargs=base_kwargs,
                                       network_architecture=args.network_architecture,
                                       share_layers=False)
        else:
            actor_critic = Policy(envs.observation_space.shape, envs.action_space,
                                  base_kwargs={'recurrent': args.recurrent_policy})
    else:
        load_data = torch.load(args.snapshot)
        actor_critic, _, _ = load_data
    actor_critic.to(device)

    augmenter = None
    augmentation_loss_weight = args.augmentation_loss_weight

    if args.augmentation_loss_weight_function_params:
        params = np.load(args.augmentation_loss_weight_function_params)
        augmentation_loss_weight_function = np.poly1d(params)
    else:
        augmentation_loss_weight_function = None

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr,
                               eps=args.eps, alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        dataloader = None
        if args.use_augmentation_loss:
            assert args.augmenter is not None

            # TODO: refactor augmenter args to be cli arguments
            if args.augmentation_dataset_folder is not None:
                dataset = ObsDataset(root_folder=args.augmentation_dataset_folder,
                                     one_file_per_step=True)

                # Batch size is depending on the rollout for the agent algorithm (defined later)
                if args.augmentation_dataloader_batch_size == 'same':
                    data_loader_batch_size = (
                                                     args.num_processes * args.num_steps) // args.num_mini_batch
                else:
                    data_loader_batch_size = int(args.augmentation_dataloader_batch_size)
                dataloader = DataLoader(dataset, batch_size=data_loader_batch_size, shuffle=True,
                                        num_workers=0, drop_last=True)

            augmenter = augmenters.get_augmenter_by_name(args.augmenter,
                                                         augmenter_args={
                                                             "use_cnn_loss": args.augmentation_use_cnn_loss,
                                                             "clip_aug_actions": args.augmentation_clip_aug_actions,
                                                             "transformer": "color_transformer",
                                                             "transformer_args": {
                                                                 "hue": 0}})
        agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                         args.value_loss_coef, args.entropy_coef, lr=args.lr,
                         eps=args.eps,
                         max_grad_norm=args.max_grad_norm,
                         augmenter=augmenter,
                         return_images=args.save_train_images,
                         augmentation_data_loader=dataloader,
                         augmentation_loss_weight=augmentation_loss_weight,
                         augmentation_loss_weight_function=augmentation_loss_weight_function)

    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, acktr=True)

    obs = envs.reset()
    if args.combi_policy:
        rollouts = CombiRolloutStorage(args.num_steps, args.num_processes,
                                       envs.observation_space, envs.action_space,
                                       actor_critic.recurrent_hidden_state_size)
        rollouts.obs_img[0].copy_(obs['img'])
        rollouts.obs_robot[0].copy_(obs['robot_state'])
        rollouts.obs_task[0].copy_(obs['task_state'])
    else:
        rollouts = RolloutStorage(args.num_steps, args.num_processes,
                                  envs.observation_space.shape, envs.action_space,
                                  actor_critic.recurrent_hidden_state_size)
        rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=args.rew_q_len)
    curr_episode_rewards = deque(maxlen=args.rew_q_len)
    reg_episode_rewards = deque(maxlen=args.rew_q_len)
    eval_reg_episode_rewards = deque(maxlen=32)
    train_success = deque(maxlen=args.rew_q_len)
    curr_success = deque(maxlen=args.rew_q_len)
    reg_success = deque(maxlen=args.rew_q_len)
    difficulty_cur = 0
    difficulty_reg = 0
    desired_rew_region = (args.desired_rew_region_lo, args.desired_rew_region_hi)
    incr = args.incr
    eval_episode_rewards = []

    start = time.time()

    # print(actor_critic.state_dict()['base.cnn.0.weight'])
    # actor_critic.state_dict()['base.cnn.0.weight'].fill_(1)
    # print(actor_critic.state_dict()['base.cnn.0.weight'])
    # exit()

    for j in tqdm(range(num_updates), desc="Updates"):
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
        elif args.use_sr_schedule and args.algo == "ppo":
            update_sr_schedule(agent.optimizer, np.mean(eval_episode_rewards) if len(
                eval_episode_rewards) > 1 else 0,
                               args.lr)
        if args.algo == 'ppo' and args.use_linear_clip_decay:
            agent.clip_param = args.clip_param * (1 - j / float(num_updates))
        elif args.algo == 'ppo' and args.use_linear_clip_decay_less:
            agent.clip_param = args.clip_param * (1 - j / float(2 * num_updates))

        for step in tqdm(range(args.num_steps), desc="Env steps"):
            # Sample actions
            with torch.no_grad():
                if args.combi_policy:
                    act_args = ({'img': rollouts.obs_img[step],
                                 'robot_state': rollouts.obs_robot[step],
                                 'task_state': rollouts.obs_task[step]},
                                rollouts.recurrent_hidden_states[step],
                                rollouts.masks[step])
                    if args.augmentation_use_cnn_loss:
                        value, action, action_log_prob, recurrent_hidden_states, cnn_output = actor_critic.act(
                            *act_args)
                    else:
                        value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(*act_args)
                else:
                    value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])

            data = {'update_step': j,
                    'num_updates': num_updates,
                    'eprewmean': np.mean(episode_rewards) if len(episode_rewards) > 1 else None,
                    'curr_eprewmean': np.mean(curr_episode_rewards) if len(
                        curr_episode_rewards) > 1 else 0,
                    'eval_eprewmean': np.mean(eval_episode_rewards) if len(
                        eval_episode_rewards) > 1 else 0,
                    'reg_eprewmean': np.mean(reg_episode_rewards) if len(
                        reg_episode_rewards) > 1 else 0,
                    'curr_success_rate': np.mean(curr_success) if len(curr_success) > 1 else 0,
                    'reg_success_rate': np.mean(reg_success) if len(reg_success) > 1 else 0,
                    'eval_reg_eprewmean': np.mean(eval_reg_episode_rewards) if len(
                        eval_reg_episode_rewards) > 1 else 0,
                    'difficulty_cur': difficulty_cur,
                    'difficulty_reg': difficulty_reg}

            # Observe reward and next obs
            obs, reward, done, infos = envs.step_with_curriculum_reset(action, data)
            # obs, reward, done, infos = envs.step(action)

            # visualize env 0
            # img = obs['img'].cpu().numpy()[0, ::-1, :, :].transpose((1, 2, 0)).astype(np.uint8)
            # # print(obs['robot_state'].cpu().numpy()[0])
            # cv2.imshow("win", cv2.resize(img, (300, 300)))
            # k = cv2.waitKey(10) % 256
            # if k == ord('a'):
            #     difficulty_cur += 0.1
            #     difficulty_cur = np.clip(difficulty_cur, 0, 1)
            #     print(difficulty_cur)
            # if done[0]:
            #     print(reward)

            for i, info in enumerate(infos):
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    train_success.append(float(info['task_success']))
                    if 'reset_info' in info.keys() and info['reset_info'] == 'curriculum':
                        # if i == 0: print("curriculum")
                        num_resets += 1
                        curr_episode_rewards.append(info['episode']['r'])
                        curr_success.append(float(info['task_success']))
                    elif 'reset_info' in info.keys() and info['reset_info'] == 'regular':
                        # if i == 0: print("regular")
                        num_regular_resets += 1
                        num_resets += 1
                        reg_episode_rewards.append(info['episode']['r'])
                        eval_reg_episode_rewards.append(info['episode']['r'])
                        reg_success.append(float(info['task_success']))
                if 'episode_info' in info.keys():
                    info['episode_info']['env'] = i
                    info['episode_info']['difficulty_cur'] = difficulty_cur
                    info['episode_info']['difficulty_reg'] = difficulty_reg
                    info['episode_info']['reward'] = reward[i].numpy()[0]
                    info['episode_info']['progress'] = j / num_updates
                    json.dump(info['episode_info'], curriculum_log_file, cls=NumpyEncoder)
                    curriculum_log_file.write('\n')
                    curriculum_log_file.flush()

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                       for done_ in done])
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward,
                            masks)

        if args.adaptive_curriculum and len(curr_success) > 1:
            if np.mean(curr_success) > desired_rew_region[1]:
                difficulty_cur += incr
            elif np.mean(curr_success) < desired_rew_region[0]:
                difficulty_cur -= incr
            difficulty_cur = np.clip(difficulty_cur, 0, 1)
        if args.adaptive_curriculum and len(reg_success) > 1:
            if np.mean(reg_success) > desired_rew_region[1]:
                difficulty_reg += incr
            elif np.mean(reg_success) < desired_rew_region[0]:
                difficulty_reg -= incr
            difficulty_reg = np.clip(difficulty_reg, 0, 1)
        with torch.no_grad():
            if args.combi_policy:
                next_value = actor_critic.get_value({'img': rollouts.obs_img[-1],
                                                     'robot_state': rollouts.obs_robot[-1],
                                                     'task_state': rollouts.obs_task[-1]},
                                                    rollouts.recurrent_hidden_states[-1],
                                                    rollouts.masks[-1]).detach()
            else:
                next_value = actor_critic.get_value(rollouts.obs[-1],
                                                    rollouts.recurrent_hidden_states[-1],
                                                    rollouts.masks[-1]).detach()

        total_num_steps = (j + 1) * args.num_processes * args.num_steps

        # Add current step information to agent as some agent need this info for calculation of losses
        if hasattr(agent, 'set_current_num_steps'):
            agent.set_current_num_steps(total_num_steps)

        # Update agent
        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)
        value_loss, action_loss, dist_entropy, additional_data_after_update = agent.update(
            rollouts)
        rollouts.after_update()

        action_loss_original = additional_data_after_update[
            "action_loss_original"] if "action_loss_original" in additional_data_after_update else None
        action_loss_aug = additional_data_after_update[
            "action_loss_aug"] if "action_loss_aug" in additional_data_after_update else None
        action_loss_aug_weighted = additional_data_after_update[
            "action_loss_aug_weighted"] if "action_loss_aug_weighted" in additional_data_after_update else None
        grad_norm = additional_data_after_update[
            "grad_norm"] if "grad_norm" in additional_data_after_update else None
        action_max_value_aug = additional_data_after_update[
            "action_max_value_aug"] if "action_max_value_aug" in additional_data_after_update else None
        action_max_value = additional_data_after_update[
            "action_max_value"] if "action_max_value" in additional_data_after_update else None
        agent_train_images = additional_data_after_update[
            "images"] if "images" in additional_data_after_update else None

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

            if args.combi_policy:
                save_model = [save_model,
                              getattr(get_vec_normalize(envs), 'ob_robot_rms', None),
                              getattr(get_vec_normalize(envs), 'ob_task_rms', None)]
            else:
                save_model = [save_model,
                              getattr(get_vec_normalize(envs), 'ob_rms', None)]

            torch.save(save_model, os.path.join(save_path, args.env_name + "_" + str(j) + ".pt"))

            # Save visualization of last training step
            if args.save_train_images and agent_train_images is not None:
                images = agent_train_images["obs"]
                for image_idx, image in enumerate(images):
                    tb_writer_img.add_images("Policy Update {}".format(j), image / 255.0, image_idx)

                if args.use_augmentation_loss:
                    augmentation_obs_keys = ["obs_aug_orig", "obs_aug_augmented"]
                    aug_images = map(functools.partial(torch.cat, dim=3), zip(
                        *[agent_train_images[obs_key] for obs_key in augmentation_obs_keys]))
                    for image_idx, image in enumerate(aug_images):
                        # TODO: Change obs range to [0, 1]
                        tb_writer_img.add_images("Policy Update Augmentation{}".format(j),
                                                 image / 255.0, image_idx)


        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()
            log_output = "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward " \
                         "{:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".format(j,
                                                                                total_num_steps,
                                                                                int(
                                                                                    total_num_steps / (
                                                                                            end - start)),
                                                                                len(
                                                                                    episode_rewards),
                                                                                np.mean(
                                                                                    episode_rewards),
                                                                                np.median(
                                                                                    episode_rewards),
                                                                                np.min(
                                                                                    episode_rewards),
                                                                                np.max(
                                                                                    episode_rewards),
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
                args.env_name, args.seed + args.num_processes * j, args.num_processes,
                args.gamma, eval_log_dir, args.add_timestep, device, True,
                num_frame_stack=args.num_framestack,
                dont_normalize_obs=args.dont_normalize_obs)

            vec_norm = get_vec_normalize(eval_envs)
            if vec_norm is not None:
                vec_norm.eval()
                if args.combi_policy:
                    vec_norm.ob_robot_rms = get_vec_normalize(envs).ob_robot_rms
                    vec_norm.ob_task_rms = get_vec_normalize(envs).ob_task_rms
                else:
                    vec_norm.ob_rms = get_vec_normalize(envs).ob_rms

            eval_episode_rewards = []

            obs = eval_envs.reset()
            eval_recurrent_hidden_states = torch.zeros(args.num_processes,
                                                       actor_critic.recurrent_hidden_state_size,
                                                       device=device)
            eval_masks = torch.zeros(args.num_processes, 1, device=device)

            save_cnt = 0
            if args.save_eval_images and j % 300 == 0:
                os.mkdir(os.path.join(eval_log_dir, "iter_{}".format(j)))
            while len(eval_episode_rewards) <= eval_steps:
                with torch.no_grad():
                    if args.augmentation_use_cnn_loss:
                        _, action, _, eval_recurrent_hidden_states, _ = actor_critic.act(
                            obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)
                    else:
                        _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                            obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

                # Obser reward and next obs
                obs, reward, done, infos = eval_envs.step(action)
                if args.save_eval_images and j % 300 == 0 and save_cnt < 150:
                    img = obs['img'].cpu().numpy()[0, ::-1, :, :].transpose((1, 2, 0)).astype(
                        np.uint8)
                    cv2.imwrite(
                        os.path.join(eval_log_dir, "iter_{}/img_{}.png".format(j, save_cnt)), img)

                    # Tensorboard expects images to be in range [0, 1] for FloatTensor
                    # TODO: Change obs range to [0, 1]
                    tb_writer_img.add_images("eval_" + str(j), obs['img'].cpu() / 255.0, save_cnt,
                                             dataformats='NCHW')
                    save_cnt += 1

                eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                                for done_ in done])
                for info in infos:
                    if 'episode' in info.keys():
                        eval_episode_rewards.append(info['episode']['r'])
                        eval_reg_episode_rewards.append(info['episode']['r'])

            eval_envs.close()
            if args.tensorboard:
                tb_writer.add_scalar("eval_eprewmean_updates", np.mean(eval_episode_rewards), j)
                tb_writer.add_scalar("eval_eprewmean_steps", np.mean(eval_episode_rewards),
                                     total_num_steps)
                tb_writer.add_scalar("eval_success_rate",
                                     np.mean(np.array(eval_episode_rewards) > 0).astype(np.float),
                                     total_num_steps)
                tb_writer.flush()

            eval_log_output = "\nEvaluation using {} episodes: mean reward {:.5f}\n\n".format(
                len(eval_episode_rewards),
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
            tb_writer.add_scalar("training_success_rate", np.mean(train_success), total_num_steps)
            tb_writer.add_scalar("curr_success_rate", np.mean(curr_success), total_num_steps)
            tb_writer.add_scalar("reg_success_rate", np.mean(reg_success), total_num_steps)
            tb_writer.add_scalar("eprewmedian_steps", np.median(episode_rewards), total_num_steps)
            tb_writer.add_scalar("difficulty_cur", difficulty_cur, total_num_steps)
            tb_writer.add_scalar("difficulty_reg", difficulty_reg, total_num_steps)
            tb_writer.add_scalar("dist_entropy", dist_entropy, total_num_steps)
            tb_writer.add_scalar("action_loss_sum", action_loss, total_num_steps)
            tb_writer.add_scalar("action_loss_original", action_loss_original, total_num_steps)
            tb_writer.add_scalar("action_loss_augmented", action_loss_aug, total_num_steps)
            tb_writer.add_scalar("action_loss_augmented_weighted", action_loss_aug_weighted, total_num_steps)
            tb_writer.add_scalar("action_max_value_aug", action_max_value_aug, total_num_steps)
            tb_writer.add_scalar("action_max_value", action_max_value, total_num_steps)
            tb_writer.add_scalar("value_loss", value_loss, total_num_steps)
            tb_writer.add_scalar("grad_norm", grad_norm, total_num_steps)

        if args.tensorboard and len(curr_episode_rewards) > 1:
            tb_writer.add_scalar("curr_eprewmean_steps", np.mean(curr_episode_rewards),
                                 total_num_steps)
            tb_writer.add_scalar("regular_resets_ratio",
                                 num_regular_resets / num_resets if num_resets > 0 else 0,
                                 total_num_steps)
        if args.tensorboard and len(reg_episode_rewards) > 1:
            tb_writer.add_scalar("reg_eprewmean_steps", np.mean(reg_episode_rewards),
                                 total_num_steps)
        if args.tensorboard:
            tb_writer.flush()

    if args.tensorboard:
        tb_writer.close()
        if args.save_eval_images or args.save_train_images:
            tb_writer_img.close()


if __name__ == "__main__":
    train(sys.argv)
