import copy
from types import SimpleNamespace

import datetime
import functools
import glob
import json
import os
import sys
import shutil
import time
from collections import deque

import cv2
import numpy as np
import torch
import yaml

from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from a2c_ppo_acktr import algo
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.augmentation import augmenters
from a2c_ppo_acktr.augmentation.datasets import ObsDataset
from a2c_ppo_acktr.augmentation.weighting import FixedWeight, FunctionWeight, MovingAverageFromLossWeight
from a2c_ppo_acktr.combi_policy import CombiPolicy
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import CombiRolloutStorage, RolloutStorage
from a2c_ppo_acktr.utils import get_vec_normalize, update_linear_schedule, \
    update_linear_schedule_half, \
    update_linear_schedule_less, update_sr_schedule
from a2c_ppo_acktr.visualize import visdom_plot

# DO NOT REMOVE - importing this class also imports gym environments
from gym_grasping.envs.grasping_env import GraspingEnv


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


def validate_inputs(cfg: SimpleNamespace):
    """
    Validates the input configuration
    :param cfg: configuration to be validated
    """
    assert cfg.learning.rl.algo in ['a2c', 'ppo', 'acktr']
    if cfg.learning.rl.actor_critic.recurrent_policy:
        assert cfg.learning.rl.algo in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'

    # TODO: add check for augmenter: augmentation not implemented for a2c and ACKTR


def set_seeds(cfg: SimpleNamespace):
    """
    Sets seed in torch and cuda
    :param cfg:
    """
    torch.manual_seed(cfg.globals.seed)
    torch.cuda.manual_seed_all(cfg.globals.seed)


def setup_cuda(cfg: SimpleNamespace):
    """
    Sets up cuda-specific setings
    :param cfg:
    :returns device used for training
    """
    if cfg.globals.cuda_enabled and cfg.globals.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if cfg.globals.cuda_enabled else "cpu")
    return device


def get_training_name(cfg: SimpleNamespace):
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M") + "_seed-{}".format(cfg.globals.seed)


def get_log_dir(cfg: SimpleNamespace):
    if cfg.experiment.tag is not None:
        return os.path.join(cfg.experiment.root_dir, cfg.experiment.tag, cfg.training_name)
    else:
        return os.path.join(cfg.experiment.root_dir, cfg.training_name)


def setup_dirs_and_logging(cfg: SimpleNamespace):
    """
    Create directory for storing training results
    :param cfg:
    :return:
    """
    cfg.training_name = get_training_name(cfg)
    cfg.log_dir = get_log_dir(cfg)
    cfg.save_dir = os.path.join(cfg.log_dir, "save")
    cfg.eval_log_dir = os.path.join(cfg.log_dir, "_eval")

    cfg.tensorboard = True  # TODO: Add param
    if cfg.tensorboard:
        tb_writer = SummaryWriter(log_dir=os.path.join(cfg.log_dir, "tb"))
        if cfg.experiment.save_train_images or cfg.experiment.save_eval_images:
            tb_writer_img = SummaryWriter(log_dir=os.path.join(cfg.log_dir, "tb"),
                                          filename_suffix="_imgs")
        else:
            tb_writer_img = None
    try:
        os.makedirs(cfg.log_dir)
    except OSError:
        files = glob.glob(os.path.join(cfg.log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)

    try:
        os.makedirs(cfg.eval_log_dir)
    except OSError:
        files = glob.glob(os.path.join(cfg.eval_log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)

    log_file = open(os.path.join(cfg.log_dir, "log.txt"), "wt")

    # TODO: Remove?
    # curriculum_log_file = open(os.path.join(cfg.log_dir, "curriculum_log.json"), 'w')

    # Copy original configuration if present to new location
    shutil.copyfile(str(cfg.config[0]), os.path.join(cfg.log_dir, "config.yaml"))

    return tb_writer, tb_writer_img, log_file


def setup_visualization(cfg: SimpleNamespace):
    if cfg.experiment.vis:
        from visdom import Visdom
        viz = Visdom(port=cfg.experiment.vis_port)
        win = None
        return viz, win
    else:
        return None, None


def get_curriculum_args(cfg: SimpleNamespace, num_updates, tb_writer):
    curr_args = None

    if cfg.learning.curriculum.enable:
        desired_rew_region = (
            cfg.learning.curriculum.desired_rew_region_lo, cfg.learning.curriculum.desired_rew_region_hi)
        curr_args = {"num_updates": num_updates,
                     "num_update_steps": cfg.globals.num_steps,
                     "desired_rew_region": desired_rew_region,
                     "incr": cfg.learning.curriculum.incr,
                     "tb_writer": tb_writer,
                     "num_processes": cfg.globals.num_processes}
    return curr_args


def get_env_args(cfg):
    env_kwargs = {"env_params_sampler_dict": cfg.env.params_file,
                  "data_folder_path": cfg.env.data_folder_path}
    return env_kwargs


def load_actor_critic_from_snapshot(snapshot_path):
    if os.path.isabs(snapshot_path):
        load_data = torch.load(snapshot_path)
    else:
        load_data = torch.load(os.path.join(os.getcwd(), snapshot_path))
    actor_critic, _, _ = load_data
    return actor_critic



def setup_actor_critic(cfg, envs):
    if cfg.learning.rl.actor_critic.combi_policy:
        base_kwargs = {'recurrent': cfg.learning.rl.actor_critic.recurrent_policy,
                       'cnn_architecture': cfg.learning.rl.actor_critic.cnn_architecture}
        if cfg.learning.consistency_loss.use_cnn_loss:
            base_kwargs["return_cnn_output"] = True

        actor_critic = CombiPolicy(envs.observation_space, envs.action_space,
                                   base_kwargs=base_kwargs,
                                   network_architecture=cfg.learning.rl.actor_critic.network_architecture,
                                   share_layers=False)
    else:
        actor_critic = Policy(envs.observation_space.shape, envs.action_space,
                              base_kwargs={'recurrent': cfg.learning.rl.actor_critic.recurrent_policy})

    if cfg.learning.rl.actor_critic.snapshot is not None:
        checkpoint = load_actor_critic_from_snapshot(cfg.learning.rl.actor_critic.snapshot)
        actor_critic.load_state_dict(checkpoint.state_dict())
    return actor_critic


def setup_consistency_loss(cfg):
    """
    Configures consistency loss parameters

    :returns
        augmentation_loss_weight
        augmentation_loss_weight_function
    """
    consistency_loss_weight = cfg.learning.consistency_loss.loss_weight

    if cfg.learning.consistency_loss.loss_weight_function_params is not None:
        weight_function_params = np.load(cfg.learning.consistency_loss.loss_weight_function_params)
        weight_function = np.poly1d(weight_function_params)
    else:
        weight_function = None

    if cfg.learning.consistency_loss.use_action_loss_as_weight:
        weighter = MovingAverageFromLossWeight(num_values=20, with_fixed_weight=consistency_loss_weight,
                                               with_loss_weight_function=weight_function)
    elif cfg.learning.consistency_loss.loss_weight_function_params:
        weighter = FunctionWeight(function=weight_function,
                                  with_fixed_weight=cfg.learning.consistency_loss.loss_weight)
    else:
        weighter = FixedWeight(weight=cfg.learning.consistency_loss.loss_weight)

    return weighter


def setup_a2c_agent(cfg, actor_critic):
    agent = algo.A2C_ACKTR(actor_critic, cfg.learning.rl.value_loss_coef,
                           cfg.learning.rl.entropy_coef, lr=cfg.learning.optimizer.lr,
                           eps=cfg.learning.optimizer.eps, alpha=cfg.learning.optimizer.alpha,
                           max_grad_norm=cfg.learning.rl.max_grad_norm)


def load_transfomer_and_args(args_path):
    with open(args_path) as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        tranformer_data = yaml.load(file, Loader=yaml.FullLoader)
        transformer_name = tranformer_data["transformer"]["type"]
        transformer_args = tranformer_data["transformer"]["args"]
        transformer = augmenters.get_transformer_by_name(transformer_name, **transformer_args)
        return transformer


def setup_augmenter(cfg):
    if cfg.learning.consistency_loss.transformer.args_path is not None:
        transformer = load_transfomer_and_args(cfg.learning.consistency_loss.transformer.args_path)
    else:
        # Default argument for backward compatibility
        transformer_args = {"hue": 0}
        transformer = augmenters.get_transformer_by_name("color_transformer", **transformer_args)

    augmenter = augmenters.get_augmenter_by_name(cfg.learning.consistency_loss.augmenter,
                                                 augmenter_args={
                                                     "use_cnn_loss": cfg.learning.consistency_loss.use_cnn_loss,
                                                     "clip_aug_actions": cfg.learning.consistency_loss.clip_aug_actions,
                                                     "transformer": transformer
                                                 })
    return augmenter


def setup_ppo_agent(cfg, actor_critic):
    dataloader = None
    augmenter = None
    augmentation_loss_weight = 0.0
    augmentation_loss_weight_function = None
    if cfg.learning.consistency_loss.enable:
        assert cfg.learning.consistency_loss.augmenter is not None

        augmenter = setup_augmenter(cfg)
        augmentation_loss_weight_function = setup_consistency_loss(cfg)

        if cfg.learning.consistency_loss.dataset_folder is not None:
            dataset = ObsDataset(root_folder=cfg.learning.consistency_loss.dataset_folder,
                                 one_file_per_step=True)

            # Batch size is depending on the rollout for the agent algorithm (defined later)
            if cfg.learning.consistency_loss.dataloader_batch_size == 'same':
                data_loader_batch_size = (
                                                 cfg.globals.num_processes * cfg.globals.num_steps) // cfg.globals.num_mini_batch
            else:
                data_loader_batch_size = int(cfg.learning.consistency_loss.dataloader_batch_size)
            dataloader = DataLoader(dataset, batch_size=data_loader_batch_size, shuffle=True,
                                    num_workers=0, drop_last=True)

    agent = algo.PPO(actor_critic, cfg.learning.rl.ppo.clip_param, cfg.learning.rl.ppo.epoch,
                     cfg.globals.num_mini_batch,
                     cfg.learning.rl.value_loss_coef, cfg.learning.rl.entropy_coef, lr=cfg.learning.optimizer.lr,
                     eps=cfg.learning.optimizer.eps,
                     max_grad_norm=cfg.learning.rl.max_grad_norm,
                     augmenter=augmenter,
                     return_images=cfg.experiment.save_train_images,
                     augmentation_data_loader=dataloader,
                     augmentation_loss_weight_function=augmentation_loss_weight_function)
    return agent


def _setup_rollouts(cfg, actor_critic, obs, envs):
    # === RL rollout setup
    if cfg.learning.rl.actor_critic.combi_policy:
        rollouts = CombiRolloutStorage(cfg.globals.num_steps, cfg.globals.num_processes,
                                       envs.observation_space, envs.action_space,
                                       actor_critic.recurrent_hidden_state_size)
        rollouts.obs_img[0].copy_(obs['img'])
        rollouts.obs_robot[0].copy_(obs['robot_state'])
        rollouts.obs_task[0].copy_(obs['task_state'])
    else:
        rollouts = RolloutStorage(cfg.globals.num_steps, cfg.globals.num_processes,
                                  envs.observation_space.shape, envs.action_space,
                                  actor_critic.recurrent_hidden_state_size)
        rollouts.obs[0].copy_(obs)

    return rollouts


def update_learning_rate(cfg, agent, update_step, num_updates, eval_episode_rewards):
    if cfg.learning.optimizer.use_linear_lr_decay:
        # decrease learning rate linearly
        if cfg.learning.rl.algo == "acktr":
            # use optimizer's learning rate since it's hard-coded in kfac.py
            update_linear_schedule(agent.optimizer, update_step, num_updates, agent.optimizer.lr)
        else:
            update_linear_schedule(agent.optimizer, update_step, num_updates, cfg.learning.optimizer.lr)

    elif cfg.learning.optimizer.use_linear_lr_decay_less and cfg.learning.rl.algo == "ppo":
        update_linear_schedule_less(agent.optimizer, update_step, num_updates, cfg.learning.optimizer.lr)

    elif cfg.learning.optimizer.use_linear_lr_decay_half and cfg.learning.rl.algo == "ppo":
        update_linear_schedule_half(agent.optimizer, update_step, num_updates, cfg.learning.optimizer.lr)

    elif cfg.learning.optimizer.use_sr_schedule and cfg.learning.rl.algo == "ppo":
        update_sr_schedule(agent.optimizer, np.mean(eval_episode_rewards) if len(
            eval_episode_rewards) > 1 else 0,
                           cfg.learning.optimizer.lr)

    if cfg.learning.rl.algo == 'ppo' and cfg.learning.rl.ppo.use_linear_clip_decay:
        agent.clip_param = cfg.learning.rl.ppo.clip_param * (1 - update_step / float(num_updates))
    elif cfg.learning.rl.algo == 'ppo' and cfg.learning.rl.ppo.use_linear_clip_decay_less:
        agent.clip_param = cfg.learning.rl.ppo.clip_param * (1 - update_step / float(2 * num_updates))


def save_model(cfg, envs, actor_critic, current_update_step):
    save_path = os.path.join(cfg.save_dir, cfg.learning.rl.algo)
    try:
        os.makedirs(save_path)
    except OSError:
        pass

    # A really ugly way to save a model to CPU
    save_model = actor_critic
    if cfg.globals.cuda_enabled:
        save_model = copy.deepcopy(actor_critic).cpu()

    if cfg.learning.rl.actor_critic.combi_policy:
        save_model = [save_model,
                      getattr(get_vec_normalize(envs), 'ob_robot_rms', None),
                      getattr(get_vec_normalize(envs), 'ob_task_rms', None)]
    else:
        save_model = [save_model,
                      getattr(get_vec_normalize(envs), 'ob_rms', None)]

    torch.save(save_model, os.path.join(save_path, cfg.env.name + "_" + str(current_update_step) + ".pt"))


def save_image(cfg, agent_train_images, tb_writer_img, current_update_step):
    # Save visualization of last training step
    if cfg.experiment.save_train_images and agent_train_images is not None:
        images = agent_train_images["obs"]
        for image_idx, image in enumerate(images):
            tb_writer_img.add_images("Policy Update {}".format(current_update_step), image / 255.0, image_idx)

        if cfg.learning.consistency_loss.enable:
            augmentation_obs_keys = ["obs_aug_orig", "obs_aug_augmented"]
            aug_images = map(functools.partial(torch.cat, dim=3), zip(
                *[agent_train_images[obs_key] for obs_key in augmentation_obs_keys]))
            for image_idx, image in enumerate(aug_images):
                tb_writer_img.add_images("Policy Update Augmentation{}".format(current_update_step),
                                         image / 255.0, image_idx)


def eval_episode(cfg, env_name, update_step, num_updates, actor_critic, device, eval_log_dir, tb_writer_img,
                 eval_name="eval"):
    eval_steps = 32 if update_step < num_updates - 1 else 100

    env_args = {"env_params_sampler_dict": cfg.env.params_file,
                "data_folder_path": cfg.env.data_folder_path}
    eval_envs = make_vec_envs(
        env_name, cfg.globals.seed + cfg.globals.num_processes * update_step, cfg.globals.num_processes,
        cfg.learning.rl.gamma, eval_log_dir, cfg.env.add_timestep, device, True,
        num_frame_stack=cfg.env.num_framestack,
        normalize_obs=cfg.env.normalize_obs,
        env_args=env_args
    )

    # TODO: Dead code?
    # vec_norm = get_vec_normalize(eval_envs)
    # if vec_norm is not None:
    #     vec_norm.eval()
    #     if cfg.learning.rl.actor_critic.combi_policy:
    #         vec_norm.ob_robot_rms = get_vec_normalize(envs).ob_robot_rms
    #         vec_norm.ob_task_rms = get_vec_normalize(envs).ob_task_rms
    #     else:
    #         vec_norm.ob_rms = get_vec_normalize(envs).ob_rms

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(cfg.globals.num_processes,
                                               actor_critic.recurrent_hidden_state_size,
                                               device=device)
    eval_masks = torch.zeros(cfg.globals.num_processes, 1, device=device)

    save_cnt = 0
    if cfg.experiment.save_eval_images and update_step % 300 == 0:
        os.mkdir(os.path.join(eval_log_dir, "iter_{}_{}".format(update_step, eval_name)))

    while len(eval_episode_rewards) <= eval_steps:
        with torch.no_grad():
            if cfg.learning.consistency_loss.use_cnn_loss:
                _, action, _, eval_recurrent_hidden_states, _ = actor_critic.act(
                    obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)
            else:
                _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                    obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

        # Obser reward and next obs
        obs, reward, done, infos = eval_envs.step(action)
        if cfg.experiment.save_eval_images and update_step % 300 == 0 and save_cnt < 150:
            img = obs['img'].cpu().numpy()[0, ::-1, :, :].transpose((1, 2, 0)).astype(
                np.uint8)
            cv2.imwrite(
                os.path.join(eval_log_dir, "iter_{}_{}/img_{}.png".format(update_step, save_cnt, eval_name)), img)

            # Tensorboard expects images to be in range [0, 1] for FloatTensor
            # TODO: Change obs range to [0, 1]
            tb_writer_img.add_images(eval_name + "_" + str(update_step), obs['img'].cpu() / 255.0, save_cnt,
                                     dataformats='NCHW')
            save_cnt += 1

        eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                        for done_ in done])
        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    eval_envs.close()
    return eval_episode_rewards


def setup_agent(cfg, actor_critic):
    if cfg.learning.rl.algo == 'a2c':
        return setup_a2c_agent(cfg, actor_critic)
    elif cfg.learning.rl.algo == 'ppo':
        return setup_ppo_agent(cfg, actor_critic)
    elif cfg.learning.rl.algo == 'acktr':
        return algo.A2C_ACKTR(actor_critic, cfg.learning.rl.value_loss_coef,
                              cfg.learning.rl.entropy_coef, acktr=True)


def save_eval_episode_rewards(cfg, eval_episode_rewards, step, total_num_steps, tb_writer, log_file, eval_name="_"):
    if cfg.tensorboard:
        tb_writer.add_scalar("eval{}eprewmean_updates".format(eval_name), np.mean(eval_episode_rewards), step)
        tb_writer.add_scalar("eval{}eprewmean_steps".format(eval_name), np.mean(eval_episode_rewards),
                             total_num_steps)
        tb_writer.add_scalar("eval{}success_rate".format(eval_name),
                             np.mean(np.array(eval_episode_rewards) > 0).astype(np.float),
                             total_num_steps)
        tb_writer.flush()

    eval_log_output = "\nEvaluation using {} episodes on env {} (name {}): mean reward {:.5f}\n\n".format(
        len(eval_episode_rewards), cfg.env.name, eval_name, np.mean(eval_episode_rewards))
    print()
    print(eval_log_output)
    print()
    log_file.write(eval_log_output)
    log_file.flush()


def train(sysargs):
    cfg = get_args(sysargs[1:])

    # ======== Setup
    validate_inputs(cfg)
    set_seeds(cfg)
    visdom, window = setup_visualization(cfg)

    tb_writer, tb_writer_img, log_file = setup_dirs_and_logging(cfg)
    device = setup_cuda(cfg)

    num_updates = int(cfg.env.num_env_steps) // cfg.globals.num_steps // cfg.globals.num_processes
    curr_args = get_curriculum_args(cfg, num_updates, tb_writer)
    env_args = get_env_args(cfg)

    envs = make_vec_envs(cfg.env.name, cfg.globals.seed, cfg.globals.num_processes,
                         cfg.learning.rl.gamma, cfg.log_dir, cfg.env.add_timestep, device,
                         allow_early_resets=False, curr_args=curr_args,
                         num_frame_stack=cfg.env.num_framestack,
                         normalize_obs=cfg.env.normalize_obs,
                         env_args=env_args)

    actor_critic = setup_actor_critic(cfg, envs)
    actor_critic.to(device)

    obs = envs.reset()
    rollouts = _setup_rollouts(cfg, actor_critic, obs, envs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=cfg.learning.curriculum.rew_q_len)
    train_success = deque(maxlen=cfg.learning.curriculum.rew_q_len)
    eval_episode_rewards = []
    agent = setup_agent(cfg, actor_critic)

    # ======== Training loop
    start = time.time()
    for update_step in tqdm(range(num_updates), desc="Updates"):
        # === Update LR
        update_learning_rate(cfg, agent, update_step, num_updates, eval_episode_rewards)

        # === Rollout collection
        for step in tqdm(range(cfg.globals.num_steps), desc="Env steps"):
            # Sample actions
            with torch.no_grad():
                if cfg.learning.rl.actor_critic.combi_policy:
                    act_args = ({'img': rollouts.obs_img[step],
                                 'robot_state': rollouts.obs_robot[step],
                                 'task_state': rollouts.obs_task[step]},
                                rollouts.recurrent_hidden_states[step],
                                rollouts.masks[step])
                    if cfg.learning.consistency_loss.use_cnn_loss:
                        value, action, action_log_prob, recurrent_hidden_states, cnn_output = actor_critic.act(
                            *act_args)
                    else:
                        value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(*act_args)
                else:
                    value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])

            obs, reward, done, infos = envs.step(action)

            for i, info in enumerate(infos):
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    train_success.append(float(info['task_success']))

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                       for done_ in done])
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward,
                            masks)

        with torch.no_grad():
            if cfg.learning.rl.actor_critic.combi_policy:
                next_value = actor_critic.get_value({'img': rollouts.obs_img[-1],
                                                     'robot_state': rollouts.obs_robot[-1],
                                                     'task_state': rollouts.obs_task[-1]},
                                                    rollouts.recurrent_hidden_states[-1],
                                                    rollouts.masks[-1]).detach()
            else:
                next_value = actor_critic.get_value(rollouts.obs[-1],
                                                    rollouts.recurrent_hidden_states[-1],
                                                    rollouts.masks[-1]).detach()

        total_num_steps = (update_step + 1) * cfg.globals.num_processes * cfg.globals.num_steps

        # Add current step information to agent as some agent need this info for calculation of losses
        if hasattr(agent, 'set_current_num_steps'):
            agent.set_current_num_steps(total_num_steps)

        # Update agent
        rollouts.compute_returns(next_value, cfg.learning.rl.gae.enable, cfg.learning.rl.gamma, cfg.learning.rl.gae.tau)
        value_loss, action_loss, dist_entropy, update_log = agent.update(rollouts)
        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (update_step % cfg.experiment.save_interval == 0 or update_step == num_updates - 1) and cfg.save_dir != "":
            save_model(cfg, envs, actor_critic, update_step)
            save_image(cfg, update_log['images'], tb_writer_img, update_step)

        if update_step % cfg.experiment.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()
            log_output = "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward " \
                         "{:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".format(update_step,
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

        if (cfg.experiment.eval_interval is not None and len(
                episode_rewards) > 1 and update_step % cfg.experiment.eval_interval == 0):
            # Evaluation on training domain
            eval_episode_rewards = eval_episode(cfg, cfg.env.name, update_step, num_updates, actor_critic, device,
                                                cfg.eval_log_dir, tb_writer_img, "eval")
            save_eval_episode_rewards(cfg, eval_episode_rewards, update_step, total_num_steps, tb_writer, log_file)

            if cfg.learning.consistency_loss.eval_target_env is not None:
                # Evaluation on target domain (if set, only simulated envs supported)
                eval_target_domain_env = cfg.learning.consistency_loss.eval_target_env
                eval_episode_rewards_target_domain = eval_episode(cfg, eval_target_domain_env, update_step, num_updates,
                                                                  actor_critic, device, cfg.eval_log_dir,
                                                                  tb_writer_img, "eval_target_domain")

                save_eval_episode_rewards(cfg, eval_episode_rewards_target_domain, update_step, total_num_steps, tb_writer, log_file,
                                          eval_name="_target_domain_")

        if cfg.experiment.vis and update_step % cfg.experiment.vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                window = visdom_plot(visdom, window, cfg.log_dir, cfg.training_name,
                                     cfg.learning.rl.algo, cfg.env.num_env_steps)
            except IOError:
                pass

        if cfg.tensorboard and len(episode_rewards) > 1:
            tb_writer.add_scalar("eprewmean_updates", np.mean(episode_rewards), update_step)
            tb_writer.add_scalar("eprewmean_steps", np.mean(episode_rewards), total_num_steps)
            tb_writer.add_scalar("training_success_rate", np.mean(train_success), total_num_steps)
            tb_writer.add_scalar("eprewmedian_steps", np.median(episode_rewards), total_num_steps)

            for scalar in update_log.keys():
                if "image" not in scalar:
                    tb_writer.add_scalar(scalar, update_log[scalar], total_num_steps)

        if cfg.tensorboard:
            tb_writer.flush()

    if cfg.tensorboard:
        tb_writer.close()
        if cfg.experiment.save_eval_images or cfg.experiment.save_train_images:
            tb_writer_img.close()


if __name__ == "__main__":
    train(sys.argv)
