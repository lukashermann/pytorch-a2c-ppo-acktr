import os

import gym
import numpy as np
import torch
from collections import deque
from gym.spaces.box import Box
from gym.spaces.dict import Dict
from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize as VecNormalize_
from baselines.common.vec_env.vec_normalize import DictVecNormalize as DictVecNormalize_


# try:
#     import dm_control2gym
# except ImportError:
#     pass
#
# try:
#     import roboschool
# except ImportError:
#     pass
#
# try:
#     import pybullet_envs
# except ImportError:
#     pass


def make_env(env_id, seed, rank, log_dir, add_timestep, allow_early_resets, **env_kwargs):
    def _thunk():
        env = gym.make(env_id, **env_kwargs)

        env.seed(seed + rank)

        obs_shape = env.observation_space.shape

        if add_timestep and len(
                obs_shape) == 1 and str(env).find('TimeLimit') > -1:
            env = AddTimestep(env)

        if log_dir is not None:
            env = bench.Monitor(env, os.path.join(log_dir, str(rank)),
                                allow_early_resets=allow_early_resets)

        # if is_atari:
        #     if len(env.observation_space.shape) == 3:
        #         env = wrap_deepmind(env)
        # elif len(env.observation_space.shape) == 3:
        #     raise NotImplementedError("CNN models work only for atari,\n"
        #         "please use a custom wrapper for a custom pixel input env.\n"
        #         "See wrap_deepmind for an example.")

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        if isinstance(env.observation_space, Dict):
            obs_shape = env.observation_space.spaces['img'].shape
            if len(obs_shape) == 3 and obs_shape[2] in [1, 3, 5]:
                env = DictTransposeImage(env)
        else:
            obs_shape = env.observation_space.shape
            if len(obs_shape) == 3 and obs_shape[2] in [1, 3, 5]:
                env = TransposeImage(env)

        return env

    return _thunk


def make_vec_envs(env_name, seed, num_processes, gamma, log_dir, add_timestep,
                  device, allow_early_resets, curr_args=None, num_frame_stack=None, normalize_obs=True, **kwargs):
    envs = [make_env(env_name, seed, i, log_dir, add_timestep, allow_early_resets, **kwargs)
            for i in range(num_processes)]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)
    if normalize_obs:
        if isinstance(envs.observation_space, Dict):
            if gamma is None:
                envs = DictVecNormalize(envs, ret=False)
            else:
                envs = DictVecNormalize(envs, gamma=gamma)
        elif len(envs.observation_space.shape) == 1:
            if gamma is None:
                envs = VecNormalize(envs, ret=False)
            else:
                envs = VecNormalize(envs, gamma=gamma)

    if isinstance(envs.observation_space, Dict):
        envs = DictVecPyTorch(envs, device)
    else:
        envs = VecPyTorch(envs, device)

    if num_frame_stack > 1:
        if isinstance(envs.observation_space, Dict):
            envs = DictVecPyTorchFrameStack(envs, num_frame_stack, device)
        else:
            envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
    if curr_args is not None:
        envs = CurriculumInfoWrapper(envs, **curr_args)
    return envs


# Can be used to test recurrent policies for Reacher-v2
class MaskGoal(gym.ObservationWrapper):
    def observation(self, observation):
        if self.env._elapsed_steps > 0:
            observation[-2:0] = 0
        return observation


class AddTimestep(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(AddTimestep, self).__init__(env)
        self.observation_space = Box(
            self.observation_space.low[0],
            self.observation_space.high[0],
            [self.observation_space.shape[0] + 1],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return np.concatenate((observation, [self.env._elapsed_steps]))


class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


class DictTransposeImage(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(DictTransposeImage, self).__init__(env)
        img_obs_shape = self.observation_space.spaces['img'].shape
        self.observation_space.spaces['img'] = Box(
            self.observation_space.spaces['img'].low[0, 0, 0],
            self.observation_space.spaces['img'].high[0, 0, 0],
            [img_obs_shape[2], img_obs_shape[1], img_obs_shape[0]],
            dtype=self.observation_space.spaces['img'].dtype)

    def observation(self, observation):
        observation['img'] = observation['img'].transpose(2, 0, 1)
        return observation


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def reset_from_curriculum(self, data):
        obs = self.venv.reset_from_curriculum(data)
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        actions = actions.squeeze(1).cpu().numpy()
        self.venv.step_async(actions)

    def step_async_with_curriculum_reset(self, actions, data):
        actions = actions.squeeze(1).cpu().numpy()
        self.venv.step_async_with_curriculum_reset(actions, data)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


class DictVecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(DictVecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = {'img': torch.from_numpy(obs['img']).float().to(self.device),
               'robot_state': torch.from_numpy(obs['robot_state']).float().to(self.device),
               'task_state': torch.from_numpy(obs['task_state']).float().to(self.device)}
        return obs

    def reset_from_curriculum(self, data):
        obs = self.venv.reset_from_curriculum(data)
        obs = {'img': torch.from_numpy(obs['img']).float().to(self.device),
               'robot_state': torch.from_numpy(obs['robot_state']).float().to(self.device),
               'task_state': torch.from_numpy(obs['task_state']).float().to(self.device)}
        return obs

    def step_async(self, actions):
        actions = actions.squeeze(1).cpu().numpy()
        self.venv.step_async(actions)

    def step_async_with_curriculum_reset(self, actions, data):
        actions = actions.squeeze(1).cpu().numpy()
        self.venv.step_async_with_curriculum_reset(actions, data)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = {'img': torch.from_numpy(obs['img']).float().to(self.device),
               'robot_state': torch.from_numpy(obs['robot_state']).float().to(self.device),
               'task_state': torch.from_numpy(obs['task_state']).float().to(self.device)}
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


class VecNormalize(VecNormalize_):

    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs):
        if self.ob_rms:
            if self.training:
                self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


class DictVecNormalize(DictVecNormalize_):

    def __init__(self, *args, **kwargs):
        super(DictVecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs):
        if isinstance(obs, dict):
            if self.ob_robot_rms:
                if self.training:
                    self.ob_robot_rms.update(obs['robot_state'])
                obs['robot_state'] = np.clip((obs['robot_state'] - self.ob_robot_rms.mean) / np.sqrt(self.ob_robot_rms.var + self.epsilon), -self.clipob,
                                             self.clipob)
                if self.training:
                    self.ob_task_rms.update(obs['task_state'])
                obs['task_state'] = np.clip(
                    (obs['task_state'] - self.ob_task_rms.mean) / np.sqrt(self.ob_task_rms.var + self.epsilon),
                    -self.clipob,
                    self.clipob)
                return obs
        else:
            if self.ob_rms:
                if self.training:
                    self.ob_rms.update(obs)
                obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
                return obs
            else:
                return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs,) + low.shape).to(device)

        observation_space = gym.spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:]
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()


class DictVecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space.spaces['img']  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)
        self.device = device
        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs,) + low.shape).to(device)
        observation_space = gym.spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype)
        venv.observation_space.spaces['img'] = observation_space
        VecEnvWrapper.__init__(self, venv, observation_space=venv.observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = self.stacked_obs[:, self.shape_dim0:]
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs['img']
        obs['img'] = self.stacked_obs
        return obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        self.stacked_obs = torch.zeros(self.stacked_obs.shape).to(self.device)
        self.stacked_obs[:, -self.shape_dim0:] = obs['img']
        obs['img'] = self.stacked_obs
        return obs

    def reset_from_curriculum(self, data):
        obs = self.venv.reset_from_curriculum(data)
        self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        self.stacked_obs[:, -self.shape_dim0:] = obs['img']
        obs['img'] = self.stacked_obs
        return obs

    def close(self):
        self.venv.close()


class CurriculumInfoWrapper(VecEnvWrapper):
    def __init__(self, venv, num_updates, num_update_steps, desired_rew_region, incr, tb_writer, num_processes):
        self.venv = venv
        super(CurriculumInfoWrapper, self).__init__(venv)
        self.num_updates = num_updates
        self.num_update_steps = num_update_steps
        self.num_processes = num_processes
        self.step_counter = 0
        self.update_counter = 0
        self.difficulty_cur = 0
        self.difficulty_reg = 0
        self.curr_episode_rewards = deque(maxlen=20)
        self.reg_episode_rewards = deque(maxlen=20)
        self.curr_success = deque(maxlen=20)
        self.reg_success = deque(maxlen=20)
        self.desired_rew_region = desired_rew_region
        self.incr = incr
        self.tb_writer = tb_writer
        self.num_regular_resets = 0
        self.num_resets = 0

    def update_difficulties(self):
        if len(self.curr_success) > 1:
            if np.mean(self.curr_success) > self.desired_rew_region[1]:
                self.difficulty_cur += self.incr
            elif np.mean(self.curr_success) < self.desired_rew_region[0]:
                self.difficulty_cur -= self.incr
            self.difficulty_cur = np.clip(self.difficulty_cur, 0, 1)
        if len(self.reg_success) > 1:
            if np.mean(self.reg_success) > self.desired_rew_region[1]:
                self.difficulty_reg += self.incr
            elif np.mean(self.reg_success) < self.desired_rew_region[0]:
                self.difficulty_reg -= self.incr
            self.difficulty_reg = np.clip(self.difficulty_reg, 0, 1)

    def create_data_dict(self):
        return {'update_step': self.update_counter,
                'num_updates': self.num_updates,
                'eprewmean': None,
                'curr_eprewmean': np.mean(self.curr_episode_rewards) if len(self.curr_episode_rewards) > 1 else 0,
                'eval_eprewmean': None,
                'reg_eprewmean': np.mean(self.reg_episode_rewards) if len(self.reg_episode_rewards) > 1 else 0,
                'curr_success_rate': np.mean(self.curr_success) if len(self.curr_success) > 1 else 0,
                'reg_success_rate': np.mean(self.reg_success) if len(self.reg_success) > 1 else 0,
                'eval_reg_eprewmean': None,
                'difficulty_cur': self.difficulty_cur,
                'difficulty_reg': self.difficulty_reg}

    def step(self, action):
        self.step_counter += 1
        if self.step_counter % self.num_update_steps == 0:
            self.update_counter += 1
            self.update_difficulties()
            self.write_tb_log()
            self.num_regular_resets = 0
            self.num_resets = 0
        data = self.create_data_dict()
        self.step_async_with_curriculum_reset(action, data)
        return self.step_wait()

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_async_with_curriculum_reset(self, actions, data):
        self.venv.step_async_with_curriculum_reset(actions, data)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        for i, info in enumerate(infos):
            if 'episode' in info.keys():
                if 'reset_info' in info.keys() and info['reset_info'] == 'curriculum':
                    self.curr_episode_rewards.append(info['episode']['r'])
                    self.curr_success.append(float(info['task_success']))
                    self.num_resets += 1
                elif 'reset_info' in info.keys() and info['reset_info'] == 'regular':
                    self.reg_episode_rewards.append(info['episode']['r'])
                    self.reg_success.append(float(info['task_success']))
                    self.num_resets += 1
                    self.num_regular_resets += 1
        return obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        return obs

    def reset_from_curriculum(self, data):
        obs = self.venv.reset_from_curriculum(data)
        return obs

    def write_tb_log(self):
        total_num_steps = (self.update_counter + 1) * self.num_processes * self.num_update_steps
        self.tb_writer.add_scalar("curr_success_rate", np.mean(self.curr_success) if len(self.curr_success) else 0, total_num_steps, total_num_steps)
        self.tb_writer.add_scalar("reg_success_rate", np.mean(self.reg_success) if len(self.reg_success) else 0, total_num_steps)
        self.tb_writer.add_scalar("difficulty_cur", self.difficulty_cur, total_num_steps)
        self.tb_writer.add_scalar("difficulty_reg", self.difficulty_reg, total_num_steps)

        if len(self.curr_episode_rewards) > 1:
            self.tb_writer.add_scalar("curr_eprewmean_steps", np.mean(self.curr_episode_rewards), total_num_steps)
            self.tb_writer.add_scalar("regular_resets_ratio", self.num_regular_resets / self.num_resets if self.num_resets > 0 else 0,total_num_steps)
        if len(self.reg_episode_rewards) > 1:
            self.tb_writer.add_scalar("reg_eprewmean_steps", np.mean(self.reg_episode_rewards), total_num_steps)

    def close(self):
        self.venv.close()



