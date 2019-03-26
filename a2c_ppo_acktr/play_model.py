import argparse
import os
import cv2
import numpy as np
import torch
from gym.spaces import Dict
from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs, TransposeImage, VecNormalize, DictVecNormalize, DummyVecEnv, DictVecPyTorch, DictTransposeImage
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize


def build_env(env, normalize_obs=True):

    if isinstance(env.observation_space, Dict):
        env = DictTransposeImage(env)
        env = DummyVecEnv([lambda: env])
        if normalize_obs:
            env = DictVecNormalize(env)
        env = DictVecPyTorch(env, 'cpu')
    else:
        env = TransposeImage(env)
        env = DummyVecEnv([lambda: env])
        env = VecPyTorch(env, 'cpu')
    return env


def render_obs(obs, sleep=1, res=(300, 300)):
    if isinstance(obs, dict):
        img = obs['img'].cpu().numpy()[0, ::-1, :, :].transpose((1, 2, 0)).astype(np.uint8)
    else:
        img = obs.cpu().numpy()[0, ::-1, :, :].transpose((1, 2, 0)).astype(np.uint8)
    cv2.imshow("win", cv2.resize(img, res))
    cv2.waitKey(sleep)


class Model:
    def __init__(self, env, snapshot, deterministic=True):
        load_data = torch.load(snapshot)
        if len(load_data) == 3:
            self.actor_critic, self.ob_robot_rms, self.ob_task_rms = load_data
        else:
            self.actor_critic, self.ob_rms = load_data
        self.deterministic = deterministic
        vec_norm = get_vec_normalize(env)
        if vec_norm is not None:
            vec_norm.eval()
            if len(load_data) == 3:
                vec_norm.ob_robot_rms = self.ob_robot_rms
                vec_norm.ob_task_rms = self.ob_task_rms
            else:
                vec_norm.ob_rms = self.ob_rms
        self.recurrent_hidden_states = torch.zeros(1, self.actor_critic.recurrent_hidden_state_size)
        self.masks = torch.zeros(1, 1)

    def step(self, obs, done):
        self.masks.fill_(0.0 if done else 1.0)
        with torch.no_grad():
            value, action, _, self.recurrent_hidden_states = self.actor_critic.act(obs, self.recurrent_hidden_states,
                                                                                   self.masks,
                                                                                   deterministic=self.deterministic)
        return action

