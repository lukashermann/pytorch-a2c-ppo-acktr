import argparse
import os
import cv2
import numpy as np
import torch

from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize

def build_env():
    pass


class Model:
    def __init__(self, env, snapshot, deterministic=True):
        self.actor_critic, self.ob_rms = torch.load(snapshot)
        self.deterministic = deterministic
        # vec_norm = get_vec_normalize(env)
        # if vec_norm is not None:
        #     vec_norm.eval()
        #     vec_norm.ob_rms = self.ob_rms
        self.recurrent_hidden_states = torch.zeros(1, self.actor_critic.recurrent_hidden_state_size)
        self.masks = torch.zeros(1, 1)

    def step(self, obs, done):
        self.masks.fill_(0.0 if done else 1.0)
        with torch.no_grad():
            value, action, _, self.recurrent_hidden_states = self.actor_critic.act(obs, self.recurrent_hidden_states,
                                                                                   self.masks,
                                                                                   deterministic=self.deterministic)
        return action

