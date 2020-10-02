import torch
import torch.nn as nn

from a2c_ppo_acktr.envs import VecNormalize, DictVecNormalize


# Get a render function
def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize) or isinstance(venv, DictVecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def update_linear_schedule_less(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the lr linearly (half as much as normal)"""
    lr = initial_lr - (initial_lr * (epoch / float(2 * total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def update_linear_schedule_half(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the lr linearly till middle of training"""
    if epoch > total_num_epochs / 2:
        lr = initial_lr / 2
    else:
        lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def update_sr_schedule(optimizer, sr, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * sr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
