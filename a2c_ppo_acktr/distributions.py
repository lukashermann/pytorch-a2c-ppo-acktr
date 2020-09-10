import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from a2c_ppo_acktr.utils import AddBias, init

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

#
# Standardize distribution interfaces
#

# Categorical
# FixedCategorical = torch.distributions.Categorical
#
# old_sample = FixedCategorical.sample
# FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)
#
# log_prob_cat = FixedCategorical.log_prob
# FixedCategorical.log_probs = lambda self, actions: log_prob_cat(self, actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)
#
# FixedCategorical.mode = lambda self: self.probs.argmax(dim=-1, keepdim=True)


# Normal
FixedNormal = torch.distributions.Normal

log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(self, actions).sum(-1, keepdim=True)

normal_entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: normal_entropy(self).sum(-1)

FixedNormal.mode = lambda self: self.mean


# Bernoulli
FixedBernoulli = torch.distributions.Bernoulli

log_prob_bernoulli = FixedBernoulli.log_prob
FixedBernoulli.log_probs = lambda self, actions: log_prob_bernoulli(self, actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

bernoulli_entropy = FixedBernoulli.entropy
FixedBernoulli.entropy = lambda self: bernoulli_entropy(self).sum(-1)
FixedBernoulli.mode = lambda self: torch.gt(self.probs, 0.5).float()


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        raise NotImplementedError
#     def __init__(self, num_inputs, num_outputs):
#         super(Categorical, self).__init__()
#
#         init_ = lambda m: init(m,
#             nn.init.orthogonal_,
#             lambda x: nn.init.constant_(x, 0),
#             gain=0.01)
#
#         self.linear = init_(nn.Linear(num_inputs, num_outputs))
#
#     def forward(self, x):
#         x = self.linear(x)
#         return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())


class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Bernoulli, self).__init__()

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)


class MultiDiscrete(nn.Module):
    def __init__(self, num_inputs, num_outputs, nvec):
        super(MultiDiscrete, self).__init__()

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               gain=0.01)
        self.nvec = nvec
        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)

        def get_multi_categorical(logits):
            start = 0
            ans = []
            for n in self.nvec:
                ans.append(torch.distributions.Categorical(logits=logits[:, start: start + n]))
                start += n
            return MultiCategorical(ans)
        return get_multi_categorical(logits=x)


class MultiCategorical(torch.distributions.Distribution):

    def __init__(self, dists):
        super().__init__()
        self.dists = dists

    def log_probs(self, value):
        ans = []
        for d, v in zip(self.dists, torch.split(value, 1, dim=-1)):
            ans.append(d.log_prob(v.squeeze(-1)))
        return torch.stack(ans, dim=-1).sum(dim=-1).unsqueeze(1)

    def entropy(self):
        return torch.stack([d.entropy() for d in self.dists], dim=-1).sum(dim=-1)

    def sample(self, sample_shape=torch.Size()):
        return torch.stack([d.sample(sample_shape) for d in self.dists], dim=-1)

    def mode(self):
        return torch.stack([torch.argmax(d.probs, dim=1) for d in self.dists], dim=-1)

