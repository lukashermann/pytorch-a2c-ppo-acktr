import torch
import torch.nn as nn
import numpy as np

from a2c_ppo_acktr.distributions import Categorical, DiagGaussian, Bernoulli, MultiDiscrete
from a2c_ppo_acktr.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def nature_cnn(init_, input_shape, output_size):
    return nn.Sequential(
        init_(nn.Conv2d(input_shape, 32, 8, stride=4)),
        nn.ReLU(),
        init_(nn.Conv2d(32, 64, 4, stride=2)),
        nn.ReLU(),
        init_(nn.Conv2d(64, 32, 3, stride=1)),
        nn.ReLU(),
        Flatten(),
        init_(nn.Linear(32 * 7 * 7, output_size)),
        nn.ReLU()
    )


def cnn_small_filters(init_, input_shape, output_size):
    return nn.Sequential(
        init_(nn.Conv2d(input_shape, 32, 3, stride=2)),
        nn.ReLU(),
        init_(nn.Conv2d(32, 64, 3, stride=2)),
        nn.ReLU(),
        init_(nn.Conv2d(64, 32, 3, stride=2)),
        nn.ReLU(),
        Flatten(),
        init_(nn.Linear(32 * 9 * 9, output_size)),
        nn.ReLU()
    )


def cnn_4_layers(init_, input_shape, output_size):
    return nn.Sequential(
        init_(nn.Conv2d(input_shape, 32, 3, stride=2)),
        nn.ReLU(),
        init_(nn.Conv2d(32, 32, 3, stride=2)),
        nn.ReLU(),
        init_(nn.Conv2d(32, 32, 3, stride=2)),
        nn.ReLU(),
        init_(nn.Conv2d(32, 32, 3, stride=1)),
        nn.ReLU(),
        Flatten(),
        init_(nn.Linear(32 * 7 * 7, output_size)),
        nn.ReLU()
    )


class CombiPolicy(nn.Module):
    def __init__(self, obs_space, action_space, base=None, base_kwargs=None, network_architecture='symm',
                 share_layers=True):
        super(CombiPolicy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}


        if base is None:
            if network_architecture == 'asymm_robot':
                base = CNNAsymmCombi1
            elif network_architecture == 'asymm_robot_task':
                base = CNNAsymmCombi2
            elif network_architecture == 'symm':
                base = CNNCombi
            elif network_architecture == 'resnet':
                base = CNNAsymmCombiResNet
            elif network_architecture == 'combi2':
                base = CNNCombi2
            else:
                raise ValueError

        if "return_cnn_output" in base_kwargs:
            self.return_cnn_output = base_kwargs["return_cnn_output"]
        else:
            self.return_cnn_output = False

        if "return_action_probs" in base_kwargs:
            self.return_action_probs = base_kwargs["return_action_probs"]
        else:
            self.return_action_probs = False

        self.base = base(obs_space, **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == 'MultiDiscrete':
            num_outputs = np.sum(action_space.nvec)
            self.dist = MultiDiscrete(self.base.output_size, num_outputs, action_space.nvec)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        if self.return_cnn_output:
            value, actor_features, rnn_hxs, cnn_output = self.base(inputs, rnn_hxs, masks)
        else:
            value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        if self.return_action_probs:
            action_probs = dist.probs()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        if self.return_cnn_output:
            return value, action, action_log_probs, rnn_hxs, cnn_output
        if self.return_action_probs:
            return value, action, action_log_probs, rnn_hxs, action_probs
        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):

        if self.return_cnn_output:
            value, _, _, _ = self.base(inputs, rnn_hxs, masks)
        else:
            value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        if self.return_cnn_output:
            value, actor_features, rnn_hxs, cnn_output = self.base(inputs, rnn_hxs, masks)
        else:
            value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):

    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                         .any(dim=-1)
                         .nonzero()
                         .squeeze()
                         .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1)
                )

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNAsymmCombi(NNBase):
    def __init__(self, obs_space, recurrent=False, cnn_architecture='nature', output_fc_size=128):
        super(CNNAsymmCombi, self).__init__(recurrent, output_fc_size, output_fc_size)


        state_fc_size = 64
        cnn_fc_size = 512
        img_obs_shape = obs_space.spaces['img'].shape[0]
        robot_state_obs_shape = obs_space.spaces['robot_state'].shape[0]
        task_state_obs_shape = obs_space.spaces['task_state'].shape[0]

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))

        self.cnn = nature_cnn(init_, img_obs_shape, cnn_fc_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.actor_state = nn.Sequential(
            init_(nn.Linear(robot_state_obs_shape, state_fc_size)),
            nn.Tanh(),
            init_(nn.Linear(state_fc_size, state_fc_size)),
            nn.Tanh()
        )

        self.critic_state = nn.Sequential(
            init_(nn.Linear(robot_state_obs_shape + task_state_obs_shape, state_fc_size)),
            nn.Tanh(),
            init_(nn.Linear(state_fc_size, state_fc_size)),
            nn.Tanh()
        )

        self.actor_fuse = nn.Sequential(
            init_(nn.Linear(state_fc_size + cnn_fc_size, output_fc_size)),
            nn.Tanh())

        self.critic_fuse = nn.Sequential(
            init_(nn.Linear(state_fc_size + cnn_fc_size, output_fc_size)),
            nn.Tanh())

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(nn.Linear(output_fc_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        img_input = inputs['img']
        robot_state_input = inputs['robot_state']
        task_state_input = inputs['task_state']

        cnn_output = self.cnn(img_input / 255.0)

        h_actor1 = self.actor_state(robot_state_input)
        h_critic1 = self.critic_state(torch.cat((robot_state_input, task_state_input), 1))

        h_actor2 = self.actor_fuse(torch.cat((cnn_output, h_actor1), 1))
        h_critic2 = self.critic_fuse(torch.cat((cnn_output, h_critic1), 1))

        return self.critic_linear(h_critic2), h_actor2, rnn_hxs


class CNNCombi(NNBase):
    def __init__(self, obs_space, recurrent=False, cnn_architecture="nature", output_fc_size=128, return_cnn_output=False, **kwargs):
        super(CNNCombi, self).__init__(recurrent, output_fc_size, output_fc_size)

        # If set to true, in addition to the regular network output, the network will return the cnn output
        self.return_cnn_output = return_cnn_output

        state_fc_size = 64
        cnn_fc_size = 512
        img_obs_shape = obs_space.spaces['img'].shape[0]
        state_obs_shape = obs_space.spaces['robot_state'].shape[0]

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))

        if cnn_architecture == 'small_filters':
            self.cnn = cnn_small_filters(init_, img_obs_shape, cnn_fc_size)
        elif cnn_architecture == '4_layers':
            self.cnn = cnn_4_layers(init_, img_obs_shape, cnn_fc_size)
        else:
            self.cnn = nature_cnn(init_, img_obs_shape, cnn_fc_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.actor_state = nn.Sequential(
            init_(nn.Linear(state_obs_shape, state_fc_size)),
            nn.Tanh(),
            init_(nn.Linear(state_fc_size, state_fc_size)),
            nn.Tanh()
        )

        self.critic_state = nn.Sequential(
            init_(nn.Linear(state_obs_shape, state_fc_size)),
            nn.Tanh(),
            init_(nn.Linear(state_fc_size, state_fc_size)),
            nn.Tanh()
        )

        self.actor_fuse = nn.Sequential(
            init_(nn.Linear(state_fc_size + cnn_fc_size, output_fc_size)),
            nn.Tanh())

        self.critic_fuse = nn.Sequential(
            init_(nn.Linear(state_fc_size + cnn_fc_size, output_fc_size)),
            nn.Tanh())

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(nn.Linear(output_fc_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        img_input = inputs['img']
        state_input = inputs['robot_state']

        cnn_output = self.cnn(img_input / 255.0)

        h_actor1 = self.actor_state(state_input)
        h_critic1 = self.critic_state(state_input)

        h_actor2 = self.actor_fuse(torch.cat((cnn_output, h_actor1), 1))
        h_critic2 = self.critic_fuse(torch.cat((cnn_output, h_critic1), 1))

        if self.return_cnn_output:
            return self.critic_linear(h_critic2), h_actor2, rnn_hxs, cnn_output
        return self.critic_linear(h_critic2), h_actor2, rnn_hxs


class CNNCombi2(NNBase):
    def __init__(self, obs_space, recurrent=False, cnn_architecture="nature", output_fc_size=256):
        super(CNNCombi2, self).__init__(recurrent, output_fc_size, output_fc_size)

        state_fc_size = 64
        cnn_fc_size = 256
        img_obs_shape = obs_space.spaces['img'].shape[0]
        state_obs_shape = obs_space.spaces['robot_state'].shape[0]

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))

        if cnn_architecture == 'small_filters':
            self.cnn = cnn_small_filters(init_, img_obs_shape, cnn_fc_size)
        elif cnn_architecture == '4_layers':
            self.cnn = cnn_4_layers(init_, img_obs_shape, cnn_fc_size)
        else:
            self.cnn = nature_cnn(init_, img_obs_shape, cnn_fc_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.actor_state = nn.Sequential(
            init_(nn.Linear(state_obs_shape, state_fc_size)),
            nn.Tanh(),
            init_(nn.Linear(state_fc_size, state_fc_size)),
            nn.Tanh()
        )

        self.critic_state = nn.Sequential(
            init_(nn.Linear(state_obs_shape, state_fc_size)),
            nn.Tanh(),
            init_(nn.Linear(state_fc_size, state_fc_size)),
            nn.Tanh()
        )

        self.actor_fuse = nn.Sequential(
            init_(nn.Linear(state_fc_size + cnn_fc_size, output_fc_size)),
            nn.Tanh())

        self.critic_fuse = nn.Sequential(
            init_(nn.Linear(state_fc_size + cnn_fc_size, output_fc_size)),
            nn.Tanh())

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(nn.Linear(output_fc_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        img_input = inputs['img']
        state_input = inputs['robot_state']

        cnn_output = self.cnn(img_input / 255.0)

        h_actor1 = self.actor_state(state_input)
        h_critic1 = self.critic_state(state_input)

        h_actor2 = self.actor_fuse(torch.cat((cnn_output, h_actor1), 1))
        h_critic2 = self.critic_fuse(torch.cat((cnn_output, h_critic1), 1))

        return self.critic_linear(h_critic2), h_actor2, rnn_hxs


class CNNAsymmCombi1(NNBase):
    def __init__(self, obs_space, recurrent=False, cnn_architecture='nature', output_fc_size=128):
        cnn_fc_size = 512
        super(CNNAsymmCombi1, self).__init__(recurrent, cnn_fc_size, cnn_fc_size)

        state_fc_size = 64
        img_obs_shape = obs_space.spaces['img'].shape[0]
        robot_state_obs_shape = obs_space.spaces['robot_state'].shape[0]
        task_state_obs_shape = obs_space.spaces['task_state'].shape[0]

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))

        self.cnn = nature_cnn(init_, img_obs_shape, cnn_fc_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.critic_state = nn.Sequential(
            init_(nn.Linear(robot_state_obs_shape, state_fc_size)),
            nn.Tanh(),
            init_(nn.Linear(state_fc_size, state_fc_size)),
            nn.Tanh()
        )

        self.critic_fuse = nn.Sequential(
            init_(nn.Linear(state_fc_size + cnn_fc_size, output_fc_size)),
            nn.Tanh())

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(nn.Linear(output_fc_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        img_input = inputs['img']
        robot_state_input = inputs['robot_state']
        task_state_input = inputs['task_state']

        cnn_output = self.cnn(img_input / 255.0)

        h_critic1 = self.critic_state(robot_state_input)

        h_critic2 = self.critic_fuse(torch.cat((cnn_output, h_critic1), 1))

        return self.critic_linear(h_critic2), cnn_output, rnn_hxs


class CNNAsymmCombi2(NNBase):
    def __init__(self, obs_space, recurrent=False, cnn_architecture='nature', output_fc_size=128):
        cnn_fc_size = 512
        super(CNNAsymmCombi2, self).__init__(recurrent, cnn_fc_size, cnn_fc_size)

        state_fc_size = 64
        img_obs_shape = obs_space.spaces['img'].shape[0]
        robot_state_obs_shape = obs_space.spaces['robot_state'].shape[0]
        task_state_obs_shape = obs_space.spaces['task_state'].shape[0]

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))

        self.cnn = nature_cnn(init_, img_obs_shape, cnn_fc_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.critic_state = nn.Sequential(
            init_(nn.Linear(robot_state_obs_shape + task_state_obs_shape, state_fc_size)),
            nn.Tanh(),
            init_(nn.Linear(state_fc_size, state_fc_size)),
            nn.Tanh()
        )

        self.critic_fuse = nn.Sequential(
            init_(nn.Linear(state_fc_size + cnn_fc_size, output_fc_size)),
            nn.Tanh())

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(nn.Linear(output_fc_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        img_input = inputs['img']
        robot_state_input = inputs['robot_state']
        task_state_input = inputs['task_state']

        cnn_output = self.cnn(img_input / 255.0)

        h_critic1 = self.critic_state(torch.cat((robot_state_input, task_state_input), 1))

        h_critic2 = self.critic_fuse(torch.cat((cnn_output, h_critic1), 1))

        return self.critic_linear(h_critic2), cnn_output, rnn_hxs


class CNNAsymmCombiResNet(NNBase):
    def __init__(self, obs_space, recurrent=False, cnn_architecture='nature', output_fc_size=128):
        cnn_fc_size = 512
        super(CNNAsymmCombiResNet, self).__init__(recurrent, cnn_fc_size, cnn_fc_size)

        state_fc_size = 64
        img_obs_shape = obs_space.spaces['img'].shape[0]
        robot_state_obs_shape = obs_space.spaces['robot_state'].shape[0]
        task_state_obs_shape = obs_space.spaces['task_state'].shape[0]

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))

        self.down_sample_1 = nn.Sequential(
            init_(nn.Conv2d(img_obs_shape, 32, 3)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 3, stride=2))
        )
        self.res_block_1 = nn.Sequential(
            nn.ReLU(),
            init_(nn.Conv2d(64, 64, 3, padding=1)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 64, 3, padding=1))
        )
        self.down_sample_2 = nn.Sequential(
            nn.ReLU(),
            init_(nn.Conv2d(64, 64, 3, stride=2))
        )
        self.res_block_2 = nn.Sequential(
            nn.ReLU(),
            init_(nn.Conv2d(64, 64, 3, padding=1)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 64, 3, padding=1))
        )
        self.down_sample_3 = nn.Sequential(
            nn.ReLU(),
            init_(nn.Conv2d(64, 64, 3, stride=2)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(64 * 9 * 9, cnn_fc_size)),
            nn.ReLU()
        )

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.critic_state = nn.Sequential(
            init_(nn.Linear(robot_state_obs_shape, state_fc_size)),
            nn.Tanh(),
            init_(nn.Linear(state_fc_size, state_fc_size)),
            nn.Tanh()
        )

        self.critic_fuse = nn.Sequential(
            init_(nn.Linear(state_fc_size + cnn_fc_size, output_fc_size)),
            nn.Tanh())

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(nn.Linear(output_fc_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        img_input = inputs['img']
        robot_state_input = inputs['robot_state']
        task_state_input = inputs['task_state']

        x_skip = self.down_sample_1(img_input / 255.0)
        x = self.res_block_1(x_skip)
        x_skip = self.down_sample_2(x + x_skip)
        x = self.res_block_2(x_skip)
        cnn_output = self.down_sample_3(x + x_skip)

        h_critic1 = self.critic_state(robot_state_input)

        h_critic2 = self.critic_fuse(torch.cat((cnn_output, h_critic1), 1))

        return self.critic_linear(h_critic2), cnn_output, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
