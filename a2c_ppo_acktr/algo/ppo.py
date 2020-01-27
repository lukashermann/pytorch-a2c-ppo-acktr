import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym.wrappers import transform_observation
from torch.utils.tensorboard import SummaryWriter

from a2c_ppo_acktr.augmentation.augmenters import Augmenter


class PPO():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 augmenter: Augmenter = None,
                 augmentation_loss_random_prob: float = None,
                 return_images: bool=False):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

        self.augmenter = augmenter
        self.augmentation_loss_random_prob = augmentation_loss_random_prob
        self.return_images = return_images

    def update(self, rollouts):

        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        action_loss_aug_epoch = 0
        dist_entropy_epoch = 0

        # Store images of training step
        images_epoch = {"obs": []}
        if self.augmenter:
            images_epoch["obs_aug"] = []

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                adv_targ = sample

                action_loss_aug = 0
                if self.augmenter is not None:
                    # Only calculate augmentation loss sporadically
                    if not self.augmentation_loss_random_prob or \
                            self.augmentation_loss_random_prob > random.random():

                        obs_batch_aug = self.augmenter.augment_batch(obs_batch)

                        value_unlab, action_unlab, action_log_probs_unlab, rnn_hxs_unlab = \
                            self.actor_critic.act(
                                obs_batch,
                                recurrent_hidden_states_batch,
                                masks_batch,
                                deterministic=True)

                        value_unlab_aug, action_unlab_aug, action_log_probs_unlab_aug, rnn_hxs_unlab_aug = \
                            self.actor_critic.act(
                                obs_batch_aug,
                                recurrent_hidden_states_batch,
                                masks_batch,
                                deterministic=True)

                        action_loss_aug = torch.nn.functional.mse_loss(action_unlab,
                                                                       action_unlab_aug.detach())

                        if self.return_images:
                            images_epoch["obs_aug"].append(obs_batch_aug['img'].cpu())

                if self.return_images:
                    images_epoch["obs"].append(obs_batch['img'].cpu())

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch,
                    masks_batch, actions_batch)

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.augmenter is not None:
                    action_loss = action_loss + action_loss_aug

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                                         (values - value_preds_batch).clamp(-self.clip_param,
                                                                            self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()

                (value_loss * self.value_loss_coef + action_loss -
                 dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                if self.augmenter is not None:
                    action_loss_aug_epoch += action_loss_aug.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        action_loss_aug_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, action_loss_aug_epoch, images_epoch
