import random

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from gym.wrappers import transform_observation

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
                 return_images: bool = False,
                 augmentation_data_loader=None,
                 augmentation_loss_weight=None,
                 augmentation_loss_weight_function=None):

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
        self.augmemtation_data_loader = augmentation_data_loader

        self.augmentation_loss_weight_function = augmentation_loss_weight_function
        if augmentation_loss_weight is not None:
            # Define inner function which simply returns constant value
            self.augmentation_loss_weight = augmentation_loss_weight

            def constanct_loss_weight(*args): return self.augmentation_loss_weight

            self.augmentation_loss_weight_function = constanct_loss_weight
        self.return_images = return_images
        self.current_num_steps = 0

    def set_current_num_steps(self, steps):
        self.current_num_steps = steps

    def update(self, rollouts):
        update_log = self.init_update_logging(with_augmentation=self.augmenter != None)

        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-5)



        for e in tqdm(range(self.ppo_epoch), desc="PPO Epochs"):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            # Relaod data loader iterator
            if self.augmenter is not None and self.augmemtation_data_loader is not None:
                augmentation_data_loader_iter = iter(self.augmemtation_data_loader)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                adv_targ = sample

                action_loss_aug = 0
                if self.augmenter is not None:
                    if self.augmemtation_data_loader is not None:
                        aug_obs_batch_orig = next(augmentation_data_loader_iter)

                        # Move data to model's device
                        device = obs_batch["img"].device
                        aug_obs_batch_orig = {k: v.to(device) for k, v in
                                              aug_obs_batch_orig.items()}
                    else:
                        aug_obs_batch_orig = obs_batch

                    action_loss_aug, aug_obs_batch_augmented, augmenter_loss_data = self.augmenter.calculate_loss(
                        actor_critic=self.actor_critic,
                        obs_batch=aug_obs_batch_orig,
                        recurrent_hidden_states_batch=recurrent_hidden_states_batch,
                        masks_batch=masks_batch,
                        return_images=self.return_images)

                    if self.return_images:
                        update_log['images']["obs_aug_orig"].append(aug_obs_batch_orig['img'].cpu())
                        update_log['images']["obs_aug_augmented"].append(aug_obs_batch_augmented['img'].cpu())

                if self.return_images:
                    update_log['images']["obs"].append(obs_batch['img'].cpu())

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
                    action_loss.retain_grad()  # retain grad for norm calculation
                    action_loss_aug.retain_grad()

                    action_loss_aug_weighted = self.weight_augmentation_loss(step=self.current_num_steps,
                                                                             action_loss_aug=action_loss_aug)
                    action_loss_sum = action_loss + action_loss_aug_weighted
                else:
                    action_loss_aug_weighted = 0
                    action_loss_sum = action_loss
                action_loss_sum.retain_grad()  # retain grad for norm calculation

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

                (value_loss * self.value_loss_coef + action_loss_sum -
                 dist_entropy * self.entropy_coef).backward()
                total_norm = nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                                      self.max_grad_norm)
                self.optimizer.step()

                update_log['value_loss'] += value_loss.item()
                update_log['action_loss'] += action_loss.item()
                update_log['action_loss_sum'] += action_loss_sum.item()
                update_log['dist_entropy'] += dist_entropy.item()
                update_log['total_norm'] += total_norm  # no .item() call, as it is already a float value

                if self.augmenter is not None:
                    update_log['action_loss_aug'] += action_loss_aug.item()
                    update_log['action_loss_aug_weighted'] += action_loss_aug_weighted
                    # In order to analyse the action space we report the max action performed in the augmentation step
                    if augmenter_loss_data["action_aug_max_value"] >= update_log['action_aug_max_value']:
                        update_log['action_aug_max_value'] = augmenter_loss_data["action_aug_max_value"]

                max_actions_batch = torch.max(actions_batch)
                if max_actions_batch >= update_log['action_max_value']:
                    update_log['action_max_value'] = max_actions_batch

        num_updates = self.ppo_epoch * self.num_mini_batch

        update_log['value_loss'] /= num_updates
        update_log['action_loss'] /= num_updates
        update_log['action_loss_sum'] /= num_updates
        update_log['dist_entropy'] /= num_updates
        update_log['total_norm'] /= num_updates
        if self.augmenter is not None:
            update_log['action_loss_aug'] /= num_updates
            update_log['action_loss_aug_weighted'] /= num_updates

        return update_log['value_loss'], update_log['action_loss'], update_log['dist_entropy'], update_log

    def weight_augmentation_loss(self, step, action_loss_aug, use_absolute_value=True):
        if self.augmentation_loss_weight_function:
            factor = abs(self.augmentation_loss_weight_function(
                step)) if use_absolute_value else self.augmentation_loss_weight_function(step)
            return factor * action_loss_aug
        else:
            return self.augmentation_loss_weight * action_loss_aug

    def init_update_logging(self, with_augmentation=False):
        update_log = {
            'value_loss': 0,
            'action_loss': 0,
            'action_loss_sum': 0,
            'action_max_value': 0,
            'dist_entropy': 0,
            'total_norm': 0,
            'grad_norm': 0,
            'images': {"obs": []}
        }

        if with_augmentation:
            update_log['action_loss_aug'] = 0
            update_log['action_loss_aug_weighted'] = 0
            update_log['action_aug_max_value'] = 0

            update_log['images']["obs_aug_orig"] = []
            update_log['images']["obs_aug_augmented"] = []

        return update_log