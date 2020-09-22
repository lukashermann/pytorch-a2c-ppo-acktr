import copy

import torch
import torch.nn as nn
import torch.optim as optim
from a2c_ppo_acktr.combi_policy import CombiPolicy

from a2c_ppo_acktr.augmentation.augmenters import Augmenter


def are_models_equal(actor_critic, target_actor_critic):
    for p1, p2 in zip(actor_critic.parameters(), target_actor_critic.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


class Consistency:
    def __init__(self,
                 actor_critic,
                 num_mini_batch,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 augmenter: Augmenter = None,
                 return_images: bool = False,
                 augmentation_data_loader=None,
                 augmentation_loss_weight_function=None,
                 force_ignore_loss_aug=False,
                 target_actor_critic=None):
        self.actor_critic = actor_critic

        # Start with with same instance of actor critic, let update_target_critic handle the rest
        self.target_actor_critic = target_actor_critic
        self.update_target_critic()

        self.num_mini_batch = num_mini_batch

        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

        assert augmenter is not None
        assert augmentation_data_loader is not None
        self.augmenter = augmenter
        self.augmentation_data_loader = augmentation_data_loader

        # If this is set to true, the augmentation loss is calculated, but not added to the loss
        self.force_ignore_loss_aug = force_ignore_loss_aug

        if self.augmenter is not None:
            assert augmentation_loss_weight_function is not None
        self.augmentation_loss_weight_function = augmentation_loss_weight_function

        self.return_images = return_images
        self.current_num_steps = 0

    def set_current_num_steps(self, steps):
        self.current_num_steps = steps

    def update(self, rollouts):
        update_log = self.init_update_logging(with_augmentation=True)

        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-5)

        data_generator = rollouts.feed_forward_generator(
            advantages, self.num_mini_batch)

        augmentation_data_loader_iter = iter(self.augmentation_data_loader)

        for sample in data_generator:
            obs_batch, recurrent_hidden_states_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
            adv_targ = sample

            if self.actor_critic.return_cnn_output:
                _, actor_critic_action, _, _, _ = self.actor_critic.act(obs_batch, None, None, deterministic=True)
                _, target_action, _, _, _ = self.target_actor_critic.act(obs_batch, None, None, deterministic=True)
            else:
                _, actor_critic_action, _, _ = self.actor_critic.act(obs_batch, None, None, deterministic=True)
                _, target_action, _, _ = self.target_actor_critic.act(obs_batch, None, None, deterministic=True)

            action_loss = torch.nn.functional.mse_loss(target_action.detach(), actor_critic_action)

            aug_obs_batch_orig = next(augmentation_data_loader_iter)
            # Move data to model's device
            device = obs_batch["img"].device
            aug_obs_batch_orig = {k: v.to(device) for k, v in
                                  aug_obs_batch_orig.items()}

            action_loss_aug, aug_obs_batch_augmented, augmenter_loss_data = self.augmenter.calculate_loss(
                actor_critic=self.actor_critic,
                obs_batch=aug_obs_batch_orig,
                recurrent_hidden_states_batch=None,
                masks_batch=None,
                return_images=self.return_images)

            action_loss.retain_grad()  # retain grad for norm calculation
            action_loss_aug.retain_grad()
            action_loss_aug_weight = self.augmentation_loss_weight_function(self.current_num_steps, action_loss.item())
            action_loss_aug_weighted = action_loss_aug_weight * action_loss_aug

            if self.force_ignore_loss_aug:
                action_loss_sum = action_loss
            else:
                action_loss_sum = action_loss + action_loss_aug_weighted

            self.optimizer.zero_grad()
            action_loss_sum.retain_grad()  # retain grad for norm calculation
            action_loss_sum.backward()
            self.optimizer.step()

            # print("Actor Critic and target are equal: {}".format(are_models_equal(self.actor_critic, self.target_actor_critic)))
            # print("Actor critic params")
            # print(list(self.actor_critic.parameters())[0].grad)

            if self.return_images:
                update_log['images']["obs"].append(obs_batch['img'].cpu())
                update_log['images']["obs_aug_orig"].append(aug_obs_batch_orig['img'].cpu())
                update_log['images']["obs_aug_augmented"].append(aug_obs_batch_augmented['img'].cpu())

            update_log['action_loss_sum'] = action_loss_sum.item()
            update_log['action_loss'] = action_loss.item()
            update_log['action_loss_aug'] = action_loss_aug.item()
            update_log['action_loss_aug_weight'] = action_loss_aug_weight
            update_log['action_loss_aug_weighted'] = action_loss_aug_weighted
            # In order to analyse the action space we report the max action performed in the augmentation step
            if augmenter_loss_data["action_aug_max_value"] >= update_log['action_aug_max_value']:
                update_log['action_aug_max_value'] = augmenter_loss_data["action_aug_max_value"]

            update_log['action_loss_ratio'] = action_loss / action_loss_aug_weighted

            max_actions_batch = torch.max(actions_batch)
            if max_actions_batch >= update_log['action_max_value']:
                update_log['action_max_value'] = max_actions_batch

        return update_log['value_loss'], update_log['action_loss'], update_log['dist_entropy'], update_log

    def init_update_logging(self, with_augmentation=False):
        update_log = {
            'value_loss': 0,
            'action_loss': 0,
            'action_loss_sum': 0,
            'action_max_value': 0,
            'dist_entropy': 0,
            'total_norm': 0,
            'grad_norm': 0,
            'action_loss_ratio': 0,
            'action_loss_aug_weight': 0,
            'images': {"obs": []}
        }

        if with_augmentation:
            update_log['action_loss_aug'] = 0
            update_log['action_loss_aug_weighted'] = 0
            update_log['action_aug_max_value'] = 0

            update_log['images']["obs_aug_orig"] = []
            update_log['images']["obs_aug_augmented"] = []

        return update_log

    def update_target_critic(self):
        self.target_actor_critic.load_state_dict(self.actor_critic.state_dict())
        self.target_actor_critic.eval()

        for param in self.target_actor_critic.parameters():
            param.requires_grad = False
