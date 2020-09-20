import random

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from a2c_ppo_acktr.augmentation.augmenters import Augmenter


class Consistency():
    def __init__(self,
                 actor_critic,
                 num_mini_batch,
                 # value_loss_coef,
                 # entropy_coef,
                 max_grad_norm=None,
                 num_epochs=4,
                 lr=None,
                 eps=None,
                 # max_grad_norm=None,
                 # use_clipped_value_loss=True,
                 augmenter: Augmenter = None,
                 return_images: bool = False,
                 augmentation_data_loader=None,
                 augmentation_loss_weight_function=None,
                 force_ignore_loss_aug=False):
        #
        self.actor_critic = actor_critic
        self.num_mini_batch = num_mini_batch
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

        self.augmenter = augmenter
        self.augmemtation_data_loader = augmentation_data_loader

        self.max_grad_norm = max_grad_norm

        self.num_epochs = num_epochs

        # If this is set to true, the augmentation loss is calculated, but not added to the loss
        self.force_ignore_loss_aug = force_ignore_loss_aug

        if self.augmenter is not None:
            assert augmentation_loss_weight_function is not None
        self.augmentation_loss_weight_function = augmentation_loss_weight_function

        self.return_images = return_images
        self.current_num_steps = 0

    def set_current_num_steps(self, steps):
        self.current_num_steps = steps

    def update(self, _):
        update_log = self.init_update_logging(with_augmentation=self.augmenter != None)

        for e in tqdm(range(self.num_epochs), desc="PPO Epochs"):
            # Relaod data loader iterator
            if self.augmenter is not None and self.augmemtation_data_loader is not None:
                augmentation_data_loader_iter = iter(self.augmemtation_data_loader)

            for aug_obs_batch_orig in augmentation_data_loader_iter:

                action_loss_aug = 0

                # Move data to model's device
                aug_obs_batch_orig = {k: v.to(0) for k, v in
                                      aug_obs_batch_orig.items()}

                action_loss_aug, aug_obs_batch_augmented, augmenter_loss_data = self.augmenter.calculate_loss(
                    actor_critic=self.actor_critic,
                    obs_batch=aug_obs_batch_orig,
                    recurrent_hidden_states_batch=None,
                    masks_batch=None,
                    return_images=self.return_images)

                if self.return_images:
                    update_log['images']["obs_aug_orig"].append(aug_obs_batch_orig['img'].cpu())
                    update_log['images']["obs_aug_augmented"].append(aug_obs_batch_augmented['img'].cpu())

                action_loss_aug.retain_grad()
                action_loss_aug_weight = self.augmentation_loss_weight_function(self.current_num_steps, _)
                action_loss_aug_weighted = action_loss_aug_weight * action_loss_aug

                action_loss_sum = action_loss_aug_weighted
                action_loss_sum.retain_grad()  # retain grad for norm calculation

                self.optimizer.zero_grad()

                action_loss_sum.backward()
                total_norm = nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                                      self.max_grad_norm)
                self.optimizer.step()

                update_log['action_loss_sum'] += action_loss_sum.item()
                update_log['total_norm'] += total_norm  # no .item() call, as it is already a float value

                if self.augmenter is not None:
                    update_log['action_loss_aug'] += action_loss_aug.item()
                    update_log['action_loss_aug_weight'] += action_loss_aug_weight
                    update_log['action_loss_aug_weighted'] += action_loss_aug_weighted
                    # In order to analyse the action space we report the max action performed in the augmentation step
                    if augmenter_loss_data["action_aug_max_value"] >= update_log['action_aug_max_value']:
                        update_log['action_aug_max_value'] = augmenter_loss_data["action_aug_max_value"]

        num_updates = self.num_epochs * self.num_mini_batch

        update_log['action_loss_sum'] /= num_updates
        update_log['total_norm'] /= num_updates
        if self.augmenter is not None:
            update_log['action_loss_aug'] /= num_updates
            update_log['action_loss_aug_weight'] /= num_updates
            update_log['action_loss_aug_weighted'] /= num_updates

        return 0, update_log['action_loss_sum'], 0, update_log

    def init_update_logging(self, with_augmentation=False):
        update_log = {
            'action_loss_sum': 0,
            'total_norm': 0,
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
