import torch
from torch import Tensor
from torch import stack as torch_stack
from torch.distributions import Transform
from torchvision import transforms


def get_augmenter_by_name(name, **kwargs):
    augmenter_args = kwargs["augmenter_args"] if "augmenter_args" in kwargs else {}

    if name == "transforms_batch":
        return TransformsBatchAugmenter(**augmenter_args)
    else:
        raise ValueError("Invalid augementer: {}".format(name))


def get_transformer_by_name(name, **kwargs):
    if name == "color_transformer":
        return create_color_transformer(**kwargs)
    else:
        raise ValueError("Invalid Transformer: {}".format(name))


def create_color_transformer(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5):
    """
    Creates transformer for randomly changing the colors of the input image
    :return:
    """

    # TODO: Checkout kornia for image transforms on GPU (https://github.com/kornia/kornia)
    return transforms.Compose([
        # CIFAR10Policy(),
        transforms.Lambda(lambda img: img / 255.0),  # TODO: Change obs range to [0, 1]
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation,
                               hue=hue),
        # transforms.Cutout(n_holes=2, length=),
        transforms.ToTensor(),
        transforms.Lambda(lambda img: img * 255.0),  # TODO: Change obs range to [0, 1]
    ])


class Augmenter(object):
    def augment_batch(self, batch, **kwargs):
        pass

    def augment_image(self, image, **kwargs):
        pass

    def calculate_loss(self, **kwargs):
        pass


class TransformsAugmenter(Augmenter):
    def __init__(self, transformer, **kwargs):

        assert transformer is not None

        transformer_args = kwargs["transformer_args"] if "transformer_args" in kwargs else None

        if type(transformer) == str:
            self.transformer = get_transformer_by_name(transformer, **transformer_args)
        else:
            self.transformer = transformer

        self.use_cnn_loss = kwargs["use_cnn_loss"] if "use_cnn_loss" in kwargs else False
        self.clip_aug_actions = kwargs["clip_aug_actions"] if "clip_aug_actions" in kwargs else False

    def _calculate_augmentation_loss(self, obs_batch, obs_batch_aug, **kwargs):

        assert "actor_critic" in kwargs
        assert "recurrent_hidden_states_batch" in kwargs
        assert "masks_batch" in kwargs

        additional_data = {}

        # Unpack
        actor_critic, \
        recurrent_hidden_states_batch, masks_batch = kwargs['actor_critic'], \
                                                     kwargs["recurrent_hidden_states_batch"], \
                                                     kwargs["masks_batch"]

        if self.use_cnn_loss:
            value_unlab, action_unlab, action_log_probs_unlab, rnn_hxs_unlab, cnn_output_unlab = \
                actor_critic.act(
                    obs_batch,
                    recurrent_hidden_states_batch,
                    masks_batch,
                    deterministic=True)

            value_unlab_aug, action_unlab_aug, action_log_probs_unlab_aug, rnn_hxs_unlab_aug, cnn_output_unlab_aug = \
                actor_critic.act(
                    obs_batch_aug,
                    recurrent_hidden_states_batch,
                    masks_batch,
                    deterministic=True)
        else:
            value_unlab, action_unlab, action_log_probs_unlab, rnn_hxs_unlab = \
                actor_critic.act(
                    obs_batch,
                    recurrent_hidden_states_batch,
                    masks_batch,
                    deterministic=True)

            value_unlab_aug, action_unlab_aug, action_log_probs_unlab_aug, rnn_hxs_unlab_aug = \
                actor_critic.act(
                    obs_batch_aug,
                    recurrent_hidden_states_batch,
                    masks_batch,
                    deterministic=True)

        # Clip actions
        if self.clip_aug_actions:
            action_unlab = action_unlab.clamp(-1, 1)
            action_unlab_aug = action_unlab_aug.clamp(-1, 1)

        # Detach action_unlab to prevent the gradient flow through the network
        if self.use_cnn_loss:
            aug_loss = torch.nn.functional.cross_entropy(cnn_output_unlab.detach(),
                                              cnn_output_unlab_aug)
        else:
            aug_loss = torch.nn.functional.mse_loss(action_unlab.detach(),
                                                           action_unlab_aug)

        # Determine max action for tracing actions
        additional_data["action_aug_max_value"] = torch.max(torch.max(action_unlab), torch.max(action_unlab_aug))

        return aug_loss, obs_batch_aug, additional_data


class TransformsBatchAugmenter(TransformsAugmenter):

    def calculate_loss(self, obs_batch, **kwargs):
        obs_batch_aug = self.augment_batch(obs_batch)
        return self._calculate_augmentation_loss(obs_batch, obs_batch_aug, **kwargs)

    def augment_batch(self, batch, **kwargs):
        """
        Augments a batch of images

        If the batch is a dictionary, use dict_key to specify which key contains the image
        """
        dict_key = kwargs["dict_key"] if "dict_key" in kwargs else "img"

        return self.__transform_obs_batch(batch, self.transformer, dict_key)

    def augment_image(self, image, **kwargs):
        """

        :param imgae:
        :param kwargs:
        :return:
        """
        return self.__transform_obs_img(image, transformer=self.transformer)

    def __transform_obs_img(self, obs_img: Tensor, transformer: Transform):
        """
        Transforms a single observation using PyTorch transformer
        :param obs_img: image tensor
        :param transformer: transformer used to transform the image
        :return: transformed obs_img
        """
        return transformer(obs_img.cpu())

    def __transform_obs_batch(self, obs, transformer, img_key="img"):
        """
        Wrapper function for transforming gym observations.
        :param obs: observations from gym environment
        :param transformer: PyTorch transformer which will be used to transform image
        :img_key: key which contains image in _obs_ dict
        :return:
        """
        # Retrieve original device
        device = obs[img_key].device

        # Create a detached copy of original observations
        obs_aug = {
            key: value.clone().detach() for (key, value) in obs.items()
        }

        # Unwrap batch of observations, apply transformation and create tensor of same shape as input
        transformed_obs = torch_stack(
            [self.__transform_obs_img(obs_img, transformer) for obs_img in obs_aug[img_key]])
        obs_aug[img_key] = transformed_obs
        obs_aug[img_key] = obs_aug[img_key].to(device)
        return obs_aug
