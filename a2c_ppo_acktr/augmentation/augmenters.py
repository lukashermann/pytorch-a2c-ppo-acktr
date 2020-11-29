import torch
from a2c_ppo_acktr.augmentation.randaugment import AUGMENTATION_LIST_SMALL_RANGE, RandAugment, \
    AUGMENTATION_LIST_DEFAULT, AUGMENTATION_LIST_OURS, RANDAUGMENT_MAP
import a2c_ppo_acktr.augmentation.randaugment as randaugment

from torch import Tensor
from torch import stack as torch_stack
from torch.distributions import Transform
from torchvision import transforms

DEFAULT_COLOR_JITTER = {"brightness": 0.5,
                        "contrast": 0.5,
                        "saturation": 0.5,
                        "hue": 0.5}

def encode_target(target_action_probs: torch.Tensor):
    """
    TODO: Refactor to utils method
    :param target_action_probs: probabilities of actions
    :return: maximized target actions

    Example:
        input: [0.5, 0.3, 0.2]
        output: [1.0, 0, 0]

    >>> input = [0.5, 0.3, 0.2]
    >>> output = encode_target(input)
    >>> output
    [1.0, 0.0, 0.0]
    """
    results = []
    for probs in target_action_probs:
        max_idx = torch.argmax(probs, 1)
        # one_hot = torch.FloatTensor(probs.shape)
        # one_hot = one_hot.zero_()
        # one_hot = one_hot.scatter_(1, max_idx, 1.0)
        results.append(max_idx)
    return results

def get_augmenter_by_name(name, **kwargs):
    augmenter_args = kwargs["augmenter_args"] if "augmenter_args" in kwargs else {}

    if name == "transforms_batch":
        return TransformsBatchAugmenter(**augmenter_args)
    else:
        raise ValueError("Invalid augementer: {}".format(name))


def get_transformer_by_name(name, **kwargs):
    """
    Returns: Transformer instance with given parameters. See randaugment module for supported types

    Args:
        name:
        **kwargs:

    get_transformer_by_name("RandAugment", **{"num_augmentations": 10})

    """
    if name == "color_transformer":
        return create_color_transformer(**kwargs)

    kwargs = get_transformer_args_from_dict(**kwargs)
    try:
        augmenter = getattr(randaugment, name)(**kwargs)
    except:
        raise ValueError("Invalid RandAugment class: {}".format(name))

    return transforms_with_randaugment(augmenter, **kwargs)


def get_transformer_args_from_dict(**kwargs):
    """
    Parses augmentation arguments from dict

    Main purpose: create augmentation_list from configuration file.

    List of Augmentations
    >>> args = {'num_augmentations': 3, 'magnitude': 0.3, 'augmentation_list': ['CenterCropAndResize', 'Rotate']}
    >>> get_transformer_args_from_dict(**args)
    {'num_augmentations': 3, 'magnitude': 0.3, 'augmentation_list': [CenterCropAndResize[0.0, 1.0], Rotate[-30, 30]]}


    List of Augmentations with custom init arguments
    >>> args = {'num_augmentations': 3, 'magnitude': 0.3, 'augmentation_list': [{'CenterCropAndResize': {'min_value': 0.2, 'max_value': 0.5}}, 'Rotate']}
    >>> get_transformer_args_from_dict(**args)
    {'num_augmentations': 3, 'magnitude': 0.3, 'augmentation_list': [CenterCropAndResize[0.2, 0.5], Rotate[-30, 30]]}

    Single Augmentation
    >>> args = {'num_augmentations': 3, 'magnitude': 0.3, 'augmentation_list': 'CenterCropAndResize'}
    >>> get_transformer_args_from_dict(**args)
    {'num_augmentations': 3, 'magnitude': 0.3, 'augmentation_list': [CenterCropAndResize[0.0, 1.0]]}

    Single augmentation with custom init arguments
    >>> args = {'num_augmentations': 3, 'magnitude': 0.3, 'augmentation_list': {'CenterCropAndResize': {'min_value': 0.2, 'max_value': 0.5}}}
    >>> get_transformer_args_from_dict(**args)
    {'num_augmentations': 3, 'magnitude': 0.3, 'augmentation_list': [CenterCropAndResize[0.2, 0.5]]}

    Predefined List
    >>> args = {'num_augmentations': 3, 'magnitude': 0.3, 'augmentation_list': 'AUGMENTATION_LIST_DEFAULT'}
    >>> get_transformer_args_from_dict(**args)
    {'num_augmentations': 3, 'magnitude': 0.3, 'augmentation_list': [Identity, AutoContrast, Equalize, Rotate[-30, 30], Solarize[0, 254], Color[0.1, 1.9], Posterize[4, 8], Contrast[0.1, 1.9], Brightness[0.1, 1.9], Sharpness[0.1, 1.9], ShearX[-0.3, 0.3], ShearY[-0.3, 0.3], TranslateX[-0.45, 0.45], TranslateY[-0.45, 0.45]]}

    Invalid predefined List
    >>> args = {'num_augmentations': 3, 'magnitude': 0.3, 'augmentation_list': 'SOME_LIST'}
    >>> get_transformer_args_from_dict(**args)
    Traceback (most recent call last):
    ...
    ValueError: Invalid parameter: augmentation_list SOME_LIST
    """
    if "augmentation_list" in kwargs:
        augmentation_list = []

        augmentations_list_args = kwargs["augmentation_list"]

        if type(augmentations_list_args) == list:
            for augmenter in augmentations_list_args:
                augmentation_list.append(create_augmentation_from_dict(augmenter))
        else:
            # Either preconfigured List or single augmentation
            if type(augmentations_list_args) == dict:
                augmentation_list.append(create_augmentation_from_dict(augmentations_list_args))
            elif augmentations_list_args in RANDAUGMENT_MAP:
                augmentation_list.append(create_augmentation_from_dict(augmentations_list_args))
            elif augmentations_list_args == "AUGMENTATION_LIST_DEFAULT":
                augmentation_list = AUGMENTATION_LIST_DEFAULT
            elif augmentations_list_args == "AUGMENTATION_LIST_SMALL_RANGE":
                augmentation_list = AUGMENTATION_LIST_SMALL_RANGE
            elif augmentations_list_args == "AUGMENTATION_LIST_OURS":
                augmentation_list = AUGMENTATION_LIST_OURS
            else:
                raise ValueError("Invalid parameter: augmentation_list {}".format(augmentations_list_args))

        kwargs["augmentation_list"] = augmentation_list
    return kwargs


def create_augmentation_from_dict(augmenter_args):
    """
    Args:
        augmenter_args:

    Returns:

    """
    if type(augmenter_args) == str:
        return RANDAUGMENT_MAP[augmenter_args]
    elif type(augmenter_args) == dict:
        augmentation_name = list(augmenter_args.keys())[0]
        augmentation_params = augmenter_args[augmentation_name]

        # Instantiate object and set attributes of object
        augmentation = getattr(randaugment, augmentation_name)(**augmentation_params)

        return augmentation
    else:
        raise ValueError("Unknown Augmentation {}".format(augmenter_args))


def transforms_with_randaugment(randaugment, with_color_jitter=False, color_jitter_params=DEFAULT_COLOR_JITTER, **kwargs):
    transforms_list = [
        transforms.Lambda(lambda img: img / 255.0),  # TODO: Change obs range to [0, 1]
        transforms.ToPILImage()
    ]

    transforms_list.append(randaugment)
    if with_color_jitter:
        transforms_list.append(transforms.ColorJitter(**color_jitter_params))

    transforms_list.append(transforms.ToTensor())
    transforms_list.append(transforms.Lambda(lambda img: img * 255.0))  # TODO: Change obs range to [0, 1])
    return transforms.Compose(transforms_list)


def create_color_transformer(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5):
    """
    Creates transformer for randomly changing the colors of the input image
    :return:
    """
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

        if type(transformer) == str:
            transformer_args = kwargs["transformer_args"] if "transformer_args" in kwargs else None
            self.transformer = get_transformer_by_name(transformer, **transformer_args)
        else:
            self.transformer = transformer

        self.use_cnn_loss = kwargs["use_cnn_loss"] if "use_cnn_loss" in kwargs else False
        self.with_actions_probs = kwargs["with_action_probs"] if "with_action_probs" in kwargs else False
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
            with torch.no_grad():
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
        elif self.with_actions_probs:
            with torch.no_grad():
                value_unlab, action_unlab, action_log_probs_unlab, rnn_hxs_unlab, action_probs = \
                    actor_critic.act(
                        obs_batch,
                        recurrent_hidden_states_batch,
                        masks_batch,
                        deterministic=True)

            value_unlab_aug, action_unlab_aug, action_log_probs_unlab_aug, rnn_hxs_unlab_aug, action_probs_aug = \
                actor_critic.act(
                    obs_batch_aug,
                    recurrent_hidden_states_batch,
                    masks_batch,
                    deterministic=True)
            action_probs_aug = encode_target(action_probs_aug)
        else:
            with torch.no_grad():
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
            aug_loss = torch.nn.functional.mse_loss(action_unlab.detach(), action_unlab_aug)
            # Cosine similarity is defined between -1 to 1, where 1 is most similar -> Flip via 1 - loss
            # aug_loss = 1 - torch.nn.functional.cosine_similarity(cnn_output_unlab.detach(), cnn_output_unlab_aug).mean()
        elif self.with_actions_probs:
            losses = []
            for a, t in zip(action_probs, action_probs_aug):
                losses.append(torch.nn.functional.cross_entropy(a, t))
            aug_loss = sum(losses)
        else:
            aug_loss = torch.nn.functional.mse_loss(action_unlab.detach(), action_unlab_aug)

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
