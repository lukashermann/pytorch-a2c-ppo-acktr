from torch import Tensor
from torch.distributions import Transform
from torch import stack as torch_stack

import a2c_ppo_acktr.augmentation.utils as transformer_utils


class Augmenter(object):
    def augment_batch(self, batch, **kwargs):
        pass

    def augment_image(self, image, **kwargs):
        pass


class TransformsAugmenter(Augmenter):

    def __init__(self, transformer, **kwargs):
        """
        """
        assert transformer is not None

        transformer_args = kwargs["transformer_args"] if "transformer_args" in kwargs else None

        if type(transformer) == str:
            self.transformer = transformer_utils.get_transformer_by_name(transformer, **transformer_args)
        else:
            self.transformer = transformer

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
        device = obs[img_key].get_device()

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
