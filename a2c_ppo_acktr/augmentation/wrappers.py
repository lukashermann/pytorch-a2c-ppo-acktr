from torch import stack as torch_stack

from gym import ObservationWrapper


class AugmentationObservationWrapper(ObservationWrapper):
    """
    Wraps observations and applies augmentations to observations with a specified degree of randomization
    """

    def __init__(self, env, transforms, default_degree=1.0):
        super(AugmentationObservationWrapper, self).__init__(env)
        self.augmentation_degree = default_degree
        self.transforms = transforms
        self.img_key_in_obs = "img"

    def observation(self, observation):
        if type(observation) == dict:
            if self.img_key_in_obs in observation:
                observation[self.img_key_in_obs] = torch_stack(
                    [self.transforms(obs_img) for obs_img in observation[self.img_key_in_obs]])
            else:
                raise ValueError("Observation of dict type requires 'img' key to be present")
        else:
            return self.transforms(observation)
        return observation

    def set_augmentation_degree(self, degree):
        self.augmentation_degree = degree