"""
Static transformer creation for transforming observations
"""
from torch.distributions import Transform
from torchvision import transforms
from torch import stack as torch_stack, Tensor
import torchvision.transforms.functional as TF


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
