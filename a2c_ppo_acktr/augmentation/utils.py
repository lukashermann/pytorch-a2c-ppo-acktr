from torchvision import transforms

from a2c_ppo_acktr.augmentation import augmenters

def get_augmenter_by_name(name, **kwargs):
    augmenter_args = kwargs["augmenter_args"] if "augmenter_args" in kwargs else None

    if name == "transforms":
        return augmenters.TransformsAugmenter(**augmenter_args)


def get_transformer_by_name(name, **kwargs):
    if name == "color_transformer":
        return create_color_transformer(**kwargs)
    else:
        return create_color_transformer(**kwargs)


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


