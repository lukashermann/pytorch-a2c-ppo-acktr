"""
Static transformer creation for transforming observations
"""
from torchvision import transforms
from torch import stack as torch_stack
import torchvision.transforms.functional as TF

def get_transformer():
    transform_ori = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_aug = transforms.Compose([
        # CIFAR10Policy(),
        transforms.ToTensor(),
        # Cutout(n_holes=args.n_holes, length=args.cutout_size),
        transforms.ToPILImage(),
        # transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    return transform_aug

def transform_obs_img(obs_img, transformer):
    """
    Transforms a single observation using PyTorch transformer
    :param obs_img:
    :return:
    """
    # Convert to PIL
    obs_img = TF.to_pil_image(obs_img.cpu())
    res = transformer(obs_img)
    # imgs = transforms.ToPILImage(obs for obs in obs['img'])
    # # Apply transformation
    # transformed_obs = transformer(imgs)

    return res

def transform_obs_batch(obs, transformer, img_key="img"):
    """
    Wrapper function for transforming gym observations.
    :param obs: observations from gym environment
    :param transformer: PyTorch transformer which will be used to transform image
    :img_key: key which contains image in _obs_ dict
    :return:
    """
    # Retrieve original device
    device = obs["img"].get_device()

    # Create a detached copy of original observations
    obs_aug = {
        key: value.clone().detach() for (key, value) in obs.items()
    }

    # Unwrap batch of observations, apply transformation and create tensor of same shape as input
    transformed_obs = torch_stack([transform_obs_img(obs_img, transformer) for obs_img in obs_aug["img"]])
    obs_aug["img"] = transformed_obs
    obs_aug["img"] = obs_aug["img"].to(device)
    return obs_aug



