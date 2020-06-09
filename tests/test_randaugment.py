import argparse

import numpy as np
import os

from a2c_ppo_acktr.augmentation.randaugment import AUGMENTATION_LIST_DEFAULT, AUGMENTATION_LIST_SMALL_RANGE, RandAugment


def load_test_img(img_path):
    from PIL import Image
    img = Image.open(img_path)
    return img


def get_augmentation_list(augmentation_list):
    if augmentation_list == "default":
        return AUGMENTATION_LIST_DEFAULT
    elif augmentation_list == "small_range":
        return AUGMENTATION_LIST_SMALL_RANGE

    raise ValueError("Invalid Augmentation list")


if __name__ == '__main__':
    """
    This script displays the augmentations used in RandAugment
    """

    parser = argparse.ArgumentParser(description="Iterates all possible augmentation from given list and generates "
                                                 "augmented images for range between 0% and 100% augmentation")
    parser.add_argument("--output-dir", type=str, help="Folder where augmented images will be stored to.")
    parser.add_argument("--image-file", type=str, help="Image to be used for augmentation")
    parser.add_argument("--augmentation-list", choices=["default", "small_range"])

    args = parser.parse_args()

    output_dir = args.output_dir
    test_img_file = args.image_file
    test_img = load_test_img(test_img_file)

    augmentation_list = get_augmentation_list(args.augmentation_list)

    for augmentation in augmentation_list:
        for magnitude in np.linspace(0.0, 1.0, 10):
            aug_magnitude = augmentation.scale_magnitude_to_aug_range(magnitude)
            img = augmentation(test_img, aug_magnitude)
            img.save(os.path.join(output_dir, "{}_{}_{}.png".format(os.path.basename(test_img_file),
                                                                    augmentation.__class__.__name__, magnitude)))
