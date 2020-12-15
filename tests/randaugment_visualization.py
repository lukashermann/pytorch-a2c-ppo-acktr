import argparse

import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid

from a2c_ppo_acktr.augmentation.randaugment import AUGMENTATION_LIST_DEFAULT, AUGMENTATION_LIST_SMALL_RANGE, \
    Cutout, RANDAUGMENT_MAP, RandAugment, StaticAugmentation


def load_test_img(img_path):
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
    parser.add_argument("--plot-grid", action='store_true')

    args = parser.parse_args()

    output_dir = args.output_dir
    test_img_file = args.image_file
    test_img = load_test_img(test_img_file)

    # augmentation_list = get_augmentation_list(args.augmentation_list)
    augmentation_list = [RANDAUGMENT_MAP["CutoutGrey"],
                         RANDAUGMENT_MAP["CutoutBlack"],
                         RANDAUGMENT_MAP["CutoutBlackSingleHole"],
                         RANDAUGMENT_MAP["CutoutGreySingleHole"],
                         RANDAUGMENT_MAP["CutoutBlackVariant"],
                         RANDAUGMENT_MAP["CutoutGreyVariant"]]

    magnitudes = np.linspace(0.0, 1.0, 5)

    if args.plot_grid:
        for augmentation in augmentation_list:
            if isinstance(augmentation, StaticAugmentation):
                fig, ax = plt.subplots(nrows=1, ncols=2, figsize=[8, 6])
                fig.suptitle("{}".format(augmentation.__class__.__name__))
                ax[0].imshow(test_img)
                ax[1].imshow(augmentation(test_img, None))
                ax[0].set_title("Original")
                ax[0].axis('off')
                ax[1].axis('off')
            else:
                fig, ax = plt.subplots(nrows=1, ncols=len(magnitudes), figsize=[8, 2.7])
                fig.suptitle("{}".format(augmentation.__class__.__name__))

                for i, axi in enumerate(ax.flat):
                    row_id = i
                    col_id = i % len(magnitudes)
                    magnitude = magnitudes[col_id]

                    aug_magnitude = augmentation.scale_magnitude_to_aug_range(magnitude)
                    img = augmentation(test_img, aug_magnitude)
                    axi.imshow(img)
                    axi.set_title("{}%".format(magnitude))
                    axi.axis('off')
            plt.tight_layout(True)
            plt.savefig("{}_{}.png".format(augmentation.__class__.__name__, args.augmentation_list))
    else:

        for augmentation in augmentation_list:
            for magnitude in np.linspace(0.0, 1.0, 5):
                if isinstance(augmentation, StaticAugmentation):
                    img = augmentation(test_img, None)
                    img.save(os.path.join(output_dir, "{}_{}.png".format(os.path.basename(test_img_file),
                                                                         str(augmentation))))
                    break
                else:
                    aug_magnitude = augmentation.scale_magnitude_to_aug_range(magnitude)
                    img = augmentation(test_img, aug_magnitude)
                    img.save(os.path.join(output_dir, "{}_{}_{}.png".format(os.path.basename(test_img_file),
                                                                            str(augmentation),
                                                                            magnitude)))
