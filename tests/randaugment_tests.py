import math
import unittest
from PIL import ImageChops, Image

from a2c_ppo_acktr.augmentation.randaugment import AUGMENTATION_LIST_DEFAULT, AUGMENTATION_LIST_SMALL_RANGE, Brightness, \
    Rotate, SymmetricAugmentation


def calculate_image_differences_rms(image_1, image_2):
    diff = ImageChops.difference(image_1, image_2)
    h = diff.histogram()
    sq = (value * ((idx % 256) ** 2) for idx, value in enumerate(h))
    sum_of_squares = sum(sq)
    rms = math.sqrt(sum_of_squares / float(image_1.size[0] * image_1.size[1]))
    return rms


def get_symmetric_augs():
    return [aug for aug in AUGMENTATION_LIST_DEFAULT if isinstance(aug, SymmetricAugmentation)]


def get_symmetric_augs_small_range():
    return [aug for aug in AUGMENTATION_LIST_SMALL_RANGE if isinstance(aug, SymmetricAugmentation)]


class RandAugmentTestCase(unittest.TestCase):
    def setUp(self):
        self.test_image = Image.open("data/test.png")

    def test_symmetric_augmentation_magnitude_0_gives_identity(self):
        symmetric_augs = get_symmetric_augs()
        for aug in symmetric_augs:
            magnitude = aug.scale_magnitude_to_aug_range(0.0)
            augmented_image = aug(self.test_image, magnitude)

            diff = calculate_image_differences_rms(self.test_image, augmented_image)
            self.assertEqual(0.0, diff)

    def test_symmetric_augmentation_small_range_magnitude_0_gives_identity(self):
        symmetric_augs = get_symmetric_augs_small_range()
        for aug in symmetric_augs:
            magnitude = aug.scale_magnitude_to_aug_range(0.0)
            augmented_image = aug(self.test_image, magnitude)

            diff = calculate_image_differences_rms(self.test_image, augmented_image)
            self.assertEqual(0.0, diff)


if __name__ == '__main__':
    unittest.main()
