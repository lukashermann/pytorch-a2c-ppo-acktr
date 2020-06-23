import math
import unittest
from PIL import ImageChops, Image
import numpy as np

from a2c_ppo_acktr.augmentation.randaugment import AUGMENTATION_LIST_DEFAULT, AUGMENTATION_LIST_SMALL_RANGE, Brightness, \
    RandAugment, RangedAugmentation, Rotate, StaticAugmentation, SymmetricAugmentation


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


def get_ranged_augs():
    return [aug for aug in AUGMENTATION_LIST_DEFAULT if isinstance(aug, RangedAugmentation)]


def get_ranged_augs_small_range():
    return [aug for aug in AUGMENTATION_LIST_SMALL_RANGE if isinstance(aug, RangedAugmentation)]


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

    def test_ranged_augmentation_magnitude_0_gives_identity(self):
        ranged_augs = get_ranged_augs()
        for aug in ranged_augs:
            magnitude = aug.scale_magnitude_to_aug_range(0.0)
            augmented_image = aug(self.test_image, magnitude)

            diff = calculate_image_differences_rms(self.test_image, augmented_image)
            self.assertEqual(0.0, diff)

    def test_ranged_augmentation_small_range_magnitude_0_gives_identity(self):
        ranged_augs = get_ranged_augs_small_range()
        for aug in ranged_augs:
            magnitude = aug.scale_magnitude_to_aug_range(0.0)
            augmented_image = aug(self.test_image, magnitude)

            diff = calculate_image_differences_rms(self.test_image, augmented_image)
            self.assertEqual(0.0, diff)

    def test_augmentation_weights_sum_to_1(self):
        randaugment = RandAugment(1, magnitude=0.0)
        weights = randaugment.get_augmentation_weights_for_magnitude()
        self.assertAlmostEqual(sum(weights), 1.0)

        randaugment.set_magnitude(0.2)
        weights = randaugment.get_augmentation_weights_for_magnitude()
        self.assertAlmostEqual(sum(weights), 1.0)

        randaugment.set_magnitude(0.5)
        weights = randaugment.get_augmentation_weights_for_magnitude()
        self.assertAlmostEqual(sum(weights), 1.0)

        randaugment.set_magnitude(0.66)
        weights = randaugment.get_augmentation_weights_for_magnitude()
        self.assertAlmostEqual(sum(weights), 1.0)

        randaugment.set_magnitude(0.99)
        weights = randaugment.get_augmentation_weights_for_magnitude()
        self.assertAlmostEqual(sum(weights), 1.0)

        randaugment.set_magnitude(1.0)
        weights = randaugment.get_augmentation_weights_for_magnitude()
        self.assertAlmostEqual(sum(weights), 1.0)

    def test_static_augmentation_to_have_0_weight_for_0_magnitude(self):
        randaugment = RandAugment(1, magnitude=0.0)
        weights = randaugment.get_augmentation_weights_for_magnitude()
        static_weight_idx = np.array(
            [idx for idx, aug in enumerate(randaugment.augment_list) if isinstance(aug, StaticAugmentation)])
        static_weights = np.array(weights)[static_weight_idx]
        for weight in static_weights:
            self.assertEqual(weight, 0.0)

    def test_static_augmentation_weight_to_be_equal_for_all_augs_at_half_magnitude(self):
        randaugment = RandAugment(1, magnitude=0.5)
        weights = randaugment.get_augmentation_weights_for_magnitude()

        for weight in weights:
            self.assertEqual(weight, 1 / len(weights))

    def test_random_choice_of_augmentation_at_magnitude_0_does_not_contain_static_augs(self):
        for i in range(100):  # Repeat experiment
            randaugment = RandAugment(100, magnitude=0.0)
            augs = randaugment.choose_augs_by_magnitude()
            for aug in augs:
                self.assertIsNot(isinstance(aug, StaticAugmentation), True)


if __name__ == '__main__':
    unittest.main()
