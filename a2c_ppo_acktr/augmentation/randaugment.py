"""
This file contains an adapted implementation of RandAugment
Adapted from:

Our implementation allows dynamically changing the randomization, allowing this to be used in our evaluation.
"""
import random

import PIL
import PIL.ImageDraw
import PIL.ImageEnhance
import PIL.ImageOps
from PIL import Image
import abc
import numpy as np
from typing import List, Union


class Augmentation(abc.ABC):
    """Augmentation base class"""

    def scale_magnitude_to_aug_range(self, magnitude=None):
        pass

    def __repr__(self):
        return type(self).__name__


class StaticAugmentation(Augmentation):
    """Base class for augmentations without magnitude"""

    def __call__(self, img, _=None):
        pass


class RangedAugmentation(Augmentation):
    def __init__(self, min_value=0.0, max_value=1.0, reverse=False):
        """
        Augmentation with upper and lower range, and optional reverted scale.
        Args:
            min_value: minimum value for the augmentation
            max_value: maximum value for the augmentation
            reverse: if True, magnitude is reversed for min/max value.
        """
        self.min_value = min_value
        self.max_value = max_value
        self.reverse = reverse

    def scale_magnitude_to_aug_range(self, magnitude):
        """
        Scales given magnitude to augmentation specific range
        Args:
            magnitude: magnitude of augmentation between 0.0 and 1.0

        Returns:
            magnitude in the range of the augmentation
        >>> augmentation = RangedAugmentation(0.0, 10.0)
        >>> augmentation.scale_magnitude_to_aug_range(0.5)
        5.0
        >>> augmentation.scale_magnitude_to_aug_range(1.0)
        10.0
        >>> augmentation.scale_magnitude_to_aug_range(0.0)
        0.0
        >>> augmentation = RangedAugmentation(0, 10, reverse=True)
        >>> augmentation.scale_magnitude_to_aug_range(0.0)
        10.0
        >>> augmentation.scale_magnitude_to_aug_range(1.0)
        0.0
        >>> augmentation.scale_magnitude_to_aug_range(0.5)
        5.0
        """
        if self.reverse:
            magnitude = 1.0 - magnitude
        return magnitude * float(self.max_value - self.min_value) + self.min_value

    def __call__(self, img: PIL.Image, magnitude):
        """
        Override this method to apply augmentation with given magnitude

        Args:
            img: PIL image to be augmented
            magnitude: Magnitude in correct range for given augmentation. Use Augmentation.scale_to_range() to
                        scale to range.
        Returns:
            Augmented image
        >>> augmentation = Augmentation(-1.0, 1.0)
        >>> augmentation(None, 1.0) # Assertion is fine, no output

        >>> augmentation = Augmentation(5, 10.0)
        >>> augmentation(None, 11)  # magnitude > max magnitude value
        Traceback (most recent call last):
        AssertionError
        >>> # magnitude < min magnitude value
        >>> augmentation(None, 4)
        Traceback (most recent call last):
        AssertionError
        """
        assert self.min_value <= magnitude <= self.max_value

    def __repr__(self):
        return type(self).__name__ + "[" + str(self.min_value) + ", " + str(self.max_value) + "]"


class SymmetricAugmentation(RangedAugmentation):
    """
    Augmentations of this type will be scaled from mean(min_value, max_value) to max_value with a random sign flip
    """

    def __init__(self, min_value, max_value, randomize_sign_threshold=0.5):
        super().__init__(min_value, max_value)
        self.randomize_sign_threshold = randomize_sign_threshold

    def scale_magnitude_to_aug_range(self, magnitude):
        lower_value = np.mean((self.min_value, self.max_value))

        # Randomize sign for symmetric augmentations
        sign = 1
        rand = random.random()
        if rand > self.randomize_sign_threshold:
            sign = -1
        return sign * magnitude * float(self.max_value - lower_value) + lower_value


class Identity(StaticAugmentation):
    def __call__(self, img, _=None):
        return img


class AutoContrast(StaticAugmentation):
    def __call__(self, img, _=None):
        return PIL.ImageOps.autocontrast(img)


class Invert(StaticAugmentation):
    def __call__(self, img, _=None):
        return PIL.ImageOps.invert(img)


class Equalize(StaticAugmentation):
    def __call__(self, img, _=None):
        return PIL.ImageOps.equalize(img)


class Flip(StaticAugmentation):
    def __call__(self, img, _=None):
        return PIL.ImageOps.mirror(img)


class ShearX(SymmetricAugmentation):
    def __init__(self, min_value=-0.3, max_value=0.3):
        super().__init__(min_value, max_value)

    def __call__(self, img, magnitude):
        super().__call__(img, magnitude)
        return img.transform(img.size, PIL.Image.AFFINE, (1, magnitude, 0, 0, 1, 0))


class ShearY(SymmetricAugmentation):
    def __init__(self, min_value=-0.3, max_value=0.3):
        super().__init__(min_value, max_value)

    def __call__(self, img, magnitude):
        super().__call__(img, magnitude)
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, magnitude, 1, 0))


class TranslateX(SymmetricAugmentation):
    def __init__(self, min_value=-0.45, max_value=0.45):
        super().__init__(min_value, max_value)

    def __call__(self, img, magnitude):
        super().__call__(img, magnitude)

        magnitude = magnitude * img.size[0]
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, magnitude, 0, 1, 0))


class TranslateY(SymmetricAugmentation):
    def __init__(self, min_value=-0.45, max_value=0.45):
        super().__init__(min_value, max_value)

    def __call__(self, img, magnitude):
        super().__call__(img, magnitude)

        if random.random() > 0.5:
            magnitude = -magnitude
        magnitude = magnitude * img.size[1]
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, magnitude))


class Rotate(SymmetricAugmentation):
    def __init__(self, min_value=-30, max_value=30):
        super().__init__(min_value, max_value)

    def __call__(self, img, magnitude):
        super().__call__(img, magnitude)

        if random.random() > 0.5:
            magnitude = -magnitude
        return img.rotate(magnitude)


class Contrast(SymmetricAugmentation):
    def __init__(self, min_value=0.1, max_value=1.9):
        super().__init__(min_value, max_value)

    def __call__(self, img, magnitude):
        super().__call__(img, magnitude)
        return PIL.ImageEnhance.Contrast(img).enhance(magnitude)


class Color(SymmetricAugmentation):
    def __init__(self, min_value=0.1, max_value=1.9):
        super().__init__(min_value, max_value)

    def __call__(self, img, magnitude):
        super().__call__(img, magnitude)
        return PIL.ImageEnhance.Color(img).enhance(magnitude)


class Brightness(SymmetricAugmentation):
    def __init__(self, min_value=0.1, max_value=1.9):
        super().__init__(min_value, max_value)

    def __call__(self, img, magnitude):
        super().__call__(img, magnitude)
        return PIL.ImageEnhance.Brightness(img).enhance(magnitude)


class Sharpness(SymmetricAugmentation):
    def __init__(self, min_value=0.1, max_value=1.9):
        super().__init__(min_value, max_value)

    def __call__(self, img, magnitude):
        super().__call__(img, magnitude)
        return PIL.ImageEnhance.Sharpness(img).enhance(magnitude)


class Solarize(RangedAugmentation):
    def __init__(self, min_value=0, max_value=254):
        super().__init__(min_value, max_value, reverse=True)

    def __call__(self, img, magnitude):
        super().__call__(img, magnitude)
        return PIL.ImageOps.solarize(img, magnitude)


class Posterize(RangedAugmentation):
    def __init__(self, min_value=4, max_value=8):
        super().__init__(min_value, max_value, reverse=True)

    def __call__(self, img, magnitude):
        super().__call__(img, magnitude)
        magnitude = max(1, int(magnitude))
        return PIL.ImageOps.posterize(img, magnitude)


class Cutout(RangedAugmentation):
    black = (0, 0, 0)
    grey = (125, 123, 114)

    def __init__(self, hole_size=20, fixed_number_of_holes=-1, fill_color=grey, max_number_of_holes=4):
        """

        :param hole_size: size of holes
        :param fixed_number_of_holes: if set, the number of holes is fixed (regardless of magnitude)
        :param fill_color: color to be used to fill holes (r,g,b) int tuple
        :param max_number_of_holes: maximum number of holes ()
        """
        super().__init__(min_value=0.0, max_value=1.0)
        self.max_h_size = hole_size
        self.max_w_size = hole_size
        self.fill_color = fill_color

        self.max_number_of_holes = max_number_of_holes
        self.fixed_number_of_holes = fixed_number_of_holes

    def cutout_abs(self, img, num_holes=1):
        if self.fixed_number_of_holes > 0:
            num_holes = self.fixed_number_of_holes

        height, width = img.size

        holes = []
        for _n in range(num_holes):
            y = random.randint(0, height)
            x = random.randint(0, width)

            y1 = np.clip(y - self.max_h_size // 2, 0, height)
            y2 = np.clip(y1 + self.max_h_size, 0, height)
            x1 = np.clip(x - self.max_w_size // 2, 0, width)
            x2 = np.clip(x1 + self.max_w_size, 0, width)
            holes.append((x1, y1, x2, y2))

        img = img.copy()
        for hole in holes:
            PIL.ImageDraw.Draw(img).rectangle(hole, self.fill_color)
        return img

    def __call__(self, img, magnitude):
        super().__call__(img, magnitude)
        if magnitude <= 0.0:
            return img

        magnitude = int(magnitude * self.max_number_of_holes)

        # Make sure at least one hole is present
        if magnitude == 0:
            magnitude = 1

        return self.cutout_abs(img, num_holes=magnitude)

    def __repr__(self):
        name = type(self).__name__
        if self.fixed_number_of_holes > 0:
            name += "_fixed_holes_" + str(self.fixed_number_of_holes)
        else:
            name += "_max_holes_" + str(self.max_number_of_holes)

        name += "_color_" + str(self.fill_color)
        return name


class CenterCropAndResize(RangedAugmentation):
    def __init__(self, min_value=0.0, max_value=1.0):
        super().__init__(min_value, max_value)

    def __call__(self, img, magnitude):
        super().__call__(img, magnitude)

        output_size = img.size
        # Crop by size of image/2 * magnitude, scale by 1/2 to make croping less prominent
        crop_size = img.size[0] / 2 * magnitude / 2
        crop_size = min(crop_size, 20)  # Set 20 pixel as min crop size

        img_cropped = PIL.ImageOps.crop(img, crop_size)
        img_resized = img_cropped.resize(output_size, Image.BILINEAR)

        return img_resized


class RandomCrop(RangedAugmentation):
    def __init__(self, min_value=0.0, max_value=1.0):
        super().__init__(min_value, max_value)

    def __call__(self, img, magnitude):
        super().__call__(img, magnitude)

        # Sample cropping area

        output_size = img.size
        # Crop by size of image/2 * magnitude, scale by 1/2 to make croping less prominent
        crop_size = img.size[0] / 2 * magnitude / 2
        crop_size = min(crop_size, 20)  # Set 20 pixel as min crop size

        img_cropped = PIL.ImageOps.crop(img, crop_size)
        img_resized = img_cropped.resize(output_size, Image.BILINEAR)

        return img_resized


# List taken from RandAugment Paper
AUGMENTATION_LIST_DEFAULT = [Identity(), AutoContrast(), Equalize(),
                             Rotate(), Solarize(), Color(),
                             Posterize(), Contrast(), Brightness(),
                             Sharpness(), ShearX(), ShearY(),
                             TranslateX(), TranslateY()]

AUGMENTATION_LIST_SMALL_RANGE = [Identity(), AutoContrast(), Equalize(),
                                 Rotate(min_value=-10, max_value=10),
                                 Solarize(min_value=128, max_value=254), Color(),
                                 Posterize(min_value=6, max_value=8),
                                 Contrast(min_value=0.8, max_value=1.2),
                                 Brightness(min_value=0.8, max_value=1.2),
                                 Sharpness(), ShearX(min_value=-0.1, max_value=0.1),
                                 ShearY(min_value=-0.1, max_value=0.1),
                                 TranslateX(min_value=-0.1, max_value=0.1),
                                 TranslateY(min_value=-0.1, max_value=0.1)]

AUGMENTATION_LIST_OURS = [CenterCropAndResize(),
                          Identity(),
                          Rotate(),
                          Solarize(),
                          Color(),
                          Posterize(),
                          Contrast(),
                          Brightness(),
                          Sharpness(),
                          ShearX(),
                          ShearY()]

RANDAUGMENT_MAP = {
    "Identity": Identity(),
    "AutoContrast": AutoContrast(),
    "Equalize": Equalize(),
    "Rotate": Rotate(),
    "Rotate_ours": Rotate(min_value=-10, max_value=10),
    "Solarize": Solarize(),
    "Solarize_ours": Solarize(min_value=128, max_value=254),
    "Color": Color(),
    "Posterize": Posterize(),
    "Posterize_ours": Posterize(min_value=6, max_value=8),
    "Contrast": Contrast(),
    "Contrast_ours": Contrast(min_value=0.8, max_value=1.2),
    "Brightness": Brightness(),
    "Brightness_ours": Brightness(min_value=0.8, max_value=1.2),
    "Sharpness": Sharpness(),
    "ShearX": ShearX(),
    "ShearX_ours": ShearX(min_value=-0.1, max_value=0.1),
    "ShearY": ShearY(),
    "ShearY_ours": ShearY(min_value=-0.1, max_value=0.1),
    "TranslateX": TranslateX(),
    "TranslateY": TranslateY(),
    "CenterCropAndResize": CenterCropAndResize(),
    "CutoutBlack": Cutout(fill_color=(0, 0, 0)),
    "CutoutGrey": Cutout(fill_color=(125, 123, 114)),
    "CutoutBlackSingleHole": Cutout(fill_color=(0, 0, 0), fixed_number_of_holes=1),
    "CutoutGreySingleHole": Cutout(fill_color=(125, 123, 114), fixed_number_of_holes=1),
    "CutoutBlackVariant": Cutout(fill_color=(0, 0, 0), hole_size=5, max_number_of_holes=10),
    "CutoutGreyVariant": Cutout(fill_color=(125, 123, 114), hole_size=5, max_number_of_holes=10)
}


class RandAugment:
    """
    Randomly choose specified number of augmentations and apply them sequentially with specified magnitude.
    """

    def __init__(self, num_augmentations: int = 1, magnitude: float = 0.0,
                 augmentation_list: List = AUGMENTATION_LIST_DEFAULT,
                 min_magnitude: Union[float, int] = 0.0, max_magnitude: Union[float, int] = 1.0, **kwargs):
        """
        Args:
            num_augmentations: Number of augmentation transformations to apply sequentially.
            magnitude: Magnitude for all the transformations [float between 0.0..1.0].
            augmentation_list: list of augmentations to be applied
            min_magnitude: defines the minimum magnitude used for augmentations.
            max_magnitude: defines the maximum magnitude used for augmentations
        """
        self.num_augmentations = num_augmentations
        self.augment_list = augmentation_list

        assert min_magnitude <= max_magnitude
        self.__min_magnitude = min_magnitude
        self.__max_magnitude = max_magnitude

        # Don't set this value manually, always use setter
        self.__normalized_magnitude = None
        self.set_magnitude(magnitude)

    def set_magnitude(self, magnitude: Union[float, int]):
        """
        Sets the magnitude for augmentations. Will be normalized from [min_magnitude, max_magnitude] to [0.0...1.0]
        Args:
            magnitude: value between min_magnitude and max_magnitude

        >>> rand_aug = RandAugment(1, 30, max_magnitude=30)
        >>> rand_aug.magnitude()
        1.0
        >>> rand_aug.set_magnitude(15)
        >>> rand_aug.magnitude()
        0.5
        >>> rand_aug.set_magnitude(6)
        >>> rand_aug.magnitude()
        0.2
        >>> rand_aug.set_magnitude(-1)  # Value < min_magnitude
        Traceback (most recent call last):
        AssertionError
        >>> rand_aug.set_magnitude(31)  # Value > max_magnitude
        Traceback (most recent call last):
        AssertionError
        """
        assert self.__min_magnitude <= magnitude <= self.__max_magnitude

        normalized_magnitude = (magnitude - self.__min_magnitude) / (
                self.__max_magnitude - self.__min_magnitude)
        self.__normalized_magnitude = normalized_magnitude

    def __call__(self, img: PIL.Image) -> PIL.Image:
        # https://stackoverflow.com/questions/3679694/a-weighted-version-of-random-choice
        augs = self.choose_augs_by_magnitude()
        for augmentation in augs:
            aug_magnitude = augmentation.scale_magnitude_to_aug_range(magnitude=self.magnitude())
            img = augmentation(img, aug_magnitude)
        return img

    def magnitude(self) -> float:
        return self.__normalized_magnitude

    def choose_augs_by_magnitude(self):
        return random.choices(self.augment_list, weights=self.get_augmentation_sample_weights_for_magnitude(),
                              k=self.num_augmentations)

    def get_augmentation_sample_weights_for_magnitude(self):
        """
        Returns: weights for augmentations, one weight for each augmentation in self.augment_list
            The weight depends on the magnitude of randaugment. For magnitude -> 0.0 this method will return 0.0 weight
            for augmentation which have no range. This will make sure that static augmentations (without magnitude) are
            less common for small magnitudes. All augmentations are equally likely for magnitude == 1.
#         >>> randaugment = RandAugment(1, magnitude=0.0)
#         >>> randaugment.get_augmentation_weights_for_magnitude()
#         [0.0, 0.0, 0.0, 0.09090909090909091, 0.09090909090909091, 0.09090909090909091, 0.09090909090909091, \
# 0.09090909090909091, 0.09090909090909091, 0.09090909090909091, 0.09090909090909091, 0.09090909090909091, \
# 0.09090909090909091, 0.09090909090909091]
#         >>> randaugment = RandAugment(1, magnitude=1.0)
#         >>> randaugment.get_augmentation_weights_for_magnitude()
#         [0.07142857142857142, 0.07142857142857142, 0.07142857142857142, 0.07142857142857142, 0.07142857142857142, \
# 0.07142857142857142, 0.07142857142857142, 0.07142857142857142, 0.07142857142857142, 0.07142857142857142, \
# 0.07142857142857142, 0.07142857142857142, 0.07142857142857142, 0.07142857142857142]
#         >>> randaugment = RandAugment(1, magnitude=0.5)
#         >>> randaugment.get_augmentation_weights_for_magnitude()
#         [0.03571428571428571, 0.03571428571428571, 0.03571428571428571, 0.08116883116883117, 0.08116883116883117, \
# 0.08116883116883117, 0.08116883116883117, 0.08116883116883117, 0.08116883116883117, 0.08116883116883117, \
# 0.08116883116883117, 0.08116883116883117, 0.08116883116883117, 0.08116883116883117]
#         >>> randaugment = RandAugment(1, magnitude=0.2)
#         >>> randaugment.get_augmentation_weights_for_magnitude()
#         [0.014285714285714285, 0.014285714285714285, 0.014285714285714285, 0.08701298701298701, 0.08701298701298701, \
# 0.08701298701298701, 0.08701298701298701, 0.08701298701298701, 0.08701298701298701, 0.08701298701298701, \
# 0.08701298701298701, 0.08701298701298701, 0.08701298701298701, 0.08701298701298701]
        >>> # Only Static Aug
        >>> randaugment = RandAugment(1, magnitude=0.2, augmentation_list=[Identity()])
        >>> randaugment.get_augmentation_sample_weights_for_magnitude()
        [0.2]
        >>> randaugment = RandAugment(1, magnitude=1.0, augmentation_list=[Identity()])
        >>> randaugment.get_augmentation_sample_weights_for_magnitude()
        [1.0]
        >>> # Only Ranged Aug
        >>> randaugment = RandAugment(1, magnitude=0.3, augmentation_list=[ShearX()])
        >>> randaugment.get_augmentation_sample_weights_for_magnitude()
        [1.0]
        >>> randaugment = RandAugment(1, magnitude=0.3, augmentation_list=[ShearX(), ShearY(), Identity()])
        >>> randaugment.get_augmentation_sample_weights_for_magnitude()
        [0.45, 0.45, 0.09999999999999999]
        >>> randaugment = RandAugment(1, magnitude=1.0, augmentation_list=[ShearX(), ShearY(), Identity()])
        >>> randaugment.get_augmentation_sample_weights_for_magnitude()
        [0.33333333333333337, 0.33333333333333337, 0.3333333333333333]
        """
        # Extract all static / non-static augmentations
        static_augs = np.array(
            [aug for aug in self.augment_list if isinstance(aug, StaticAugmentation)])
        non_static_augs = np.array(
            [aug for aug in self.augment_list if isinstance(aug, RangedAugmentation)])

        # Define maximum weight each augmentation can get
        max_weight = 1.0 / len(self.augment_list)

        # Use linear function to derive static weight
        # f(x) = x * max_weight for x <= 1.0
        static_weight = self.__normalized_magnitude * max_weight
        # Calculate other weights depending on static weight (makes sure result is a prob. distribution)
        other_weight = (1 - (static_weight * len(static_augs))) / len(non_static_augs) if len(
            non_static_augs) > 0 else 0.0
        return [static_weight if isinstance(aug, StaticAugmentation) else other_weight for aug in
                self.augment_list]


class SingleSampleRandAugment(RandAugment):
    """
    Alternative implementation of RandAugment where each augmentation is only sampled once.
    """

    def __init__(self, num_augmentations: int = 1, magnitude: float = 1.0,
                 augmentation_list: List = AUGMENTATION_LIST_DEFAULT,
                 min_magnitude: Union[float, int] = 0.0, max_magnitude: Union[float, int] = 1.0, **kwargs):
        # Set magnitude to 1.0 to sample all augmentations with same probability
        super(SingleSampleRandAugment, self).__init__(num_augmentations=num_augmentations, magnitude=1.0,
                                                      augmentation_list=augmentation_list,
                                                      min_magnitude=min_magnitude, max_magnitude=max_magnitude)

        assert len(augmentation_list) >= num_augmentations

    def choose_augs_by_magnitude(self):
        """
        Returns: list of augmentations from augmentation list
        """

        sample = set()
        population = self.augment_list.copy()
        weights = self.get_augmentation_sample_weights_for_magnitude()
        while len(sample) < self.num_augmentations:
            choice = random.choices(population, weights=weights, k=1)[0]  # single choice
            sample.add(choice)
            index = population.index(choice)
            weights.pop(index)
            population.remove(choice)
            weights = [x / sum(weights) for x in weights]
        return list(sample)


class RandomMagnitudeRandaugment(RandAugment):
    def __init__(self, num_augmentations: int = 1, magnitude: float = 1.0,
                 augmentation_list: List = AUGMENTATION_LIST_DEFAULT,
                 min_magnitude: Union[float, int] = 0.0, max_magnitude: Union[float, int] = 1.0,
                 sample_min_magnitude=0.0, sample_max_magnitude=1.0, **kwargs):
        """
        Randomly samples a magnitude for each augmentation when RandAugment is called.

        Args:
            magnitude: Used for creating sample weights for augmentations (see super)
            sample_min_magnitude: set maximum magnitude when sampling magnitude
                Unrelated to min_magnitude, which is used to scale base magnitude (influencing augmentation sampling)
            sample_max_magnitude: set maximum magnitude when sampling magnitude
                Unrelated to max_magnitude, which is used to scale base magnitude (influencing augmentation sampling)
        """
        super(RandomMagnitudeRandaugment, self).__init__(num_augmentations=num_augmentations, magnitude=magnitude,
                                                         augmentation_list=augmentation_list,
                                                         min_magnitude=min_magnitude, max_magnitude=max_magnitude)

        assert sample_min_magnitude >= 0.0
        assert sample_max_magnitude <= 1.0

        self.sample_min_magnitude = sample_min_magnitude
        self.sample_max_magnitude = sample_max_magnitude

    def magnitude(self) -> float:
        """
        Randomize magnitude when scaling augmentation
        Returns: random magnitude between min/max magnitude
        """
        return round(random.uniform(self.sample_min_magnitude, self.sample_max_magnitude), 3)


class RandomMagnitudeSampledRandaugment(RandomMagnitudeRandaugment):

    def choose_augs_by_magnitude(self):
        """
        Returns: list of augmentations from augmentation list
        """

        sample = set()
        population = self.augment_list.copy()
        weights = self.get_augmentation_sample_weights_for_magnitude()
        while len(sample) < self.num_augmentations:
            choice = random.choices(population, weights=weights, k=1)[0]  # single choice
            sample.add(choice)
            index = population.index(choice)
            weights.pop(index)
            population.remove(choice)
            weights = [x / sum(weights) for x in weights]
        return list(sample)


class FixedAugment(RandAugment):
    """
    Randaugment variant where augmentations are always fixed as specified in augmentation list
    """

    def choose_augs_by_magnitude(self):
        return self.augment_list


class FixedRandomMagnitudeAugment(RandomMagnitudeRandaugment, FixedAugment):
    """
    Combines FixedAugement and RandomMagnitudeRandaugment.
    Resulting in a RandAugment variant, where the list of augmentations is fixed, but the magnitude is randomized.
    """
    pass
