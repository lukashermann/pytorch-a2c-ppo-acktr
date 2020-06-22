"""
This file contains an adapted implementation of RandAugment
Adapted from:

Our implementation allows dynamically changing the randomization, allowing this to be used in our evaluation.
"""
import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
from PIL import Image
from typing import List, Union


class Augmentation:
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


class AutoContrast(Augmentation):
    def __call__(self, img, _):
        return PIL.ImageOps.autocontrast(img)


class Invert(Augmentation):
    def __call__(self, img, _):
        return PIL.ImageOps.invert(img)


class Equalize(Augmentation):
    def __call__(self, img, _):
        return PIL.ImageOps.equalize(img)


class Flip(Augmentation):
    def __call__(self, img, _):
        return PIL.ImageOps.mirror(img)


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


class Cutout(RangedAugmentation):
    def __init__(self, min_value=0.0, max_value=0.2):
        super().__init__(min_value, max_value)

    def cutoutAbs(self, img, v):  # [0, 60] => percentage: [0, 0.2]
        # assert 0 <= v <= 20
        if v < 0:
            return img
        w, h = img.size
        x0 = np.random.uniform(w)
        y0 = np.random.uniform(h)

        x0 = int(max(0, x0 - v / 2.))
        y0 = int(max(0, y0 - v / 2.))
        x1 = min(w, x0 + v)
        y1 = min(h, y0 + v)

        xy = (x0, y0, x1, y1)
        color = (125, 123, 114)
        # color = (0, 0, 0)
        img = img.copy()
        PIL.ImageDraw.Draw(img).rectangle(xy, color)
        return img

    def __call__(self, img, magnitude):
        super().__call__(img, magnitude)
        if magnitude <= 0.0: return img

        magnitude = magnitude * img.size[0]
        return self.cutout_abs(img, magnitude)


class Identity(Augmentation):
    def __call__(self, img, _):
        return img


# List taken from RandAugment Paper
AUGMENTATION_LIST_DEFAULT = [Identity(), AutoContrast(), Equalize(),
                             Rotate(), Solarize(), Color(),
                             Posterize(), Contrast(), Brightness(),
                             Sharpness(), ShearX(), ShearY(),
                             TranslateX(), TranslateY()]

AUGMENTATION_LIST_SMALL_RANGE = [Identity(), AutoContrast(), Equalize(),
                                 Rotate(min_value=-10, max_value=10), Solarize(min_value=128, max_value=256), Color(),
                                 Posterize(min_value=6, max_value=8), Contrast(min_value=0.8, max_value=1.2),
                                 Brightness(min_value=0.8, max_value=1.2),
                                 Sharpness(), ShearX(min_value=-0.1, max_value=0.1), ShearY(min_value=-0.1, max_value=0.1),
                                 TranslateX(min_value=-0.1, max_value=0.1), TranslateY(min_value=-0.1, max_value=0.1)]


class RandAugment:
    """
    Randomly choose specified number of augmentations and apply them sequentially with specified magnitude.
    """

    def __init__(self, num_augmentations: int = 1, magnitude: float = 0.0,
                 augmentation_list: List = AUGMENTATION_LIST_DEFAULT,
                 min_magnitude: Union[float, int] = 0.0, max_magnitude: Union[float, int] = 1.0):
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
        self.__magnitude = None
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

        normalized_magnitude = (magnitude - self.__min_magnitude) / (self.__max_magnitude - self.__min_magnitude)
        self.__magnitude = normalized_magnitude

    def __call__(self, img: PIL.Image) -> PIL.Image:
        # TODO: Implement weighted random choice
        # https://stackoverflow.com/questions/3679694/a-weighted-version-of-random-choice
        augs = random.choices(self.augment_list, k=self.num_augmentations)
        for augmentation in augs:
            aug_magnitude = augmentation.scale_magnitude_to_aug_range(magnitude=self.magnitude())
            img = augmentation(img, aug_magnitude)
        return img

    def magnitude(self) -> float:
        return self.__magnitude
