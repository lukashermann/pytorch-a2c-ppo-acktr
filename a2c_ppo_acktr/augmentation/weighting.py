from abc import ABC


class Weighter(ABC):
    """
    Base class for calculating consistency loss weight
    """

    def __call__(self, step, loss, **kwargs):
        raise NotImplementedError("Override this class to use weighting method")


class FixedWeight(Weighter):
    def __init__(self, weight):
        self._fixed_weight = weight

    def __call__(self, step, loss, **kwargs):
        """

        Args:
            step:
            loss:
            **kwargs:

        Returns:
            weight for consistency loss

        >>> fixed_weight = FixedWeight(0.5)
        >>> fixed_weight(step=0, loss=0)
        0.5
        >>> fixed_weight(step=1000, loss=0.1)
        0.5
        """
        return self._fixed_weight


class FunctionWeight(FixedWeight):
    def __init__(self, function, use_absolute_value=True, with_fixed_weight=1.0):
        """
        Calculates weight according to weighting function which will be passed the current step in the training progress

        Args:
            function: function which will return weight given the current training step and non-consistency loss
            use_absolute_value: if set to True, the absolute value of the function is used
            with_fixed_weight: optional fixed weight which will be multiplied to function value.
        """
        super().__init__(weight=with_fixed_weight)
        self._weighting_function = function
        self._absolute = use_absolute_value

    def __call__(self, step, loss, **kwargs):
        """

        Args:
            step:
            loss:
            **kwargs:

        Returns:

        >>> import numpy as np
        >>> function = np.poly1d([2, 1])  # f(x) = 2x + 1
        >>> weight_from_function = FunctionWeight(function)
        >>> weight_from_function(step=1, loss=-1)
        3.0
        >>> weight_from_function = FunctionWeight(function, with_fixed_weight=0.10)
        >>> weight_from_function(step=10, loss=-1)
        2.1
        """
        if self._absolute:
            return abs(self._weighting_function(step)) * self._fixed_weight
        return self._weighting_function(step) * self._fixed_weight


class MovingAverageFromLossWeight(FunctionWeight):
    def __init__(self, with_fixed_weight=None, with_loss_weight_function=None):
        super().__init__(function=with_loss_weight_function, with_fixed_weight=with_fixed_weight)
        self.avg_loss = 0.0

    def __call__(self, step, loss, **kwargs):
        """

        Args:
            step:
            loss:
            **kwargs:

        Returns:

        >>> avg_weight = MovingAverageFromLossWeight()
        >>> avg_weight = MovingAverageFromLossWeight(with_fixed_weight=0.1)
        >>> avg_weight(step=1000, loss=10)
        0.4
        >>> import numpy as np
        >>> function = np.poly1d([2, 1])  # f(x) = 2x + 1
        >>> avg_weight = MovingAverageFromLossWeight(with_loss_weight_function=function)
        >>> avg_weight = MovingAverageFromLossWeight(with_fixed_weight=0.1, with_loss_weight_function=function)
        """
        return self.avg_loss