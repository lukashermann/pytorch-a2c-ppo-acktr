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
        Returns a fixed weight.

        Args:
            step: not used
            loss: not used
            **kwargs: not used

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
    def __init__(self, function, use_absolute_value=True, with_fixed_weight=1.0, apply_to=None):
        """
        Calculates weight according to weighting function which will be passed the current step in the training progress

        Args:
            function: function which will return weight given the current training step and non-consistency loss.
            use_absolute_value: if set to True, the absolute value of the function is used.
            with_fixed_weight: optional fixed weight which will be multiplied to function value.
            apply_to: list of strings specifying on which part of the arguments to apply the function when calling this
                class. WARNING: underlying function has to support the same number of arguments as there are elements in
                this list! There is no validation before calling the function.
                default: ['step'] -> only pass step argument to function when calling this class.
                Supported values: ['step'], ['loss'], ['step', 'loss'].
        """
        super().__init__(weight=with_fixed_weight if with_fixed_weight is not None else 1.0)
        if apply_to is None:
            apply_to = ['step']
        self._weighting_function = function
        self._absolute = use_absolute_value
        self._apply_to = apply_to

    def __call__(self, step, loss, **kwargs):
        """
        Calculate weight according to specifed function

        Args:
            step: current step in training process
            loss: current loss value
            **kwargs: not used

        Returns:
            weight for consistency loss

        >>> import numpy as np
        >>> function = np.poly1d([2, 1])  # f(x) = 2x + 1
        >>> weight_from_function = FunctionWeight(function)
        >>> weight_from_function(step=1, loss=-1)
        3.0
        >>> weight_from_function = FunctionWeight(function, with_fixed_weight=0.10)
        >>> weight_from_function(step=10, loss=-1)
        2.1
        """
        args = []
        if "step" in self._apply_to:
            args.append(step)
        if "loss" in self._apply_to:
            args.append(loss)

        if self._absolute:
            return abs(self._weighting_function(*args)) * self._fixed_weight
        return self._weighting_function(*args) * self._fixed_weight


class MovingAverageFromLossWeight(FunctionWeight):
    def __init__(self, num_values=10, with_fixed_weight=None, with_loss_weight_function=None, **kwargs):
        """
        Calculates the moving average of the loss argument.

        Args:
            num_values: number of values to use for calculating moving average
            with_fixed_weight: optional fixed weight to be multiplied with moving average
            with_loss_weight_function: optional function to be multiplied with moving average
            kwargs: additional arguments to be passed to parent classes
        """
        super().__init__(function=with_loss_weight_function, with_fixed_weight=with_fixed_weight, **kwargs)
        self.avg = None
        self._num_values = num_values

    def __call__(self, step, loss, **kwargs):
        """
        Calculates the moving average of the loss argument.
        Optionally uses weight function and fixed weight to calculate the final weight.

        Implementation details:
            computatation of online (exponential) moving average: https://stackoverflow.com/a/37830174

        Args:
            step: current step in training, only used if loss weight function is set
            loss: loss to be used for calculating moving average
            **kwargs: Unused additional arguments

        Returns:
            weight for loss

        # >>> avg_weight = MovingAverageFromLossWeight(num_values=3)
        # >>> avg_weight(step=1000, loss=10)
        # 10.0
        # >>> avg_weight(step=1000, loss=10)
        # 10.0
        # >>> avg_weight(step=1000, loss=5)
        # 8.333333333333332
        # >>> avg_weight(step=1000, loss=10)
        # 8.88888888888889
        # >>> avg_weight(step=1000, loss=10)
        # 9.25925925925926
        # >>> avg_weight = MovingAverageFromLossWeight(num_values=3, with_fixed_weight=0.1)
        # >>> avg_weight(step=1000, loss=10)
        # 1.0
        # >>> avg_weight(step=1000, loss=5)
        # 0.8333333333333333
        >>> import numpy as np
        >>> function = np.poly1d([2])  # f(x) = 2x + 0
        >>> avg_weight = MovingAverageFromLossWeight(num_values=3, with_loss_weight_function=function)
        >>> avg_weight(step=1000, loss=10)
        20.0
        >>> avg_weight(step=1000, loss=5)
        16.666666666666664
        >>> avg_weight(step=1000, loss=10)
        17.77777777777778
        >>> avg_weight = MovingAverageFromLossWeight(num_values=3, with_fixed_weight=0.1, with_loss_weight_function=function)
        >>> avg_weight(step=1000, loss=10)
        2.0
        >>> avg_weight(step=1000, loss=5)
        1.6666666666666665
        >>> avg_weight(step=1000, loss=10)
        1.777777777777778
        """
        # For numerical stability: initialize with first loss value instead of 0
        if self.avg is None:
            self.avg = loss

        self.avg -= self.avg / self._num_values
        self.avg += loss / self._num_values

        if self._weighting_function is not None:
            return self.avg * super().__call__(step, loss)
        if self._fixed_weight is not None:
            return self.avg * self._fixed_weight

        return self.avg
