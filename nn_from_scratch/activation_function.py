from abc import ABC, abstractmethod
from math import exp as e


class ActivationFunction(ABC):
    def __init(self):
        pass

    @classmethod
    @abstractmethod
    def compute_function(cls, x: float) -> float:
        pass

    @classmethod
    @abstractmethod
    def compute_derivative(cls, x: float) -> float:
        pass


class SigmoidFunction(ActivationFunction):
    def __init__(self):
        pass

    @classmethod
    def compute_function(cls, x: float) -> float:
        return 1 / (1 + e(-x))

    @classmethod
    def compute_derivative(cls, x: float) -> float:
        return cls.compute_function(x) * (1 - cls.compute_function(x))


class ConstantFunction(ActivationFunction):
    def __init__(self):
        pass

    @classmethod
    def compute_function(cls, x: float) -> float:
        return x

    @classmethod
    def compute_derivative(cls, x: float) -> float:
        return 1
