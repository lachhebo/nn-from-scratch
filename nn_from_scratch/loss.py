from abc import ABC, abstractmethod
from math import exp as e
from typing import List


class LossFunction(ABC):
    def __init(self):
        pass

    @classmethod
    @abstractmethod
    def compute_loss(cls, batch_prediction: List[float],
                     batch_ground_truth: List[float]) -> float:
        pass


class LeastSquaredError(LossFunction):
    def __init__(self):
        pass

    @classmethod
    def compute_loss(cls, batch_predictions: List[float],
                     batch_ground_truths: List[float]) -> float:
        loss = 0
        for value_prediction, value_ground_truth in zip(batch_predictions,
                                                        batch_ground_truths):
            loss += (value_prediction - value_ground_truth) ** 2
        return loss
