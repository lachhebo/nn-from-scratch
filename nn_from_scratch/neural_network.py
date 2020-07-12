from typing import List
from nn_from_scratch.layer import Layer
from nn_from_scratch.loss import LossFunction, LeastSquaredError


class NeuralNetwork(object):
    def __init__(self, layers: List[Layer],
                 loss: LossFunction = LeastSquaredError):
        self.layers = layers
        self.loss = loss

    def forward(self, input_vector: List[float]):
        for layer in self.layers:
            input_vector = layer.forward(input_vector)
        return input_vector

    def backpropagation(self, ):