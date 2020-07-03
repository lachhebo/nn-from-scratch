from typing import List
from nn_from_scratch.layer import Layer


class NeuralNetwork(object):
    def __init__(self, layers: List[Layer]):
        self.layers = layers

    def forward(self, input_vector: List[float]):
        for layer in self.layers:
            input_vector = layer.forward(input_vector)
        return input_vector

