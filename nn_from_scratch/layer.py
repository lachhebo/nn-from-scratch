from nn_from_scratch.exception import InputSizeError
from typing import List
from nn_from_scratch.activation_function import ActivationFunction, \
    ConstantFunction
from random import uniform as rand


class Layer:
    def __init__(self,
                 size_weights: int,
                 activation_function: ActivationFunction = ConstantFunction):
        self.__size = size_weights
        self.__weights = None
        self.__bias = None
        self.__activation_function = activation_function

    @property
    def weights(self):
        return self.__weights

    @property
    def bias(self):
        return self.__bias

    @property
    def activation_function(self):
        return self.__activation_function

    @weights.setter
    def weights(self, weights):
        self.__weights = weights

    @bias.setter
    def bias(self, bias):
        self.__bias = bias

    def __call__(self, *args, **kwargs):
        self.__weights = [rand(-1, 1) for i in range(self.__size)]
        self.__bias = rand(-1, 1)
        return self

    def forward(self, input_vector: List[float]) -> List:
        if len(input_vector) != len(self.weights):
            raise InputSizeError(
                'The input size must be the same than the weight')
        output_res = []
        for weight, input_value in zip(self.weights, input_vector):
            calculation_neuron = weight * input_value + self.bias
            output_res.append(
                self.activation_function.compute_function(
                    calculation_neuron)
            )
        return output_res
