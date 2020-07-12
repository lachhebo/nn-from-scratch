from nn_from_scratch.exception import InputSizeError
from typing import List
from nn_from_scratch.activation_function import ActivationFunction, \
    ConstantFunction
from random import uniform as rand


class Layer:
    def __init__(self,
                 number_neuron: int,
                 input_size: int,
                 activation_function: ActivationFunction = ConstantFunction):
        self.__number_neuron = number_neuron
        self.__input_size = input_size
        self.__weights = None
        self.__bias = None
        self.__activation_function = activation_function
        self.__neuron_activations = None

    @property
    def weights(self)-> List[List[float]]:
        return self.__weights

    @property
    def bias(self) -> List[float]:
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

    def generate_weight_matrix(self):
        self.__weights = [[rand(-1, 1) for i in range(self.__input_size)] for j in range(self.__number_neuron)]
        self.__bias = [rand(-1, 1) for _ in range(self.__number_neuron)]
        return self

    def forward(self, input_vector: List[float]) -> List:
        if len(input_vector) != self.__input_size:
            raise InputSizeError
        output_res = []
        for i in range(self.__number_neuron):
            calculation_neuron = 0
            for j in range(self.__input_size):
                calculation_neuron += input_vector[j]*self.__weights[i][j]
            output_value = self.activation_function.compute_function(calculation_neuron + self.bias[i])
            output_res.append(output_value)
        return output_res
