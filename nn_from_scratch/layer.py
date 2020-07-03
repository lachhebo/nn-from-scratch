from nn_from_scratch.exception import InputSizeError
from typing import List
from nn_from_scratch.activation_function import ActivationFunction, ConstantFunction


class Layer:
    def __init__(self,
                 weights,
                 bias,
                 activation_function: ActivationFunction = ConstantFunction):
        self.weights = weights
        self.bias = bias
        self.activation_function = activation_function


    def forward(self, input_vector: List[float]) -> List:
        if len(input_vector) != len(self.weights):
            raise InputSizeError(
                'The input size must be the same than the weight')
        output_res = []
        for i in range(len(input_vector)):
            calculation_neuron = self.weights[i] * input_vector[i] + self.bias
            output_res.append(
                self.activation_function.compute_function(calculation_neuron)
            )
        return output_res
