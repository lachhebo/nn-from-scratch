import pytest

from nn_from_scratch.activation_function import SigmoidFunction, \
    ConstantFunction
from nn_from_scratch.layer import Layer
from nn_from_scratch.neural_network import NeuralNetwork


def test_given_an_input_vector_and_a_two_layers_nn_we_get_mathematically_coherent_result():
    # given
    layer1 = Layer([1, 5, 1], 1)
    layer2 = Layer([2, 1, 2], 0)
    neural_network = NeuralNetwork([layer1, layer2])
    input_vector = [2, 3, 2]
    expected_output = [6, 16, 6]

    # when
    actual_output = neural_network.forward(input_vector)

    # then
    assert expected_output == actual_output


def test_given_an_input_vector_and_a_two_sigmoid_layers_nn_we_get_mathematically_coherent_result():
    # given
    layer1 = Layer([1, 5, 1], 1, activation_function=ConstantFunction)
    layer2 = Layer([2, 1, 2], 0, activation_function=SigmoidFunction)
    neural_network = NeuralNetwork([layer1, layer2])
    input_vector = [2, 3, 2]
    expected_output = [0.998, 1.0, 0.998]

    # when
    actual_output = neural_network.forward(input_vector)

    # then
    expected_output = [round(x, 3) for x in expected_output]
    actual_output = [round(x, 3) for x in actual_output]
    assert expected_output == actual_output
