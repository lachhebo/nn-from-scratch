import pytest

from nn_from_scratch.activation_function import SigmoidFunction
from nn_from_scratch.layer import Layer
from nn_from_scratch.exception import InputSizeError


def test_size_input_is_equal_to_size_of_weights_for_each_neuron():
    # given
    input_size = 2
    layer = Layer(3, input_size=2).generate_weight_matrix()

    # then
    for neuron_weights in layer.weights:
        assert len(neuron_weights) == input_size


def test_layer_output_in_forward_pass_must_be_mathematically_correct_taking_weights_and_bias():
    # given
    layer = Layer(3, input_size=2)
    layer.weights = [[1, 1], [1, 2], [0, 0]]
    layer.bias = [1, 0, 0]
    input_vector = [2, 3]

    expected_output = [6, 8, 0]

    # when
    actual_output = layer.forward(input_vector)

    # then
    assert expected_output == actual_output


def test_layer_output_in_forward_pass_must_be_mathematically_correct_taking_weights_bias_and_sigmoid_activation():
    # given
    layer = Layer(3, input_size=2, activation_function=SigmoidFunction)
    layer.weights = [[1, 1], [1, 2], [0, 0]]
    layer.bias = [1, 0, 0]
    input_vector = [2, 3]

    expected_output = [0.9975, 0.9997, 0.5]

    # when
    actual_output = layer.forward(input_vector)

    # then
    expected_output = [round(x, 4) for x in expected_output]
    actual_output = [round(x, 4) for x in actual_output]
    assert expected_output == actual_output
