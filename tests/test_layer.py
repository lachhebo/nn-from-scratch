import pytest

from nn_from_scratch.activation_function import SigmoidFunction
from nn_from_scratch.layer import Layer
from nn_from_scratch.exception import InputSizeError


def test_layer_should_raise_error_if_input_size_if_different_than_size_layer():
    # given
    layer = Layer(3)()
    input_data = [9, 8]

    # when
    with pytest.raises(InputSizeError):
        # then
        layer.forward(input_data)


def test_layer_output_in_forward_pass_must_be_mathematically_correct_taking_weights_and_bias():
    # given
    layer = Layer(3)
    layer.weights = [1, 5, 1]
    layer.bias = 5

    input_vector = [2, 3, 2]
    expected_output = [7, 20, 7]

    # when
    actual_output = layer.forward(input_vector)

    # then
    assert expected_output == actual_output


def test_layer_output_in_forward_pass_must_be_mathematically_correct_taking_weights_bias_and_sigmoid_activation():
    # given
    layer = Layer(3, activation_function=SigmoidFunction)
    layer.weights = [1, 5, 1]
    layer.bias = 5
    input_vector = [2, 3, 2]
    expected_output = [0.9990889488055994,
                       0.9999999979388463,
                       0.9990889488055994]

    # when
    actual_output = layer.forward(input_vector)

    # then
    expected_output = [round(x, 4) for x in expected_output]
    actual_output = [round(x, 4) for x in actual_output]
    assert expected_output == actual_output
