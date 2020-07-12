import pytest

from nn_from_scratch.activation_function import SigmoidFunction, \
    ConstantFunction
from nn_from_scratch.layer import Layer
from nn_from_scratch.loss import LeastSquaredError
from nn_from_scratch.neural_network import NeuralNetwork


def test_given_an_input_vector_and_a_two_layers_nn_we_get_mathematically_coherent_result():
    # given
    input_vector = [1, 2]

    layer1 = Layer(3, 2)
    layer1.weights = [[1, 2], [0, 1], [1, 0]]
    layer1.bias = [1, 2, 0]

    layer2 = Layer(3, 3)
    layer2.weights = [[0, 1, 2], [1, 2, 1], [1, 0, 2]]
    layer2.bias = [0, 2, 1]

    neural_network = NeuralNetwork([layer1, layer2])
    expected_output = [6, 17, 9]

    # when
    actual_output = neural_network.forward(input_vector)

    # then
    assert expected_output == actual_output


def test_given_an_input_vector_and_a_two_sigmoid_layers_nn_we_get_mathematically_coherent_result():
    # given
    input_vector = [1, 2]

    layer1 = Layer(3, 2, SigmoidFunction)
    layer1.weights = [[1, 2], [0, 1], [1, 0]]
    layer1.bias = [1, 2, 0]

    layer2 = Layer(3, 3, SigmoidFunction)
    layer2.weights = [[0, 1, 2], [1, 2, 1], [1, 0, 2]]
    layer2.bias = [0, 2, 1]

    neural_network = NeuralNetwork([layer1, layer2])
    expected_output = [0.92, 0.997, 0.97]

    # when
    actual_output = neural_network.forward(input_vector)

    # then
    expected_output = [round(x, 3) for x in expected_output]
    actual_output = [round(x, 3) for x in actual_output]
    assert expected_output == actual_output


@pytest.mark.skip
def test_for_a_given_batch_on_a_nn_we_can_get_the_loss():
    # given
    layer1 = Layer(3, activation_function=ConstantFunction)
    layer1.weights = [1, 5, 1]
    layer1.bias = 1

    layer2 = Layer(3, activation_function=SigmoidFunction)
    layer2.weights = [2, 1, 2]
    layer2.bias = 0

    neural_network = NeuralNetwork([layer1, layer2], loss=LeastSquaredError)
    ground_truth = [1, 2, 2]
    expected_output = 2
    batch_input = [2, 3, 2]

    # when
    actual_output = neural_network.compute_loss(batch_input, ground_truth)

    # then
    expected_output = [round(x, 3) for x in expected_output]
    actual_output = [round(x, 3) for x in actual_output]
    assert expected_output == actual_output
