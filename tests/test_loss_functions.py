from nn_from_scratch.loss import LeastSquaredError


def test_lse_given_nn_output_forward_and_ground_truth_we_get_correct_value():
    # given
    predicted_output = [1.2, 3]
    ground_truth_output = [1.5, 5]
    lse = LeastSquaredError()
    expected_loss = 4.09
    # when
    output_loss = lse.compute_loss(predicted_output, ground_truth_output)
    # then
    assert expected_loss == output_loss
