from nn_from_scratch.batch_generator import BatchGenerator


def test_generator_batch_return_length_of_dataframe_on_one_epoch():
    # given
    dataset = [[4, 3], [3, 3], [1, 1], [0, 1], [0, 0]]
    batch_size = 2
    batchgenerator = BatchGenerator(dataset, batch_size)
    batchgenerator.new_epoch()
    expected_iteration_number = len(dataset) // batch_size
    expected_len = expected_iteration_number * batch_size

    # when
    output_len = 0

    while batchgenerator.is_next_batch():
        batch = list(batchgenerator.next_batch())
        output_len += len(batch)

    # then
    assert output_len == expected_len
