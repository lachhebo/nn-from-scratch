# on peut fournir au réseaux de neurone un ensemble de batch pris aléatoirement avec "sampling".


def test_generator_batch_return_length_of_dataframe_on_one_epoch():
    # given
    dataset = [[4,3], [3,3], [1,1], [0,1], [0,0]]
    batch_size = 2
    batchgenerator = BatchGenerator(batch_size,dataset)
    expected_iteration_number = len(dataset) % batch_size 
    expected_len = expected_iteration_number * batch_size

    # when 
    output_len = 0
    iteration_number  = 0
    while batchgenerator.is_next_batch() :
        ouput_len += len(batchgenerator.next_batch())
        iteration_number += 1


    # then
    assert ouput_len == expected_len
    assert iteration_number == expected_iteration_number