from random import shuffle
from typing import List


class BatchGenerator(object):
    def __init__(self, dataset: List[List], batch_size: int):
        self.__batch_size = batch_size
        self.__dataset = dataset
        self.__pile_dataset_index: List = []

    @property
    def batch_size(self):
        return self.__batch_size

    @property
    def dataset(self):
        return self.__dataset

    def new_epoch(self):
        self.__pile_dataset_index = [x for x in range(len(self.__dataset))]
        shuffle(self.__pile_dataset_index)

    def is_next_batch(self):
        return len(self.__pile_dataset_index) >= self.__batch_size

    def next_batch(self):
        if self.is_next_batch():
            for i in range(self.batch_size):
                yield self.__dataset[self.__pile_dataset_index[0]]
                self.__pile_dataset_index.pop(0)
