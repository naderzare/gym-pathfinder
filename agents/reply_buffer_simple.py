import random
import copy
from agents.transit import Transition

class Buffer:
    def __init__(self, size, min_size=600):
        self.buffer_number = 1
        self.use_weight = False
        self.size = [size for _ in range(self.buffer_number)]
        self.buffer = [[None for _ in range(size)] for _ in range(self.buffer_number)]
        self.i = [0 for _ in range(self.buffer_number)]
        self.min_size = min_size
        self.buffers_weight = [0 for _ in range(self.buffer_number)]

    def add_to(self, item, buffer_number):
        self.buffer[buffer_number][self.i[buffer_number]] = item
        self.i[buffer_number] += 1
        self.i[buffer_number] = self.i[buffer_number] % self.size[buffer_number]

    def add(self, item: Transition):
        buffer_number = 0
        self.buffers_weight[buffer_number] += 1
        self.add_to(item, buffer_number)

    def get_rand(self, request_number):
        # weight = [self.buffers_weight[i] / sum(self.buffers_weight) for i in range(self.buffer_number)]
        res = []
        for buffer in range(self.buffer_number):
            number = int(copy.copy(request_number) / self.buffer_number)
            size = self.size[buffer]
            if self.buffer[buffer][self.size[buffer] - 1] is None:
                size = self.i[buffer] - 1
            if size <= self.min_size / self.buffer_number:
                continue
            number = min(number, size)
            ran = [random.randint(0, size - 1) for _ in range(number)]
            for r in ran:
                # if self.use_weight:
                #     self.buffer[buffer][r].value = weight[buffer]
                # else:
                self.buffer[buffer][r].value = 1.0
                res.append(self.buffer[buffer][r])
        return res