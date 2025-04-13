import numpy as np

class SpikeDecoder:
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size)

    def decode(self, spike_tensor):
        return np.dot(spike_tensor, self.weights)
