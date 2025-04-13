import numpy as np

class SpikingNeuronLayer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.rand(input_size, output_size)
        self.threshold = 0.5

    def forward(self, input_signal):
        potentials = np.dot(input_signal, self.weights)
        spikes = (potentials > self.threshold).astype(float)
        return spikes
