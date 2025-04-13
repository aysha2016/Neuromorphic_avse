import numpy as np
from models.snn_module import SpikingNeuronLayer

class AudioVisualEncoder:
    def __init__(self):
        self.audio_layer = SpikingNeuronLayer(128, 64)
        self.visual_layer = SpikingNeuronLayer(128, 64)

    def encode(self, audio_input, visual_input):
        audio_encoded = self.audio_layer.forward(audio_input)
        visual_encoded = self.visual_layer.forward(visual_input)
        return np.concatenate((audio_encoded, visual_encoded), axis=1)
