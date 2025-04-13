import numpy as np
from models.snn_module import SpikingNeuronLayer
from models.audio_visual_encoder import AudioVisualEncoder
from models.decoder import SpikeDecoder
from dataset.preprocess import preprocess

if __name__ == "__main__":
    video_path = "dataset/sample_video.mp4"
    audio_input, visual_input = preprocess(video_path)

    encoder = AudioVisualEncoder()
    encoded = encoder.encode(audio_input[:1], visual_input[:1])

    decoder = SpikeDecoder(128, 1)
    enhanced_audio = decoder.decode(encoded)

    print("Enhanced audio shape:", enhanced_audio.shape)
