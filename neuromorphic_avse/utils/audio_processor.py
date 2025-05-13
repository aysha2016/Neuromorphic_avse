import torch
import torchaudio
import librosa
import numpy as np
from typing import Tuple, Optional
from scipy import signal

class AudioProcessor:
    """Handles audio processing and feature extraction for the neuromorphic system."""
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 40,
        n_fft: int = 1024,
        hop_length: int = 512,
        win_length: int = 1024
    ):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        
        # Initialize MFCC transform
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': n_fft,
                'hop_length': hop_length,
                'win_length': win_length
            }
        )
        
    def load_audio(self, file_path: str) -> Tuple[torch.Tensor, int]:
        """Load audio file and convert to tensor."""
        waveform, sr = torchaudio.load(file_path)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        return waveform, self.sample_rate
    
    def extract_mfcc(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract MFCC features from waveform."""
        # Ensure waveform is mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Extract MFCCs
        mfcc = self.mfcc_transform(waveform)
        return mfcc.squeeze(0)  # Remove channel dimension
    
    def preprocess_audio(
        self,
        waveform: torch.Tensor,
        normalize: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess audio for the neuromorphic system."""
        # Extract features
        mfcc = self.extract_mfcc(waveform)
        
        # Normalize if requested
        if normalize:
            mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
            waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-8)
            
        return waveform, mfcc
    
    def apply_noise_reduction(
        self,
        waveform: torch.Tensor,
        noise_threshold: float = 0.1
    ) -> torch.Tensor:
        """Apply basic noise reduction using spectral gating."""
        # Convert to numpy for processing
        audio_np = waveform.numpy()
        
        # Compute spectrogram
        freqs, times, Sxx = signal.spectrogram(
            audio_np,
            fs=self.sample_rate,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length
        )
        
        # Estimate noise floor
        noise_floor = np.mean(Sxx, axis=1, keepdims=True)
        
        # Apply spectral gating
        mask = Sxx > (noise_floor * (1 + noise_threshold))
        Sxx_clean = Sxx * mask
        
        # Reconstruct signal
        _, audio_clean = signal.istft(
            Sxx_clean,
            fs=self.sample_rate,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length
        )
        
        return torch.from_numpy(audio_clean).float()
    
    def segment_audio(
        self,
        waveform: torch.Tensor,
        segment_length: int,
        overlap: float = 0.5
    ) -> torch.Tensor:
        """Segment audio into overlapping chunks for processing."""
        # Calculate step size
        step = int(segment_length * (1 - overlap))
        
        # Create segments
        segments = []
        for start in range(0, waveform.shape[-1] - segment_length, step):
            segment = waveform[..., start:start + segment_length]
            segments.append(segment)
            
        return torch.stack(segments)
    
    def reconstruct_audio(
        self,
        segments: torch.Tensor,
        original_length: int,
        overlap: float = 0.5
    ) -> torch.Tensor:
        """Reconstruct audio from processed segments using overlap-add."""
        segment_length = segments.shape[-1]
        step = int(segment_length * (1 - overlap))
        
        # Initialize output
        output = torch.zeros(original_length)
        weights = torch.zeros(original_length)
        
        # Overlap-add
        for i, segment in enumerate(segments):
            start = i * step
            end = start + segment_length
            
            # Apply window
            window = torch.hann_window(segment_length)
            segment = segment * window
            
            # Add to output
            output[start:end] += segment
            weights[start:end] += window
            
        # Normalize by weights
        output = output / (weights + 1e-8)
        
        return output 