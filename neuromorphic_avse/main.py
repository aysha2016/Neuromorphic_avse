import torch
import torch.nn as nn
import cv2
import argparse
from pathlib import Path
from typing import Tuple, Optional
from tqdm import tqdm

from models.encoder import NeuromorphicEncoder
from models.decoder import NeuromorphicDecoder, STDPUpdater
from utils.audio_processor import AudioProcessor
from utils.visual_processor import VisualProcessor

class AVSEPipeline:
    """Main pipeline for Audio-Visual Speech Enhancement."""
    def __init__(
        self,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        audio_config: dict = None,
        visual_config: dict = None,
        model_config: dict = None
    ):
        self.device = device
        
        # Default configurations
        self.audio_config = audio_config or {
            'sample_rate': 16000,
            'n_mfcc': 40,
            'n_fft': 1024,
            'hop_length': 512
        }
        
        self.visual_config = visual_config or {
            'target_size': (96, 96),
            'face_detector': 'sfd'
        }
        
        self.model_config = model_config or {
            'audio_input_size': 40,  # MFCC features
            'visual_input_size': 96 * 96,  # Lip region size
            'hidden_size': 256,
            'fusion_size': 128,
            'num_steps': 10
        }
        
        # Initialize processors
        self.audio_processor = AudioProcessor(**self.audio_config)
        self.visual_processor = VisualProcessor(**self.visual_config)
        
        # Initialize models
        self.encoder = NeuromorphicEncoder(
            audio_input_size=self.model_config['audio_input_size'],
            visual_input_size=self.model_config['visual_input_size'],
            hidden_size=self.model_config['hidden_size'],
            fusion_size=self.model_config['fusion_size'],
            num_steps=self.model_config['num_steps']
        ).to(device)
        
        self.decoder = NeuromorphicDecoder(
            input_size=self.model_config['fusion_size'],
            hidden_size=self.model_config['hidden_size'],
            output_size=self.audio_config['n_mfcc'],
            num_steps=self.model_config['num_steps']
        ).to(device)
        
        # Initialize STDP updater for online learning
        self.stdp_updater = STDPUpdater(learning_rate=0.01)
        
    def process_video(
        self,
        video_path: str,
        output_path: str,
        visualize: bool = False
    ) -> Tuple[str, Optional[str]]:
        """Process video file and enhance speech."""
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize video writer for visualization
        vis_writer = None
        if visualize:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            vis_path = str(Path(output_path).with_suffix('.mp4'))
            vis_writer = cv2.VideoWriter(
                vis_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (width, height)
            )
            
        # Process frames
        audio_segments = []
        visual_segments = []
        lip_regions = []
        
        print("Processing video frames...")
        with tqdm(total=frame_count) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process visual input
                lip_tensor, lip_region = self.visual_processor.preprocess_frame(frame)
                visual_segments.append(lip_tensor)
                if lip_region is not None:
                    lip_regions.append(lip_region)
                    
                # Visualize if requested
                if visualize and lip_region is not None:
                    vis_frame = self.visual_processor.visualize_lip_region(
                        frame,
                        lip_region,
                        show_landmarks=True
                    )
                    vis_writer.write(vis_frame)
                    
                pbar.update(1)
                
        cap.release()
        if vis_writer is not None:
            vis_writer.release()
            
        # Extract audio
        print("Processing audio...")
        waveform, _ = self.audio_processor.load_audio(video_path)
        waveform, mfcc = self.audio_processor.preprocess_audio(waveform)
        
        # Segment audio
        audio_segments = self.audio_processor.segment_audio(
            waveform,
            segment_length=self.audio_config['hop_length'] * 10,
            overlap=0.5
        )
        
        # Process segments
        print("Enhancing speech...")
        enhanced_segments = []
        
        for i in tqdm(range(len(audio_segments))):
            # Get corresponding visual features
            if i < len(visual_segments):
                visual_features = visual_segments[i].to(self.device)
            else:
                visual_features = torch.zeros(
                    self.visual_config['target_size']
                ).to(self.device)
                
            # Process audio segment
            audio_features = audio_segments[i].to(self.device)
            
            # Forward pass
            with torch.no_grad():
                audio_spikes, visual_spikes, fused_spikes = self.encoder(
                    audio_features.unsqueeze(0),
                    visual_features.unsqueeze(0)
                )
                
                enhanced_waveform, _ = self.decoder(
                    fused_spikes,
                    audio_features.unsqueeze(0)
                )
                
            # Apply STDP learning
            if i > 0:
                self.encoder = self.stdp_updater.update_weights(
                    self.encoder,
                    audio_spikes.squeeze(),
                    visual_spikes.squeeze()
                )
                
            enhanced_segments.append(enhanced_waveform.squeeze().cpu())
            
        # Reconstruct enhanced audio
        enhanced_audio = self.audio_processor.reconstruct_audio(
            torch.stack(enhanced_segments),
            waveform.shape[-1],
            overlap=0.5
        )
        
        # Save enhanced audio
        torchaudio.save(
            output_path,
            enhanced_audio.unsqueeze(0),
            self.audio_config['sample_rate']
        )
        
        return output_path, vis_path if visualize else None
        
def main():
    parser = argparse.ArgumentParser(description="Audio-Visual Speech Enhancement")
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument("--output", required=True, help="Path to output enhanced audio")
    parser.add_argument("--visualize", action="store_true", help="Save visualization video")
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = AVSEPipeline()
    
    # Process video
    try:
        output_path, vis_path = pipeline.process_video(
            args.input,
            args.output,
            visualize=args.visualize
        )
        print(f"Enhanced audio saved to: {output_path}")
        if vis_path:
            print(f"Visualization saved to: {vis_path}")
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        
if __name__ == "__main__":
    main()
