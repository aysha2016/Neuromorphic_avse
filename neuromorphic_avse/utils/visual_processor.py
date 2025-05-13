import cv2
import torch
import numpy as np
import face_alignment
from typing import Tuple, List, Optional
from dataclasses import dataclass

@dataclass
class LipRegion:
    """Data class for storing lip region information."""
    frame: np.ndarray
    landmarks: np.ndarray
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float

class VisualProcessor:
    """Handles visual processing and lip tracking for the neuromorphic system."""
    def __init__(
        self,
        target_size: Tuple[int, int] = (96, 96),
        face_detector: str = 'sfd',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.target_size = target_size
        self.device = device
        
        # Initialize face alignment detector
        self.face_detector = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._2D,
            flip_input=False,
            device=device,
            face_detector=face_detector
        )
        
        # Lip landmark indices (68-point facial landmarks)
        self.lip_indices = list(range(48, 68))
        
    def extract_lip_region(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Extract and normalize lip region from frame using landmarks."""
        # Get lip landmarks
        lip_landmarks = landmarks[self.lip_indices]
        
        # Calculate bounding box with padding
        x_min = int(np.min(lip_landmarks[:, 0])) - 10
        y_min = int(np.min(lip_landmarks[:, 1])) - 10
        x_max = int(np.max(lip_landmarks[:, 0])) + 10
        y_max = int(np.max(lip_landmarks[:, 1])) + 10
        
        # Ensure bounds are within frame
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(frame.shape[1], x_max)
        y_max = min(frame.shape[0], y_max)
        
        # Extract region
        lip_region = frame[y_min:y_max, x_min:x_max]
        
        # Resize to target size
        lip_region = cv2.resize(lip_region, self.target_size)
        
        # Convert to grayscale
        if len(lip_region.shape) == 3:
            lip_region = cv2.cvtColor(lip_region, cv2.COLOR_BGR2GRAY)
            
        return lip_region, (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def detect_lips(
        self,
        frame: np.ndarray
    ) -> Optional[LipRegion]:
        """Detect and extract lip region from frame."""
        # Convert to RGB for face alignment
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect landmarks
        landmarks = self.face_detector.get_landmarks(frame_rgb)
        
        if landmarks is None or len(landmarks) == 0:
            return None
            
        # Get first face
        landmarks = landmarks[0]
        
        # Extract lip region
        lip_region, bbox = self.extract_lip_region(frame, landmarks)
        
        # Calculate confidence based on landmark visibility
        confidence = np.mean(landmarks[self.lip_indices, 2]) if landmarks.shape[1] > 2 else 1.0
        
        return LipRegion(
            frame=lip_region,
            landmarks=landmarks[self.lip_indices],
            bbox=bbox,
            confidence=confidence
        )
    
    def preprocess_frame(
        self,
        frame: np.ndarray,
        normalize: bool = True
    ) -> Tuple[torch.Tensor, Optional[LipRegion]]:
        """Preprocess frame for the neuromorphic system."""
        # Detect and extract lip region
        lip_region = self.detect_lips(frame)
        
        if lip_region is None:
            return torch.zeros(self.target_size), None
            
        # Convert to tensor
        lip_tensor = torch.from_numpy(lip_region.frame).float()
        
        # Normalize if requested
        if normalize:
            lip_tensor = (lip_tensor - lip_tensor.mean()) / (lip_tensor.std() + 1e-8)
            
        return lip_tensor, lip_region
    
    def extract_motion_features(
        self,
        frames: List[torch.Tensor],
        window_size: int = 5
    ) -> torch.Tensor:
        """Extract motion features from sequence of lip frames."""
        if len(frames) < window_size:
            return torch.zeros(1, *self.target_size)
            
        # Stack frames
        frames_tensor = torch.stack(frames)
        
        # Calculate optical flow
        flow_features = []
        for i in range(len(frames) - 1):
            prev_frame = frames[i].numpy().astype(np.uint8)
            next_frame = frames[i + 1].numpy().astype(np.uint8)
            
            flow = cv2.calcOpticalFlowFarneback(
                prev_frame,
                next_frame,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )
            
            # Convert flow to magnitude and angle
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # Stack magnitude and angle
            flow_feature = np.stack([magnitude, angle], axis=-1)
            flow_features.append(torch.from_numpy(flow_feature).float())
            
        # Stack flow features
        flow_tensor = torch.stack(flow_features)
        
        # Apply temporal pooling
        pooled_features = []
        for i in range(len(flow_tensor) - window_size + 1):
            window = flow_tensor[i:i + window_size]
            pooled = torch.mean(window, dim=0)
            pooled_features.append(pooled)
            
        return torch.stack(pooled_features)
    
    def visualize_lip_region(
        self,
        frame: np.ndarray,
        lip_region: LipRegion,
        show_landmarks: bool = True
    ) -> np.ndarray:
        """Visualize detected lip region on frame."""
        # Create copy of frame
        vis_frame = frame.copy()
        
        # Draw bounding box
        x, y, w, h = lip_region.bbox
        cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw landmarks if requested
        if show_landmarks:
            for landmark in lip_region.landmarks:
                x, y = int(landmark[0]), int(landmark[1])
                cv2.circle(vis_frame, (x, y), 2, (0, 0, 255), -1)
                
        return vis_frame 