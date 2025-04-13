from utils.audio_utils import extract_audio_features
from utils.visual_utils import extract_visual_features
import moviepy.editor as mp

def process_video_file(video_path, output_audio_path):
    clip = mp.VideoFileClip(video_path)
    clip.audio.write_audiofile(output_audio_path)

def preprocess(video_path):
    audio_path = "dataset/temp_audio.wav"
    process_video_file(video_path, audio_path)
    audio_feats = extract_audio_features(audio_path)
    visual_feats = extract_visual_features(video_path)
    return audio_feats, visual_feats
