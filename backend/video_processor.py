from moviepy.editor import VideoFileClip
import numpy as np
from pydub import AudioSegment
import os

class VideoProcessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.video = VideoFileClip(filepath)
    
    def cut_silence(self, threshold=-40, min_silence_len=500):
        """Cut silent parts from video"""
        audio = AudioSegment.from_file(self.filepath)
        
        # Detect silent parts
        chunks = []
        is_silent = False
        current_chunk_start = 0
        
        for i in range(len(audio) - min_silence_len):
            segment = audio[i:i + min_silence_len]
            if segment.dBFS < threshold:
                if not is_silent:
                    chunks.append((current_chunk_start, i))
                    is_silent = True
            else:
                if is_silent:
                    current_chunk_start = i
                    is_silent = False
        
        if not is_silent:
            chunks.append((current_chunk_start, len(audio)))
        
        # Create new video without silent parts
        final_clips = []
        for start, end in chunks:
            clip = self.video.subclip(start/1000, end/1000)
            final_clips.append(clip)
        
        return final_clips
    
    def generate_thumbnail(self, time=None):
        """Generate thumbnail from video"""
        if time is None:
            time = self.video.duration / 2
        
        frame = self.video.get_frame(time)
        return frame
    
    def enhance_audio(self, boost_amount=1.5):
        """Enhance audio quality"""
        audio = self.video.audio
        enhanced = audio.volumex(boost_amount)
        return enhanced
    
    def cleanup(self):
        """Clean up resources"""
        self.video.close()
        if os.path.exists(self.filepath):
            os.remove(self.filepath)