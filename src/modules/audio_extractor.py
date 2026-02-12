"""Audio stream extraction from video files.

Uses ffmpeg to demux and convert the audio track of a video into a 16 kHz
mono WAV file suitable for Whisper transcription.
"""

import subprocess
import os
from typing import Optional

class AudioExtractor:
    """Extracts the audio track from a video file via ffmpeg.

    The output is a 16 kHz mono WAV file optimized for downstream Whisper
    transcription.

    Args:
        output_dir: Directory where the extracted audio file is written.
    """

    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def extract_audio(self, video_path: str) -> Optional[str]:
        """Extract audio from a video file.

        Args:
            video_path: Path to the source video.

        Returns:
            Path to the extracted ``.wav`` file, or *None* if extraction
            fails.

        Raises:
            FileNotFoundError: If *video_path* does not exist.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Generate output filename (WAV for faster processing with Whisper)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        audio_filename = f"{video_name}.wav"
        audio_path = os.path.join(self.output_dir, audio_filename)
        
        # Use ffmpeg to extract audio
        # Optimizations:
        # -acodec pcm_s16le: Uncompressed WAV (fastest to decode for Whisper)
        # -ar 16000: Resample to 16kHz (what Whisper expects, saves resizing later)
        # -ac 1: Mono channel (Whisper mixes to mono anyway)
        try:
            command = [
                'ffmpeg',
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le', 
                '-ar', '16000',
                '-ac', '1',
                '-y',  # Overwrite output file
                audio_path
            ]
            
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            
            if os.path.exists(audio_path):
                return audio_path
            else:
                print("Audio extraction failed: output file not created")
                return None
                
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e.stderr.decode()}")
            return None
        except FileNotFoundError:
            print("FFmpeg not found. Please install ffmpeg.")
            return None

if __name__ == "__main__":
    # Test
    pass
