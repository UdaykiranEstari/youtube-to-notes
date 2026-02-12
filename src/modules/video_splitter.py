"""Video splitting into time-based chunks.

Uses ffmpeg stream-copy to split a video file into fixed-duration segments
for parallel processing.
"""

import os
import subprocess
import math
from typing import List

class VideoSplitter:
    """Splits a video file into fixed-duration chunks using ffmpeg.

    Uses stream-copy (no re-encoding) for fast splitting.  Chunk boundaries
    may not land on exact keyframes, but this is acceptable for note
    generation.

    Args:
        output_dir: Directory where chunk files are written.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds using ffprobe.

        Args:
            video_path: Path to the video file.

        Returns:
            Duration in seconds, or ``0.0`` on failure.
        """
        try:
            cmd = [
                "ffprobe", 
                "-v", "error", 
                "-show_entries", "format=duration", 
                "-of", "default=noprint_wrappers=1:nokey=1", 
                video_path
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            return float(result.stdout.strip())
        except Exception as e:
            print(f"Error getting video duration: {e}")
            return 0.0

    def split_video(self, video_path: str, chunk_duration_minutes: int = 30) -> List[str]:
        """Split a video into time-based chunks.

        If the video is shorter than one chunk, the original path is returned
        as-is.  Existing chunk files are skipped to support resume.

        Args:
            video_path: Path to the source video file.
            chunk_duration_minutes: Maximum duration of each chunk in minutes.

        Returns:
            Ordered list of file paths for the generated (or existing) chunks.
            Returns an empty list if duration cannot be determined or ffmpeg
            fails.
        """
        duration_sec = self.get_video_duration(video_path)
        if duration_sec == 0:
            return []

        chunk_duration_sec = chunk_duration_minutes * 60
        num_chunks = math.ceil(duration_sec / chunk_duration_sec)
        
        if num_chunks <= 1:
            return [video_path]

        chunk_paths = []
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        ext = os.path.splitext(video_path)[1]

        print(f"Splitting video into {num_chunks} chunks of {chunk_duration_minutes} minutes...")

        for i in range(num_chunks):
            start_time = i * chunk_duration_sec
            chunk_filename = f"{video_name}_chunk_{i+1:03d}{ext}"
            chunk_path = os.path.join(self.output_dir, chunk_filename)
            
            # Skip if chunk already exists
            if os.path.exists(chunk_path):
                print(f"Chunk {i+1} already exists: {chunk_path}")
                chunk_paths.append(chunk_path)
                continue

            # Use ffmpeg to split
            # -ss before -i is faster (input seeking) but less accurate. 
            # -ss after -i is slower (output seeking) but accurate.
            # For splitting, we want accuracy but also speed.
            # Using -ss before -i and re-encoding is safest for keyframes, but slow.
            # Using -c copy is fast but might start at non-keyframe.
            # Let's use stream copy (-c copy) for speed, as exact split point isn't critical for notes.
            
            cmd = [
                "ffmpeg",
                "-ss", str(start_time),
                "-i", video_path,
                "-t", str(chunk_duration_sec),
                "-c", "copy",
                "-y", # Overwrite
                chunk_path
            ]
            
            try:
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                chunk_paths.append(chunk_path)
                print(f"Created chunk {i+1}: {chunk_path}")
            except subprocess.CalledProcessError as e:
                print(f"Error splitting chunk {i+1}: {e}")
                # If copy fails, maybe try re-encoding? No, too slow.
                return []

        return chunk_paths
