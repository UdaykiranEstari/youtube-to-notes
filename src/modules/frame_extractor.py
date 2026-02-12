"""Video frame extraction at specified timestamps.

Uses OpenCV to capture screenshots from a video file.  Supports both direct
(fast, exact-timestamp) and smart (sharpness-optimized) extraction modes.
"""

import cv2
import os
from typing import List, Union

class FrameExtractor:
    """Extracts still frames from a video at requested timestamps.

    Supports two modes:

    * **Direct** — grabs the frame at the exact timestamp (fast).
    * **Smart** — searches a window around the timestamp and selects the
      sharpest frame using Laplacian variance (slower but higher quality).

    Args:
        output_dir: Directory where extracted JPEG images are saved.
    """

    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def extract_frames(self, video_path: str, timestamps: List[Union[str, float]], smart_extraction: bool = False) -> List[str]:
        """
        Extracts frames from the video at the given timestamps.
        If smart_extraction is True, searches a window for the sharpest frame.
        Otherwise, extracts the frame at the exact timestamp (faster).
        timestamps can be in seconds (float) or "MM:SS" / "HH:MM:SS" (str).
        Returns a list of paths to the saved images.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if smart_extraction:
            return self._extract_frames_smart(video_path, timestamps)
        else:
            return self._extract_frames_direct(video_path, timestamps)
    
    def _extract_frames_direct(self, video_path: str, timestamps: List[Union[str, float]]) -> List[str]:
        """Fast extraction: grab frame at exact timestamp."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        saved_paths = []

        for ts in timestamps:
            seconds = self._parse_timestamp(ts)
            frame_id = int(fps * seconds)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            
            if ret:
                # Format filename: video_name_timestamp.jpg
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                ts_str = str(int(seconds)).zfill(4)
                output_filename = f"{video_name}_{ts_str}.jpg"
                output_path = os.path.join(self.output_dir, output_filename)
                
                cv2.imwrite(output_path, frame)
                saved_paths.append(output_path)
            else:
                print(f"Warning: Could not extract frame at {ts}")

        cap.release()
        return saved_paths
    
    def _extract_frames_smart(self, video_path: str, timestamps: List[Union[str, float]]) -> List[str]:
        """Smart extraction: search window for sharpest frame."""

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        saved_paths = []
        
        # Search window configuration (optimized for accuracy)
        window_seconds = 1.0  # Look +/- this amount (Total 2s window)
        step_frames = 5  # Check every 5th frame for better precision

        for idx, ts in enumerate(timestamps, 1):
            if idx % 10 == 0:  # Print progress every 10 screenshots
                print(f"Processing screenshot {idx}/{len(timestamps)}...")
            
            target_seconds = self._parse_timestamp(ts)
            target_frame = int(fps * target_seconds)
            
            start_frame = max(0, int(target_frame - (fps * window_seconds)))
            end_frame = min(total_frames, int(target_frame + (fps * window_seconds)))
            
            best_frame = None
            best_score = -1.0
            best_frame_idx = -1

            # Scan the window
            for f_idx in range(start_frame, end_frame, step_frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                score = self._calculate_sharpness(frame)
                if score > best_score:
                    best_score = score
                    best_frame = frame
                    best_frame_idx = f_idx
            
            if best_frame is not None:
                # Format filename: video_name_timestamp_frame.jpg
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                # Use the original timestamp for the filename to keep it aligned with notes, 
                # but maybe append actual time? Let's stick to the requested timestamp for matching.
                ts_str = str(int(target_seconds)).zfill(4)
                output_filename = f"{video_name}_{ts_str}.jpg"
                output_path = os.path.join(self.output_dir, output_filename)
                
                cv2.imwrite(output_path, best_frame)
                saved_paths.append(output_path)
                # print(f"Saved sharpest frame for {ts} (Score: {best_score:.2f})")
            else:
                print(f"Warning: Could not extract frame at {ts}")

        cap.release()
        return saved_paths

    def _calculate_sharpness(self, image) -> float:
        """
        Calculate the sharpness of an image using the variance of the Laplacian.
        Higher variance means sharper edges.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _parse_timestamp(self, ts: Union[str, float]) -> float:
        """Convert a timestamp to seconds.

        Args:
            ts: Seconds as a number, or a string in ``"MM:SS"`` /
                ``"HH:MM:SS"`` format.

        Returns:
            Timestamp in seconds as a float.
        """
        if isinstance(ts, (int, float)):
            return float(ts)
        
        parts = ts.split(':')
        parts = [float(p) for p in parts]
        
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        elif len(parts) == 2:
            return parts[0] * 60 + parts[1]
        else:
            return parts[0]

if __name__ == "__main__":
    # Test
    # extractor = FrameExtractor()
    # extractor.extract_frames("output/video.mp4", ["00:10", 65.5])
    pass
