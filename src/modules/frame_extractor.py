"""Video frame extraction at specified timestamps.

Uses OpenCV to capture screenshots from a video file.  Supports both direct
(fast, exact-timestamp) and smart (composite-scored, deduplicated) extraction
modes with ffmpeg fallback for codec compatibility.
"""

import cv2
import numpy as np
import os
import subprocess
from typing import List, Union, Optional, Dict, Tuple

class FrameExtractor:
    """Extracts still frames from a video at requested timestamps.

    Supports two modes:

    * **Direct** — grabs the frame at the exact timestamp (fast).
    * **Smart** — searches a ±5 s window around the timestamp and selects
      the best frame using a composite score (sharpness, temporal proximity,
      stability), with perceptual hash deduplication and ffmpeg fallback.

    Args:
        output_dir: Directory where extracted JPEG images are saved.
    """

    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Track perceptual hashes of extracted frames for dedup
        self._extracted_hashes: List[int] = []

    def extract_frames(self, video_path: str, timestamps: List[Union[str, float]], smart_extraction: bool = False) -> List[str]:
        """
        Extracts frames from the video at the given timestamps.
        If smart_extraction is True, searches a window for the best frame
        using composite scoring with deduplication.
        Otherwise, extracts the frame at the exact timestamp (faster).
        timestamps can be in seconds (float) or "MM:SS" / "HH:MM:SS" (str).
        Returns a list of paths to the saved images.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Reset dedup hashes for each extraction run
        self._extracted_hashes = []

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

            if not ret:
                # Try ffmpeg fallback
                frame = self._ffmpeg_extract_frame(video_path, seconds)

            if frame is not None:
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                ts_str = str(int(seconds)).zfill(4)
                output_filename = f"{video_name}_{ts_str}.jpg"
                output_path = os.path.join(self.output_dir, output_filename)

                cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved_paths.append(output_path)
            else:
                print(f"Warning: Could not extract frame at {ts}")

        cap.release()
        return saved_paths

    def _extract_frames_smart(self, video_path: str, timestamps: List[Union[str, float]]) -> List[str]:
        """Smart extraction: two-pass composite scoring with dedup and ffmpeg fallback.

        Pass 1 scans the search window and stores only lightweight metadata
        (sharpness score, histogram, frame index) — no frame pixel data is kept
        in memory.  Pass 2 scores all candidates, then seeks back to the single
        best frame to read and save it.  This reduces peak memory from ~300 MB
        per timestamp to ~1 MB.
        """

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        saved_paths = []

        # Search window configuration
        window_seconds = 5.0  # Look +/- this amount (total 10s window)
        step_frames = 3       # Check every 3rd frame for precision in wider window

        for idx, ts in enumerate(timestamps, 1):
            if idx % 10 == 0:
                print(f"Processing screenshot {idx}/{len(timestamps)}...")

            target_seconds = self._parse_timestamp(ts)
            target_frame = int(fps * target_seconds)

            start_frame = max(0, int(target_frame - (fps * window_seconds)))
            end_frame = min(total_frames, int(target_frame + (fps * window_seconds)))

            # --- Pass 1: collect lightweight metadata only ---
            # Each entry: (sharpness, histogram, frame_idx)
            candidates: List[Tuple[float, np.ndarray, int]] = []

            for f_idx in range(start_frame, end_frame, step_frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                sharpness = self._calculate_sharpness(frame)
                # Store a compact HSV histogram (~1.5 KB) instead of full frame (~3 MB)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
                cv2.normalize(hist, hist)
                candidates.append((sharpness, hist, f_idx))

            # ffmpeg fallback if no candidates from OpenCV
            if not candidates:
                frame = self._ffmpeg_extract_frame(video_path, target_seconds)
                if frame is not None:
                    phash = self._perceptual_hash(frame)
                    if not self._is_duplicate(phash):
                        video_name = os.path.splitext(os.path.basename(video_path))[0]
                        ts_str = str(int(target_seconds)).zfill(4)
                        output_filename = f"{video_name}_{ts_str}.jpg"
                        output_path = os.path.join(self.output_dir, output_filename)
                        cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        saved_paths.append(output_path)
                        self._extracted_hashes.append(phash)
                    else:
                        saved_paths.append(None)  # Mark as skipped for dedup
                else:
                    print(f"Warning: Could not extract frame at {ts}")
                    saved_paths.append(None)
                continue

            # --- Pass 2: score candidates using stored metadata ---
            sharpness_values = [c[0] for c in candidates]
            max_sharpness = max(sharpness_values)
            min_sharpness = min(sharpness_values)
            sharpness_range = max_sharpness - min_sharpness if max_sharpness != min_sharpness else 1.0

            best_frame_idx = -1
            best_score = -1.0

            for i, (sharpness, hist, f_idx) in enumerate(candidates):
                norm_sharpness = (sharpness - min_sharpness) / sharpness_range

                frame_distance = abs(f_idx - target_frame)
                max_distance = fps * window_seconds
                temporal_proximity = 1.0 - min(frame_distance / max_distance, 1.0)

                # Stability: use stored histograms for scene transition detection
                stability = 1.0
                if i > 0 and i < len(candidates) - 1:
                    prev_hist = candidates[i - 1][1]
                    next_hist = candidates[i + 1][1]
                    corr_prev = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                    corr_next = cv2.compareHist(hist, next_hist, cv2.HISTCMP_CORREL)
                    is_transition_prev = corr_prev < 0.3
                    is_transition_next = corr_next < 0.3
                    if is_transition_prev and is_transition_next:
                        stability = 0.0
                    elif is_transition_prev or is_transition_next:
                        stability = 0.2

                composite = (norm_sharpness * 0.5) + (temporal_proximity * 0.3) + (stability * 0.2)

                if composite > best_score:
                    best_score = composite
                    best_frame_idx = f_idx

            # --- Read only the single best frame ---
            if best_frame_idx >= 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame_idx)
                ret, best_frame = cap.read()
                if not ret:
                    best_frame = self._ffmpeg_extract_frame(video_path, best_frame_idx / fps)

                if best_frame is not None:
                    phash = self._perceptual_hash(best_frame)
                    if self._is_duplicate(phash):
                        saved_paths.append(None)
                        continue

                    video_name = os.path.splitext(os.path.basename(video_path))[0]
                    ts_str = str(int(target_seconds)).zfill(4)
                    output_filename = f"{video_name}_{ts_str}.jpg"
                    output_path = os.path.join(self.output_dir, output_filename)

                    cv2.imwrite(output_path, best_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    saved_paths.append(output_path)
                    self._extracted_hashes.append(phash)
                else:
                    print(f"Warning: Could not read best frame at {ts}")
                    saved_paths.append(None)
            else:
                print(f"Warning: Could not extract frame at {ts}")
                saved_paths.append(None)

        cap.release()
        return saved_paths

    def _calculate_sharpness(self, image) -> float:
        """
        Calculate the sharpness of an image using the variance of the Laplacian.
        Higher variance means sharper edges.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _is_scene_transition(self, frame_a, frame_b) -> bool:
        """Detect scene transition between two frames using HSV histogram correlation.

        Returns True if the frames are from different scenes (low correlation).
        """
        hsv_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2HSV)
        hsv_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2HSV)

        # Compute histograms on H and S channels
        hist_a = cv2.calcHist([hsv_a], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hist_b = cv2.calcHist([hsv_b], [0, 1], None, [50, 60], [0, 180, 0, 256])

        cv2.normalize(hist_a, hist_a)
        cv2.normalize(hist_b, hist_b)

        correlation = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)
        # Correlation < 0.3 indicates a significant scene change
        return correlation < 0.3

    def _perceptual_hash(self, frame) -> int:
        """Compute a 64-bit perceptual hash of a frame.

        Resize to 8x8, convert to grayscale, threshold at mean → 64-bit int.
        """
        resized = cv2.resize(frame, (8, 8), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        mean_val = gray.mean()
        bits = (gray > mean_val).flatten()
        # Pack bits into a single integer
        hash_val = 0
        for bit in bits:
            hash_val = (hash_val << 1) | int(bit)
        return hash_val

    def _hamming_distance(self, hash_a: int, hash_b: int) -> int:
        """Compute the Hamming distance between two perceptual hashes."""
        return bin(hash_a ^ hash_b).count('1')

    def _is_duplicate(self, phash: int, threshold: int = 10) -> bool:
        """Check if a frame's perceptual hash is too similar to any previously extracted frame."""
        for existing_hash in self._extracted_hashes:
            if self._hamming_distance(phash, existing_hash) < threshold:
                return True
        return False

    def _ffmpeg_extract_frame(self, video_path: str, timestamp_seconds: float) -> Optional[np.ndarray]:
        """Extract a single frame using ffmpeg as fallback when OpenCV fails.

        Returns the frame as a numpy array, or None on failure.
        """
        try:
            # Write to a temp file, then read back with OpenCV
            temp_path = os.path.join(self.output_dir, "_temp_ffmpeg_frame.jpg")
            subprocess.run(
                [
                    "ffmpeg", "-y", "-ss", str(timestamp_seconds),
                    "-i", video_path, "-frames:v", "1",
                    "-q:v", "2", temp_path
                ],
                capture_output=True, timeout=30
            )
            if os.path.exists(temp_path):
                frame = cv2.imread(temp_path)
                os.remove(temp_path)
                return frame
        except Exception as e:
            print(f"ffmpeg fallback failed at {timestamp_seconds}s: {e}")
        return None

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
