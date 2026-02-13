"""Audio transcription with word-level timestamps.

Uses `faster-whisper <https://github.com/SYSTRAN/faster-whisper>`_ for
efficient speech-to-text conversion.  A singleton model instance is shared
across threads to avoid redundant loading.
"""

import os
from typing import List, Dict, Optional
import json

import os
from typing import List, Dict, Optional
import threading

# Global singleton for the model to avoid reloading across threads/calls
_MODEL_INSTANCE = None
_MODEL_LOCK = threading.Lock()

class Transcriber:
    """Speech-to-text transcription using faster-whisper.

    The underlying model is loaded once and shared across all instances
    (singleton pattern) to avoid expensive reloads in multi-threaded
    pipelines.

    Args:
        model_size: Whisper model size (e.g. ``"base"``, ``"small"``,
            ``"medium"``).
    """

    def __init__(self, model_size: str = "base"):
        """Initialize transcriber with faster-whisper.

        Args:
            model_size: Whisper model size to load.
        """
        global _MODEL_INSTANCE
        
        if _MODEL_INSTANCE is None:
            with _MODEL_LOCK:
                if _MODEL_INSTANCE is None:
                    try:
                        from faster_whisper import WhisperModel
                        # Run on CPU with INT8 quantization (fast & low memory)
                        # device="auto" checks for CUDA/MPS but faster-whisper on Mac M1/M2 often runs best on CPU with int8 
                        # or 'auto' might pick up MPS if available but requires float16/32. 
                        # standard 'int8' on CPU is very fast for base model.
                        
                        print(f"Loading faster-whisper model '{model_size}'...")
                        _MODEL_INSTANCE = WhisperModel(model_size, device="cpu", compute_type="int8")
                        print("Model loaded successfully.")
                        
                    except ImportError:
                        raise ImportError("Please install faster-whisper: pip install faster-whisper")
        
        self.model = _MODEL_INSTANCE
    
    def transcribe_audio(self, audio_path: str, beam_size: int = 5) -> Optional[Dict]:
        """Transcribe an audio file and return word-level timestamps.

        Args:
            audio_path: Path to the audio file (WAV recommended).
            beam_size: Beam size for decoding. Use 1 for faster
                transcription (e.g. quick summary mode), 5 for best quality.

        Returns:
            Dict with ``text``, ``segments`` (list of segment dicts with
            ``start``, ``end``, ``text``, and ``words``), ``language``,
            and ``duration``.  Returns *None* on failure.

        Raises:
            FileNotFoundError: If *audio_path* does not exist.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            segments, info = self.model.transcribe(
                audio_path,
                beam_size=beam_size,
                word_timestamps=True
            )
            
            # fast-whisper returns a generator, so we must iterate to process
            transcript_text = ""
            processed_segments = []
            
            for segment in segments:
                transcript_text += segment.text + " "
                
                # Convert segment to dictionary format compatible with our pipeline
                seg_dict = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "words": []
                }
                
                if segment.words:
                    for word in segment.words:
                        seg_dict["words"].append({
                            "word": word.word.strip(),
                            "start": word.start,
                            "end": word.end,
                            "probability": word.probability
                        })
                
                processed_segments.append(seg_dict)
            
            return {
                "text": transcript_text.strip(),
                "segments": processed_segments,
                "language": info.language,
                "duration": info.duration
            }
            
        except Exception as e:
            print(f"Transcription error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_word_level_timestamps(self, transcription: Dict) -> List[Dict]:
        """Flatten segment-level results into a single word-timestamp list.

        Falls back to segment-level timestamps when individual word data is
        unavailable.

        Args:
            transcription: Dict returned by :meth:`transcribe_audio`.

        Returns:
            List of dicts, each with ``word``, ``start``, and ``end`` keys.
        """
        words = []
        
        for segment in transcription.get("segments", []):
            segment_words = segment.get("words", [])
            if segment_words:
                words.extend(segment_words)
            else:
                # Fallback: use segment-level timestamps
                words.append({
                    "word": segment["text"].strip(),
                    "start": segment["start"],
                    "end": segment["end"]
                })
        
        return words

if __name__ == "__main__":
    # Test
    pass
