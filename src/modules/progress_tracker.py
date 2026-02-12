"""
Progress Tracker for resumable video processing.
Tracks completed steps in a JSON file to enable resume capability.
"""
import os
import json
from typing import List, Optional
from datetime import datetime

class ProgressTracker:
    """Persists processing progress to a JSON file for resume capability.

    Each processing step and chunk can be independently marked as complete.
    On restart the pipeline can skip already-finished work by querying
    this tracker.

    Args:
        output_dir: Directory where ``progress.json`` is stored.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.progress_file = os.path.join(output_dir, "progress.json")
        self._load()
    
    def _load(self):
        """Load progress from file or initialize empty."""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, "r") as f:
                    self.data = json.load(f)
            except:
                self.data = {"steps": {}, "chunks": {}}
        else:
            self.data = {"steps": {}, "chunks": {}}
    
    def _save(self):
        """Save progress to file."""
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.progress_file, "w") as f:
            json.dump(self.data, f, indent=2)
    
    def mark_step_complete(self, step_name: str):
        """Mark a processing step as complete."""
        self.data["steps"][step_name] = {
            "completed": True,
            "timestamp": datetime.now().isoformat()
        }
        self._save()
    
    def is_step_complete(self, step_name: str) -> bool:
        """Check if a step has been completed."""
        return self.data["steps"].get(step_name, {}).get("completed", False)
    
    def mark_chunk_complete(self, chunk_index: int, md_path: str):
        """Mark a chunk as processed."""
        self.data["chunks"][str(chunk_index)] = {
            "completed": True,
            "md_path": md_path,
            "timestamp": datetime.now().isoformat()
        }
        self._save()
    
    def is_chunk_complete(self, chunk_index: int) -> bool:
        """Check if a chunk has been processed."""
        return self.data["chunks"].get(str(chunk_index), {}).get("completed", False)
    
    def get_chunk_md_path(self, chunk_index: int) -> Optional[str]:
        """Get the markdown path for a completed chunk."""
        chunk_data = self.data["chunks"].get(str(chunk_index), {})
        return chunk_data.get("md_path")
    
    def get_completed_chunks(self) -> List[int]:
        """Get list of completed chunk indices."""
        return [int(k) for k, v in self.data["chunks"].items() if v.get("completed")]
    
    def reset(self):
        """Reset all progress (for fresh start)."""
        self.data = {"steps": {}, "chunks": {}}
        if os.path.exists(self.progress_file):
            os.remove(self.progress_file)
    
    def set_total_chunks(self, count: int):
        """Store total chunk count for progress display."""
        self.data["total_chunks"] = count
        self._save()
    
    def get_total_chunks(self) -> int:
        """Get total chunk count."""
        return self.data.get("total_chunks", 0)
