"""YouTube video, subtitle, and thumbnail downloader.

Wraps `yt-dlp <https://github.com/yt-dlp/yt-dlp>`_ to fetch video files,
auto-generated or manual subtitles (VTT), and high-resolution thumbnails
from YouTube.
"""

import os
import re
import json
import yt_dlp
from typing import Dict, Optional

# Pre-compiled regex for stripping VTT inline tags
_VTT_TAG_RE = re.compile(r'<[^>]+>')

class YouTubeDownloader:
    """Downloads YouTube videos, subtitles, and thumbnails via yt-dlp.

    Args:
        output_dir: Directory where downloaded files are saved.  Created
            automatically if it does not exist.
    """

    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def get_video_info(self, url: str, quality: str = "Medium", skip_subtitles: bool = False, skip_download: bool = False, start_time: Optional[int] = None, end_time: Optional[int] = None, skip_thumbnail: bool = False) -> Dict:
        """Download a video (or just its metadata) and return file paths.

        Args:
            url: YouTube video URL.
            quality: Resolution cap — ``"High"`` (1440p), ``"Medium"``
                (1080p, default), or ``"Low"`` (720p).
            skip_subtitles: Skip subtitle download (useful when using audio
                transcription instead).
            skip_download: Only fetch metadata without downloading the video
                file.
            start_time: Optional start offset in seconds for partial download.
            end_time: Optional end offset in seconds for partial download.
            skip_thumbnail: Skip thumbnail download (useful for chunk
                processing where the main download already fetched it).

        Returns:
            Dict with keys ``id``, ``title``, ``video_path``,
            ``subtitle_path``, ``thumbnail_path``, ``duration``, and
            ``description``.
        """
        # Map quality to format selector
        if quality == "High":
            format_selector = 'bestvideo[height<=1440]+bestaudio/best[height<=1440]'
        elif quality == "Low":
            format_selector = 'bestvideo[height<=720]+bestaudio/best[height<=720]'
        else: # Medium (default)
            format_selector = 'bestvideo[height<=1080]+bestaudio/best[height<=1080]'
        
        ydl_opts = {
            'format': format_selector,  # Use quality-based format selection
            'outtmpl': os.path.join(self.output_dir, '%(id)s.%(ext)s'),
            'writeautomaticsub': not skip_subtitles,  # Only download subs if not skipping
            'writesubtitles': not skip_subtitles,     # Download manual subs if available
            'subtitlesformat': 'vtt',   # VTT is easier to parse
            'skip_download': skip_download,     # Allow skipping download
            'writethumbnail': not skip_thumbnail,  # Skip thumbnail for chunk processing
            'postprocessors': [],       # No conversion, keep original format (webm/mkv) for speed
            'quiet': False,
            'no_warnings': True,
            'ignoreerrors': True,  # Continue even if some formats fail (e.g. subtitles 429)
            'progress_hooks': [], # Removed verbose console print
        }

        # Add download ranges if specified
        if start_time is not None or end_time is not None:
             # yt_dlp expects a function or list of functions for download_ranges
             # The function receives (info_dict, ydl) and returns a list of sections
             # Each section is a dict with 'start_time' and 'end_time'
             def download_range_func(info_dict, ydl):
                 return [{'start_time': start_time or 0, 'end_time': end_time or float('inf')}]
             
             ydl_opts['download_ranges'] = download_range_func
             # Force keyframes at cuts for better precision (might re-encode, but safer)
             # Actually, yt-dlp download_ranges usually uses ffmpeg to cut.
             # We might need 'force_keyframes_at_cuts': True if we want precise cuts without re-encoding issues?
             # But let's stick to default first.


        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=not skip_download)
            
            video_id = info['id']
            video_title = info['title']
            video_ext = info.get('ext', 'mp4')
            video_path = os.path.join(self.output_dir, f"{video_id}.{video_ext}")
            
            # Single directory scan for subtitle and thumbnail
            subtitle_path = None
            thumbnail_path = None
            yt_dlp_thumb = None

            for file in os.listdir(self.output_dir):
                if not file.startswith(video_id):
                    continue
                if file.endswith('.vtt') and subtitle_path is None:
                    subtitle_path = os.path.join(self.output_dir, file)
                elif file.endswith(('.jpg', '.webp')):
                    if yt_dlp_thumb is None or file.endswith('.jpg'):
                        yt_dlp_thumb = os.path.join(self.output_dir, file)

            # Try high-res thumbnail first, fall back to yt-dlp downloaded one
            if not skip_thumbnail:
                thumbnail_path = self._download_high_res_thumbnail(video_id) or yt_dlp_thumb
            else:
                thumbnail_path = yt_dlp_thumb
            
            return {
                "id": video_id,
                "title": video_title,
                "video_path": video_path,
                "subtitle_path": subtitle_path,
                "thumbnail_path": thumbnail_path,
                "duration": info.get('duration'),
                "description": info.get('description')
            }

    def _download_high_res_thumbnail(self, video_id: str) -> Optional[str]:
        """
        Attempts to download the maxresdefault.jpg thumbnail directly.
        """
        import urllib.request
        
        url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
        output_path = os.path.join(self.output_dir, f"{video_id}_maxres.jpg")
        
        try:
            urllib.request.urlretrieve(url, output_path)
            # Check if it's a valid image (sometimes YouTube returns a small placeholder for 404)
            if os.path.getsize(output_path) > 1000: # Arbitrary small size check
                return output_path
            else:
                os.remove(output_path)
                return None
        except Exception:
            return None

    def parse_vtt(self, vtt_path: str) -> list:
        """Parse a WebVTT subtitle file into timestamped text segments.

        Deduplicates repeated lines (common in YouTube karaoke-style VTTs)
        and strips inline tags.

        Args:
            vtt_path: Path to the ``.vtt`` file.

        Returns:
            List of dicts with ``start`` (float seconds) and ``word``
            (cleaned text line) keys.
        """
        if not vtt_path or not os.path.exists(vtt_path):
            return []
            
        transcript_data = []
        current_start = 0.0
        
        with open(vtt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        def parse_timestamp(ts_str):
            try:
                parts = ts_str.split(':')
                if len(parts) == 3:
                    h, m, s = parts
                    return int(h) * 3600 + int(m) * 60 + float(s)
                elif len(parts) == 2:
                    m, s = parts
                    return int(m) * 60 + float(s)
                return 0.0
            except:
                return 0.0

        seen_lines = set()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line == 'WEBVTT':
                continue
            if line.startswith('NOTE'):
                continue
                
            if '-->' in line:
                # Parse timestamp line: "00:00:00.000 --> 00:00:02.000"
                try:
                    start_str = line.split('-->')[0].strip()
                    current_start = parse_timestamp(start_str)
                except:
                    pass
                continue
                
            # Content line — remove tags like <c> or <00:00:00>
            clean_line = _VTT_TAG_RE.sub('', line)
            
            # Avoid duplicates (VTT often repeats lines for karaoke effect)
            if clean_line and clean_line not in seen_lines:
                transcript_data.append({
                    'start': current_start,
                    'word': clean_line
                })
                seen_lines.add(clean_line)
                
        return transcript_data

if __name__ == "__main__":
    # Test
    downloader = YouTubeDownloader()
    # info = downloader.get_video_info("https://www.youtube.com/watch?v=tEHro6elD-E") # Me at the zoo
    # print(info)
