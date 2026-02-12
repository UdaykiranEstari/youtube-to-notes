"""Shared utility helpers used across the pipeline.

Provides filename sanitization, YouTube timestamp linking, and flexible
time-string parsing.
"""

import re

def sanitize_filename(name: str) -> str:
    """Remove characters unsafe for filenames and replace spaces with underscores.

    Args:
        name: Raw filename string (e.g. a video title).

    Returns:
        Sanitized string safe for use as a file or directory name.
    """
    # Logic from app.py
    safe_name = re.sub(r'[\\/*?:"<>|]', "", name)
    return safe_name.replace(" ", "_")

def make_timestamp_clickable(time_str: str, video_url: str) -> str:
    """
    Converts a timestamp string like '[04:30]' into a clickable YouTube link.
    
    Args:
        time_str: Timestamp in format [MM:SS] or [H:MM:SS]
        video_url: YouTube video URL
    
    Returns:
        Markdown link like [[04:30]](https://youtube.com/watch?v=ID&t=270s)
    """
    # Extract time from brackets [MM:SS]
    time_match = re.search(r'\[(\d+):(\d+)(?::(\d+))?\]', time_str)
    if not time_match:
        return time_str  # Return as-is if parsing fails
    
    # Parse hours, minutes, seconds
    if time_match.group(3):  # Has hours [H:MM:SS]
        hours = int(time_match.group(1))
        minutes = int(time_match.group(2))
        seconds = int(time_match.group(3))
        total_seconds = hours * 3600 + minutes * 60 + seconds
    else:  # Just [MM:SS]
        minutes = int(time_match.group(1))
        seconds = int(time_match.group(2))
        total_seconds = minutes * 60 + seconds
    
    # Build YouTube timestamp URL
    # Check if URL already has query parameters
    separator = '&' if '?' in video_url else '?'
    timestamp_url = f"{video_url}{separator}t={total_seconds}s"
    
    # Return as clickable markdown link
    return f"[{time_str}]({timestamp_url})"

def parse_time_string(time_str: str) -> int:
    """Parse a human-readable time string into total seconds.

    Supported formats:

    * ``"MM:SS"`` or ``"H:MM:SS"`` (e.g. ``"04:30"``, ``"1:30:00"``)
    * ``"Xm Ys"`` (e.g. ``"4m 30s"``)
    * ``"Xs"`` (e.g. ``"270s"``)
    * Plain integer (treated as seconds)

    Args:
        time_str: Time string to parse.

    Returns:
        Total seconds as an integer, or *None* if the string is empty or
        unparseable.
    """
    if not time_str:
        return None
        
    time_str = time_str.strip().lower()
    if not time_str:
        return None
        
    # Try MM:SS or H:MM:SS
    if ':' in time_str:
        parts = time_str.split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            
    # Try Xm Ys
    if 'm' in time_str or 's' in time_str:
        minutes = 0
        seconds = 0
        
        m_match = re.search(r'(\d+)m', time_str)
        if m_match:
            minutes = int(m_match.group(1))
            
        s_match = re.search(r'(\d+)s', time_str)
        if s_match:
            seconds = int(s_match.group(1))
            
        return minutes * 60 + seconds
        
    # Try plain number
    try:
        return int(time_str)
    except ValueError:
        return None
