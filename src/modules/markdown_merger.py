"""Markdown chunk merger with timestamp adjustment.

Combines individually generated chunk markdown files into a single unified
document, adjusting timestamps by each chunk's offset and generating a
consolidated table of contents.
"""

import re
import os
from typing import List

class MarkdownMerger:
    """Merges per-chunk markdown files into a single unified document.

    Handles timestamp offset adjustment, deduplication of summary/TOC
    sections across chunks, and generation of a consolidated table of
    contents.
    """

    def __init__(self):
        pass

    def merge_markdowns(self, markdown_files: List[str], chunk_duration_minutes: int = 30) -> str:
        """Merge multiple chunk markdown files into one document.

        Timestamps in each chunk are shifted by the chunk's offset so they
        reflect positions in the original full-length video.  Repeated
        structural sections (Summary, TOC) are stripped from all chunks after
        the first.

        Args:
            markdown_files: Ordered list of paths to chunk markdown files.
            chunk_duration_minutes: Duration of each chunk in minutes, used
                to calculate timestamp offsets.

        Returns:
            Merged markdown string (without a top-level title — the caller
            is expected to prepend one).
        """
        merged_body = ""
        chunk_duration_sec = chunk_duration_minutes * 60
        all_headers = [] # List of (level, title, slug)

        for i, md_file in enumerate(markdown_files):
            if not os.path.exists(md_file):
                print(f"Warning: Markdown file not found: {md_file}")
                continue

            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()

            # 1. Adjust Timestamps
            offset_seconds = i * chunk_duration_sec
            
            def replace_timestamp(match):
                time_str = match.group(1)
                parts = list(map(int, time_str.split(':')))
                
                total_seconds = 0
                if len(parts) == 2: # MM:SS
                    total_seconds = parts[0] * 60 + parts[1]
                elif len(parts) == 3: # HH:MM:SS
                    total_seconds = parts[0] * 3600 + parts[1] * 60 + parts[2]
                
                new_total_seconds = total_seconds + offset_seconds
                
                # Format back to HH:MM:SS or MM:SS
                hours = new_total_seconds // 3600
                minutes = (new_total_seconds % 3600) // 60
                seconds = new_total_seconds % 60
                
                if hours > 0:
                    return f"[{hours:02d}:{minutes:02d}:{seconds:02d}]"
                else:
                    return f"[{minutes:02d}:{seconds:02d}]"

            # Adjust timestamps in content
            content = re.sub(r'\[(\d{1,2}:\d{2}(?::\d{2})?)\]', replace_timestamp, content)

            # 2. Parse and Clean Content
            lines = content.split('\n')
            cleaned_lines = []
            
            # State for stripping
            skip_section = False
            
            for line in lines:
                # Check for headers
                header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
                if header_match:
                    level = len(header_match.group(1))
                    title = header_match.group(2).strip()

                    # Strip TOC from all chunks (we generate a unified one)
                    if title.lower() == "table of contents":
                        skip_section = True
                        continue

                    # Strip Summary from chunks after the first (keep only first chunk's summary)
                    if i > 0 and title.lower() in ["summary", "introduction", "overview"]:
                        skip_section = True
                        continue
                    else:
                        skip_section = False

                    # Add to headers list for unified TOC
                    if not skip_section:
                        slug = title.lower().replace(' ', '-').replace('/', '-')
                        slug = re.sub(r'[^\w\-]', '', slug)
                        all_headers.append((level, title, slug))

                if not skip_section:
                    cleaned_lines.append(line)
            
            # Join cleaned lines
            chunk_content = '\n'.join(cleaned_lines).strip()
            
            # Append to merged body
            if i > 0:
                merged_body += "\n\n" # Just spacing, no separator for seamless flow
            merged_body += chunk_content

        # 3. Generate Unified TOC
        # Note: Title is added by the main processor, chunks only contain sections
        toc = "## Table of Contents\n\n"
        for level, title, slug in all_headers:
            if level >= 2:  # Include H2 and below in TOC
                indent = "  " * (level - 2)
                toc += f"{indent}- [{title}](#{slug})\n"

        # 4. Prepend TOC to merged content (title added separately by processor)
        final_content = toc + "\n" + merged_body

        return final_content
