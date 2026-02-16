"""Video processing pipeline.

Orchestrates the end-to-end flow for turning a YouTube URL or local video file
into structured markdown notes with embedded screenshots.  Long videos are
automatically split into time-based chunks that are processed in parallel via
:class:`concurrent.futures.ThreadPoolExecutor`, then merged into a single
output document.
"""

import os
import re
import shutil
import time
import subprocess
import json
import concurrent.futures
from src.modules.downloader import YouTubeDownloader
from src.modules.frame_extractor import FrameExtractor
from src.modules.content_analyzer import ContentAnalyzer
from src.modules.audio_extractor import AudioExtractor
from src.modules.transcriber import Transcriber
from src.modules.pdf_exporter import PDFExporter
from src.modules.markdown_merger import MarkdownMerger
from src.modules.notion_client import NotionClient
from src.modules.llm_providers import create_provider, LLMProvider
from src.utils.common import sanitize_filename, make_timestamp_clickable, parse_time_string

def process_single_video_file(
    video_path: str,
    video_title: str,
    output_dir: str,
    use_audio_transcription: bool,
    quick_summary_mode: bool,
    smart_extraction: bool,
    screenshot_density: str,
    detail_level: str,
    progress_callback=None,
    raw_response_filename: str = "gemini_raw_response.txt",
    video_url: str = "",
    llm_provider: LLMProvider = None
):
    """Process a single video file (full video or one chunk).

    Runs transcription, AI analysis, frame extraction, and markdown
    generation for a single video segment.  Supports resuming from a
    previously saved raw LLM response.

    Args:
        video_path: Path to the video file on disk.
        video_title: Human-readable title for the notes.
        output_dir: Directory for all generated artifacts.
        use_audio_transcription: Extract and transcribe audio instead of
            relying on VTT subtitles.
        quick_summary_mode: Generate a brief summary without screenshots.
        smart_extraction: Use sharpness-optimized frame selection.
        screenshot_density: ``"Low"``, ``"Medium"``, or ``"High"``.
        detail_level: ``"Brief"``, ``"Standard"``, or ``"Detailed"``.
        progress_callback: Optional ``(message, progress)`` callable for
            UI updates.
        raw_response_filename: Filename for caching the raw LLM response.
        video_url: Original YouTube URL (used for clickable timestamps).
        llm_provider: LLM backend to use for analysis.

    Returns:
        Tuple of ``(md_path, analysis_dict, screenshot_paths)``.

    Raises:
        Exception: If AI analysis fails to produce a result.
    """
    # Define raw output path
    raw_response_path = os.path.join(output_dir, raw_response_filename)
    
    word_timestamps = None
    transcript_text = ""
    resume_from_raw = False
    
    # Check if we can resume from raw response
    if os.path.exists(raw_response_path) and not quick_summary_mode:
        if progress_callback:
            progress_callback(f"Found existing AI response in {raw_response_filename}. Resuming...", 0.35)
        print(f"Resuming from existing raw response: {raw_response_path}")
        resume_from_raw = True
    
    if not resume_from_raw:
        # Use audio transcription if enabled
        if use_audio_transcription:
            if progress_callback:
                progress_callback("Extracting audio...", 0.15)
            
            audio_extractor = AudioExtractor(output_dir=output_dir)
            audio_path = audio_extractor.extract_audio(video_path)
            
            if audio_path:
                if progress_callback:
                    progress_callback("Transcribing audio...", 0.25)
                
                try:
                    transcriber = Transcriber()
                    transcription = transcriber.transcribe_audio(audio_path)
                    if transcription:
                        word_timestamps = transcriber.get_word_level_timestamps(transcription)
                        transcript_text = transcription["text"]
                        # Clean up audio file
                        if os.path.exists(audio_path):
                            os.remove(audio_path)
                except Exception as e:
                    print(f"Transcription failed: {e}. Falling back to VTT subtitles.")
                    word_timestamps = None
        
        # VTT logic is tricky for chunks. Let's rely on audio for now.
        if not transcript_text and not use_audio_transcription:
             # Try to find VTT in output_dir if it matches video name?
             # Or just return error?
             # For now, let's just proceed. If transcript empty, analyzer might fail or we handle it.
             pass

    if progress_callback:
        progress_callback("Analyzing content with AI...", 0.35)
    
    analyzer = ContentAnalyzer(provider=llm_provider)
    
    # Quick Summary Mode
    if quick_summary_mode:
        analysis = analyzer.generate_quick_summary(transcript_text, video_title)
        if not analysis:
            raise Exception("Failed to generate quick summary")
        
        # Generate simplified markdown
        safe_title = sanitize_filename(analysis.get('title', video_title))
        md_filename = f"{safe_title}.md"
        md_path = os.path.join(output_dir, md_filename)
        
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# {analysis.get('title', video_title)}\n\n")
            f.write(f"**Source:** {video_url}\n\n")
            f.write("## Quick Summary\n\n")
            f.write(f"{analysis.get('summary', 'No summary available.')}\n\n")
        
        return md_path, analysis, []

    # Full Analysis
    if word_timestamps:
        analysis = analyzer.analyze_transcript(
            transcript_text, video_title, word_timestamps=word_timestamps, 
            screenshot_density=screenshot_density, detail_level=detail_level,
            raw_output_path=raw_response_path
        )
    else:
        analysis = analyzer.analyze_transcript(
            transcript_text, video_title, word_timestamps=None, 
            screenshot_density=screenshot_density, detail_level=detail_level,
            raw_output_path=raw_response_path
        )
    
    if not analysis:
        raise Exception("Failed to analyze content")

    # Post-processing: ensure even screenshot count per section for 2-column grid
    for section in analysis.get("sections", []):
        if isinstance(section.get("content"), list):
            screenshot_items = [item for item in section["content"] if item.get("type") == "screenshot"]
            if len(screenshot_items) % 2 != 0 and len(screenshot_items) > 0:
                # Drop the last screenshot entry to make count even
                last_screenshot = screenshot_items[-1]
                section["content"].remove(last_screenshot)

    if progress_callback:
        progress_callback("Extracting screenshots...", 0.55)
    
    extractor = FrameExtractor(output_dir=output_dir)
    
    # Collect timestamps
    timestamps = []
    for section in analysis.get("sections", []):
        if isinstance(section.get("content"), list):
            for item in section["content"]:
                if item.get("type") == "screenshot":
                    ts = item.get("timestamp")
                    if ts: timestamps.append(ts)
        else:
            ts_list = section.get("timestamps", [])
            if section.get("timestamp"): ts_list.append(section.get("timestamp"))
            for ts in ts_list:
                if ts: timestamps.append(ts)
    
    raw_screenshot_paths = extractor.extract_frames(
        video_path, timestamps, smart_extraction=smart_extraction
    )

    # Handle dedup: remove screenshot entries from analysis where frames were skipped (None)
    # Build a set of skipped indices so we can strip them from the analysis sections
    skipped_indices = {i for i, p in enumerate(raw_screenshot_paths) if p is None}
    if skipped_indices:
        screenshot_idx = 0
        for section in analysis.get("sections", []):
            if isinstance(section.get("content"), list):
                filtered_content = []
                for item in section["content"]:
                    if item.get("type") == "screenshot":
                        if screenshot_idx not in skipped_indices:
                            filtered_content.append(item)
                        screenshot_idx += 1
                    else:
                        filtered_content.append(item)
                section["content"] = filtered_content

    screenshot_paths = [p for p in raw_screenshot_paths if p is not None]

    if progress_callback:
        progress_callback("Generating markdown...", 0.75)
    
    # Generate Markdown
    safe_title = sanitize_filename(analysis.get('title', video_title))
    md_filename = f"{safe_title}.md"
    md_path = os.path.join(output_dir, md_filename)
    
    with open(md_path, "w", encoding="utf-8") as f:
        # Note: Title and Source are added by the main processor when merging chunks
        # Chunks only contain Summary and Sections
        f.write("## Summary\n")
        f.write(f"{analysis.get('summary', 'No summary available.')}\n\n")
        
        # Generate Table of Contents
        sections = analysis.get('sections', [])
        if sections:
            f.write("## Table of Contents\n")
            for i, section in enumerate(sections, 1):
                heading = section.get('heading', f'Section {i}')
                # Create anchor-friendly slug
                slug = heading.lower().replace(' ', '-').replace('/', '-')
                slug = re.sub(r'[^\w\-]', '', slug)
                f.write(f"- [{heading}](#{slug})\n")
            f.write("\n")
        
        screenshot_index = 0
        used_screenshots = set()

        for section in sections:
            heading = section['heading']
            f.write(f"## {heading}\n")
            
            if isinstance(section.get("content"), list):
                for item in section["content"]:
                    if item.get("type") == "text":
                        f.write(f"{item['text']}\n\n")
                    elif item.get("type") == "screenshot":
                        ts = item.get("timestamp")
                        # Format timestamp
                        minutes = int(ts // 60)
                        seconds = int(ts % 60)
                        time_str = f"[{minutes:02d}:{seconds:02d}]"
                        clickable_time = make_timestamp_clickable(time_str, video_url)
                        
                        caption = item.get("caption", clickable_time)
                        if "Screenshot at" in caption: caption = clickable_time
                        else: caption = f"{caption} {clickable_time}"
                        
                        if screenshot_index < len(screenshot_paths):
                            image_path = screenshot_paths[screenshot_index]
                            image_name = os.path.basename(image_path)
                            if image_name not in used_screenshots:
                                f.write(f"![{caption}]({image_name})\n\n")
                                used_screenshots.add(image_name)
                            screenshot_index += 1
            else:
                f.write(f"{section['content']}\n\n")
                # Fallback screenshots
                ts_list = section.get("timestamps", [])
                if section.get("timestamp"): ts_list.append(section.get("timestamp"))
                for ts in ts_list:
                    if ts and screenshot_index < len(screenshot_paths):
                        image_name = os.path.basename(screenshot_paths[screenshot_index])
                        f.write(f"![Screenshot at {ts}]({image_name})\n\n")
                        screenshot_index += 1
    
    return md_path, analysis, screenshot_paths

def process_chunk_task(
    chunk_index: int,
    total_chunks: int,
    url: str,
    video_title: str,
    base_output_dir: str,
    start_time: int,
    end_time: int,
    use_audio_transcription: bool,
    quick_summary_mode: bool,
    smart_extraction: bool,
    screenshot_density: str,
    detail_level: str,
    video_quality: str,
    llm_provider: LLMProvider = None
):
    """Worker function that downloads and processes a single YouTube chunk.

    Designed to run inside a :class:`~concurrent.futures.ThreadPoolExecutor`.
    Downloads the specified time range, processes the video, and returns
    results to the caller for merging.

    Args:
        chunk_index: Zero-based index of this chunk.
        total_chunks: Total number of chunks being processed.
        url: YouTube video URL.
        video_title: Title of the full video.
        base_output_dir: Root output directory for the video.
        start_time: Chunk start offset in seconds.
        end_time: Chunk end offset in seconds.
        use_audio_transcription: Use audio transcription instead of VTT.
        quick_summary_mode: Generate brief summary only.
        smart_extraction: Use sharpness-optimized frame selection.
        screenshot_density: Screenshot density setting.
        detail_level: Note detail level setting.
        video_quality: Download quality setting.
        llm_provider: LLM backend instance.

    Returns:
        Tuple of ``(chunk_index, md_path, analysis, chunk_dir, error)``.
        On failure *md_path*, *analysis*, and *chunk_dir* are *None* and
        *error* contains the error message.
    """
    try:
        chunk_name = f"chunk_{chunk_index+1:03d}"
        
        # 1. Download Chunk
        # To ensure unique files for parallel downloads, we use a subfolder for each chunk
        chunk_dir = os.path.join(base_output_dir, f"temp_{chunk_name}")
        if not os.path.exists(chunk_dir):
            os.makedirs(chunk_dir)
            
        chunk_downloader = YouTubeDownloader(chunk_dir)
        
        # Download specific range
        info = chunk_downloader.get_video_info(
            url,
            quality=video_quality,
            skip_subtitles=True, # Chunks don't have subs usually
            start_time=start_time,
            end_time=end_time
        )
        
        video_path = info['video_path']
        
        # 2. Process the downloaded chunk
        # We need to pass a title that indicates it's a part
        part_title = f"{video_title} (Part {chunk_index+1})"
        
        md_path, analysis, _ = process_single_video_file(
            video_path,
            part_title,
            chunk_dir, 
            use_audio_transcription,
            quick_summary_mode,
            smart_extraction,
            screenshot_density,
            detail_level,
            progress_callback=None, # No callback from worker
            raw_response_filename=f"gemini_raw_response_{chunk_name}.txt",
            video_url=url,
            llm_provider=llm_provider
        )
        
        return chunk_index, md_path, analysis, chunk_dir, None
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return chunk_index, None, None, None, str(e)

def _get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe, with OpenCV fallback.

    Args:
        video_path: Path to the video file.

    Returns:
        Duration in seconds, or ``0`` if neither method succeeds.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", video_path
            ],
            capture_output=True, text=True, timeout=30
        )
        info = json.loads(result.stdout)
        return float(info["format"]["duration"])
    except Exception as e:
        print(f"ffprobe failed: {e}")
        # Fallback: try OpenCV
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if fps > 0:
            return frame_count / fps
        return 0


def _extract_video_segment(source_path: str, output_path: str, start_time: int, end_time: int):
    """Extract a time segment from a video file using ffmpeg stream-copy.

    Args:
        source_path: Path to the source video.
        output_path: Destination path for the extracted segment.
        start_time: Segment start in seconds.
        end_time: Segment end in seconds.

    Raises:
        Exception: If ffmpeg fails to produce the output file.
    """
    duration = end_time - start_time
    subprocess.run(
        [
            "ffmpeg", "-y", "-ss", str(start_time), "-i", source_path,
            "-t", str(duration), "-c", "copy", "-avoid_negative_ts", "1",
            output_path
        ],
        capture_output=True, timeout=600
    )
    if not os.path.exists(output_path):
        raise Exception(f"ffmpeg failed to extract segment {start_time}-{end_time}")


def process_local_chunk_task(
    chunk_index: int,
    total_chunks: int,
    source_video_path: str,
    video_title: str,
    base_output_dir: str,
    start_time: int,
    end_time: int,
    quick_summary_mode: bool,
    smart_extraction: bool,
    screenshot_density: str,
    detail_level: str,
    llm_provider: LLMProvider = None,
    video_url: str = ""
):
    """Worker function that extracts and processes a chunk from a local video.

    Similar to :func:`process_chunk_task` but operates on a local file
    instead of downloading from YouTube.  Always uses audio transcription.

    Args:
        chunk_index: Zero-based index of this chunk.
        total_chunks: Total number of chunks being processed.
        source_video_path: Path to the full local video file.
        video_title: Title for the notes.
        base_output_dir: Root output directory.
        start_time: Chunk start offset in seconds.
        end_time: Chunk end offset in seconds.
        quick_summary_mode: Generate brief summary only.
        smart_extraction: Use sharpness-optimized frame selection.
        screenshot_density: Screenshot density setting.
        detail_level: Note detail level setting.
        llm_provider: LLM backend instance.
        video_url: Original YouTube URL (used for clickable timestamps).

    Returns:
        Tuple of ``(chunk_index, md_path, analysis, chunk_dir, error)``.
    """
    try:
        chunk_name = f"chunk_{chunk_index+1:03d}"

        chunk_dir = os.path.join(base_output_dir, f"temp_{chunk_name}")
        if not os.path.exists(chunk_dir):
            os.makedirs(chunk_dir)

        # Extract segment using ffmpeg
        ext = os.path.splitext(source_video_path)[1]
        chunk_video_path = os.path.join(chunk_dir, f"{chunk_name}{ext}")
        _extract_video_segment(source_video_path, chunk_video_path, start_time, end_time)

        part_title = f"{video_title} (Part {chunk_index+1})"

        md_path, analysis, _ = process_single_video_file(
            chunk_video_path,
            part_title,
            chunk_dir,
            use_audio_transcription=True,  # Always use audio for local files
            quick_summary_mode=quick_summary_mode,
            smart_extraction=smart_extraction,
            screenshot_density=screenshot_density,
            detail_level=detail_level,
            progress_callback=None,
            raw_response_filename=f"gemini_raw_response_{chunk_name}.txt",
            video_url=video_url,
            llm_provider=llm_provider
        )

        return chunk_index, md_path, analysis, chunk_dir, None

    except Exception as e:
        import traceback
        traceback.print_exc()
        return chunk_index, None, None, None, str(e)


def process_local_video(
    video_path: str,
    video_title: str = None,
    quick_summary_mode: bool = False,
    smart_extraction: bool = False,
    screenshot_density: str = "Medium",
    detail_level: str = "Detailed",
    upload_to_notion: bool = False,
    chunk_duration: int = 30,
    progress_callback=None,
    llm_provider_key: str = "vertex_ai",
    llm_api_key: str = None,
    llm_model: str = None
):
    """Process a local video file through the full notes pipeline.

    Splits the video into chunks, processes them in parallel, merges the
    results, generates a PDF, and optionally uploads to Notion.

    Args:
        video_path: Path to the local video file.
        video_title: Title for the notes.  Derived from the filename when
            *None*.
        quick_summary_mode: Generate a brief summary without screenshots.
        smart_extraction: Use sharpness-optimized frame selection.
        screenshot_density: ``"Low"``, ``"Medium"``, or ``"High"``.
        detail_level: ``"Brief"``, ``"Standard"``, or ``"Detailed"``.
        upload_to_notion: Upload the finished notes to Notion.
        chunk_duration: Maximum chunk length in minutes.
        progress_callback: Optional ``(message, progress)`` callable.
        llm_provider_key: Provider identifier (see
            :func:`~src.modules.llm_providers.create_provider`).
        llm_api_key: API key for the chosen provider.
        llm_model: Model name override.

    Returns:
        Tuple of ``(output_dir, error, pdf_path, elapsed_time_str)``.
        On failure *output_dir* is *None* and *error* contains a message.
    """
    process_start_time = time.time()

    # Create LLM provider
    try:
        llm_provider = create_provider(llm_provider_key, api_key=llm_api_key, model=llm_model)
    except Exception as e:
        return None, f"Failed to initialize LLM provider: {e}", None, None

    try:
        if progress_callback:
            progress_callback("Reading video file...", 0.05)

        # Derive title from filename if not provided
        if not video_title:
            video_title = os.path.splitext(os.path.basename(video_path))[0]

        safe_title = sanitize_filename(video_title)
        video_output_dir = os.path.join("output", safe_title)
        if not os.path.exists(video_output_dir):
            os.makedirs(video_output_dir)

        # Get duration via ffprobe
        full_duration = _get_video_duration(video_path)
        if full_duration <= 0:
            return None, "Could not determine video duration.", None, None

        # Chunking
        CHUNK_DURATION_SECONDS = chunk_duration * 60

        chunks = []
        if full_duration <= CHUNK_DURATION_SECONDS or quick_summary_mode:
            chunks.append((0, int(full_duration)))
        else:
            current_time = 0
            while current_time < full_duration:
                end = min(current_time + CHUNK_DURATION_SECONDS, int(full_duration))
                chunks.append((current_time, end))
                current_time = end

        num_chunks = len(chunks)

        if progress_callback:
            progress_callback(f"Processing {num_chunks} chunk(s) ({full_duration/60:.1f} min total)...", 0.10)

        # Parallel execution
        chunk_markdown_paths = [None] * num_chunks
        chunk_analyses = [None] * num_chunks

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for i, (cs, ce) in enumerate(chunks):
                futures.append(executor.submit(
                    process_local_chunk_task,
                    i,
                    num_chunks,
                    video_path,
                    video_title,
                    video_output_dir,
                    cs,
                    ce,
                    quick_summary_mode,
                    smart_extraction,
                    screenshot_density,
                    detail_level,
                    llm_provider
                ))

            completed_count = 0
            for future in concurrent.futures.as_completed(futures):
                i, md_path, analysis, chunk_dir, error = future.result()

                if error:
                    print(f"Chunk {i} failed: {error}")
                else:
                    if md_path and os.path.exists(md_path) and chunk_dir:
                        try:
                            with open(md_path, "r", encoding="utf-8") as f:
                                content = f.read()

                            chunk_images = [f for f in os.listdir(chunk_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.webp'))]

                            for img_name in chunk_images:
                                src_path = os.path.join(chunk_dir, img_name)
                                new_img_name = f"chunk_{i+1:03d}_{img_name}"
                                dst_path = os.path.join(video_output_dir, new_img_name)
                                shutil.move(src_path, dst_path)
                                content = content.replace(f"]({img_name})", f"]({new_img_name})")

                            new_md_name = f"chunk_{i+1:03d}_notes.md"
                            new_md_path = os.path.join(video_output_dir, new_md_name)
                            with open(new_md_path, "w", encoding="utf-8") as f:
                                f.write(content)

                            chunk_markdown_paths[i] = new_md_path
                            chunk_analyses[i] = analysis

                        except Exception as e:
                            print(f"Error processing artifacts for chunk {i}: {e}")
                            chunk_markdown_paths[i] = md_path
                            chunk_analyses[i] = analysis
                    else:
                        chunk_markdown_paths[i] = md_path
                        chunk_analyses[i] = analysis

                completed_count += 1
                progress = 0.10 + (0.70 * (completed_count / num_chunks))
                if progress_callback:
                    progress_callback(f"Completed chunk {i+1}/{num_chunks}", progress)

        # Merge results
        valid_md_paths = [p for p in chunk_markdown_paths if p]

        if not valid_md_paths:
            return None, "All chunks failed to process.", None, None

        if progress_callback:
            progress_callback("Merging notes...", 0.85)

        merger = MarkdownMerger()
        merged_content = merger.merge_markdowns(valid_md_paths, chunk_duration_minutes=chunk_duration)

        md_filename = f"{safe_title}.md"
        md_path = os.path.join(video_output_dir, md_filename)

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# {video_title}\n\n")
            f.write(f"**Source:** Local file: {os.path.basename(video_path)}\n\n")
            f.write(merged_content)

        # Generate PDF
        if progress_callback:
            progress_callback("Generating PDF...", 0.90)

        pdf_path = None
        try:
            exporter = PDFExporter()
            pdf_filename = f"{safe_title}.pdf"
            pdf_path = os.path.join(video_output_dir, pdf_filename)
            exporter.convert_markdown_to_pdf(md_path, pdf_path)
        except Exception as e:
            print(f"Error generating PDF: {e}")

        # Export to Notion
        if upload_to_notion and not quick_summary_mode:
            full_analysis = {"title": video_title, "summary": "", "sections": []}

            # Single directory scan — group images by chunk prefix
            all_files = os.listdir(video_output_dir)
            images_by_chunk = {}
            for f in all_files:
                if f.endswith(('.jpg', '.png', '.webp')) and f.startswith('chunk_'):
                    prefix = f[:9]  # "chunk_NNN"
                    images_by_chunk.setdefault(prefix, []).append(f)

            all_screenshot_paths = []
            for i, analysis in enumerate(chunk_analyses):
                if analysis:
                    if i == 0:
                        full_analysis["summary"] = analysis.get("summary", "")
                    full_analysis["sections"].extend(analysis.get("sections", []))

                    chunk_prefix = f"chunk_{i+1:03d}"
                    chunk_imgs = sorted([
                        os.path.join(video_output_dir, f)
                        for f in images_by_chunk.get(chunk_prefix, [])
                    ])
                    all_screenshot_paths.extend(chunk_imgs)

            _export_to_notion(full_analysis, all_screenshot_paths, video_output_dir, progress_callback)

        if progress_callback:
            progress_callback("Complete!", 1.0)

        elapsed_time = time.time() - process_start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"

        return video_output_dir, None, pdf_path, time_str

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, str(e), None, None


def _export_to_notion(analysis, screenshot_paths, output_dir, progress_callback=None, video_url=""):
    """Upload structured analysis and screenshots to a Notion page.

    Args:
        analysis: Analysis dict with ``title``, ``summary``, and
            ``sections``.
        screenshot_paths: Ordered list of screenshot file paths.
        output_dir: Video output directory (unused but kept for
            consistency).
        progress_callback: Optional ``(message, progress)`` callable.
        video_url: Original video URL included as a callout block.
    """
    try:
        notion = NotionClient()
        if not notion.client:
            print("Notion client not initialized (missing API Key/Page ID).")
            return

        if progress_callback:
            progress_callback("Uploading images to Imgur...", 0.93)
        print("Starting Notion upload...")

        # Pre-upload all images in parallel
        imgur_urls = {}
        if screenshot_paths:
            imgur_urls = notion.upload_images_batch(screenshot_paths)

        if progress_callback:
            progress_callback("Building Notion page...", 0.95)

        blocks = []

        # Add Summary
        if "summary" in analysis:
            blocks.append(notion.create_text_block("Summary", "heading_2"))
            blocks.append(notion.create_text_block(analysis["summary"]))

        if video_url:
            blocks.append(notion.create_text_block(f"Source: {video_url}", "callout"))

        screenshot_index = 0
        used_screenshots = set()

        for section in analysis.get("sections", []):
            blocks.append(notion.create_text_block(section["heading"], "heading_2"))

            if isinstance(section.get("content"), list):
                for item in section["content"]:
                    if item.get("type") == "text":
                        blocks.append(notion.create_text_block(item["text"]))
                    elif item.get("type") == "screenshot":
                        ts = item.get("timestamp")
                        caption = item.get("caption", f"Screenshot at {ts}")

                        if screenshot_index < len(screenshot_paths):
                            image_path = screenshot_paths[screenshot_index]
                            image_name = os.path.basename(image_path)

                            if image_name not in used_screenshots:
                                imgur_url = imgur_urls.get(image_path)

                                if imgur_url:
                                    img_block = notion.create_image_block(imgur_url)
                                    if img_block:
                                        blocks.append(img_block)
                                        blocks.append(notion.create_text_block(caption, "quote"))
                                else:
                                    blocks.append(notion.create_text_block(f"📷 [Image: {caption}] (Upload failed)", "callout"))

                                used_screenshots.add(image_name)
                            screenshot_index += 1
            else:
                blocks.append(notion.create_text_block(section["content"]))

        page_url = notion.create_page(analysis.get("title", "Video Notes"), blocks)
        
        if page_url:
            print(f"Notion page created: {page_url}")
            if progress_callback:
                progress_callback(f"Notion export successful!", 0.98)
        else:
            print("Failed to create Notion page.")

    except Exception as e:
        print(f"Error exporting to Notion: {e}")
        import traceback
        traceback.print_exc()

def process_video(
    url: str,
    video_quality: str = "Medium",
    quick_summary_mode: bool = False,
    smart_extraction: bool = False,
    use_audio_transcription: bool = False,
    screenshot_density: str = "Medium",
    detail_level: str = "Detailed",
    upload_to_notion: bool = False,
    start_time: str = None,
    end_time: str = None,
    chunk_duration: int = 30,
    progress_callback=None,
    llm_provider_key: str = "vertex_ai",
    llm_api_key: str = None,
    llm_model: str = None
):
    """Process a YouTube video through the full notes pipeline.

    This is the main entry point for YouTube URL processing.  It downloads
    the video, splits it into time-based chunks, processes each chunk in
    parallel, merges the results, generates a PDF, and optionally uploads
    to Notion.

    Args:
        url: YouTube video URL.
        video_quality: Download resolution — ``"High"`` (1440p),
            ``"Medium"`` (1080p), or ``"Low"`` (720p).
        quick_summary_mode: Generate a brief summary without screenshots.
        smart_extraction: Use sharpness-optimized frame selection.
        use_audio_transcription: Use Whisper instead of VTT subtitles.
        screenshot_density: ``"Low"``, ``"Medium"``, or ``"High"``.
        detail_level: ``"Brief"``, ``"Standard"``, or ``"Detailed"``.
        upload_to_notion: Upload finished notes to Notion.
        start_time: Optional start time string (e.g. ``"1:30"``).
        end_time: Optional end time string.
        chunk_duration: Maximum chunk length in minutes.
        progress_callback: Optional ``(message, progress)`` callable.
        llm_provider_key: Provider identifier (see
            :func:`~src.modules.llm_providers.create_provider`).
        llm_api_key: API key for the chosen provider.
        llm_model: Model name override.

    Returns:
        Tuple of ``(output_dir, error, pdf_path, elapsed_time_str)``.
        On failure *output_dir* is *None* and *error* contains a message.
    """
    process_start_time = time.time()
    
    # Create LLM provider
    try:
        llm_provider = create_provider(llm_provider_key, api_key=llm_api_key, model=llm_model)
    except Exception as e:
        return None, f"Failed to initialize LLM provider: {e}", None, None
    
    try:
        if progress_callback:
            progress_callback("Fetching video info...", 0.05)
        
        # 1. Download the full video once (instead of per-chunk)
        downloader = YouTubeDownloader("output")
        video_info = downloader.get_video_info(url, quality=video_quality, skip_download=False)

        safe_title = sanitize_filename(video_info["title"])
        video_output_dir = os.path.join("output", safe_title)
        if not os.path.exists(video_output_dir):
            os.makedirs(video_output_dir)

        # Move downloaded video to output dir
        downloaded_video_path = video_info.get("video_path")
        if downloaded_video_path and os.path.exists(downloaded_video_path):
            if os.path.dirname(os.path.abspath(downloaded_video_path)) != os.path.abspath(video_output_dir):
                dst_video = os.path.join(video_output_dir, os.path.basename(downloaded_video_path))
                shutil.move(downloaded_video_path, dst_video)
                downloaded_video_path = dst_video

        # Move thumbnail to video folder if it exists in root
        if video_info.get("thumbnail_path") and os.path.exists(video_info["thumbnail_path"]):
            src_thumb = video_info["thumbnail_path"]
            if os.path.dirname(os.path.abspath(src_thumb)) != os.path.abspath(video_output_dir):
                dst_thumb = os.path.join(video_output_dir, os.path.basename(src_thumb))
                shutil.move(src_thumb, dst_thumb)
                video_info["thumbnail_path"] = dst_thumb

        # Move subtitle file to video folder if it exists
        if video_info.get("subtitle_path") and os.path.exists(video_info["subtitle_path"]):
            src_sub = video_info["subtitle_path"]
            if os.path.dirname(os.path.abspath(src_sub)) != os.path.abspath(video_output_dir):
                dst_sub = os.path.join(video_output_dir, os.path.basename(src_sub))
                shutil.move(src_sub, dst_sub)
                video_info["subtitle_path"] = dst_sub

        # Determine Duration and Range
        full_duration = video_info.get('duration', 0)
        
        # Parse user-specified range
        s_time = parse_time_string(start_time) or 0
        e_time = parse_time_string(end_time) or full_duration
        
        # Validate range
        if e_time > full_duration: e_time = full_duration
        if s_time >= e_time: s_time = 0 # Fallback if invalid
        
        target_duration = e_time - s_time
        
        # Chunking Configuration
        CHUNK_DURATION_MINUTES = chunk_duration
        CHUNK_DURATION_SECONDS = CHUNK_DURATION_MINUTES * 60
        
        chunks = []
        if target_duration <= CHUNK_DURATION_SECONDS or quick_summary_mode:
            # Single chunk
            chunks.append((s_time, e_time))
        else:
            # Split into chunks
            current_time = s_time
            while current_time < e_time:
                end = min(current_time + CHUNK_DURATION_SECONDS, e_time)
                chunks.append((current_time, end))
                current_time = end
                
        num_chunks = len(chunks)
        
        if progress_callback:
            progress_callback(f"Processing {num_chunks} chunks in parallel ({target_duration/60:.1f} min total)...", 0.10)

        # 2. Parallel Execution — split locally, process each chunk
        chunk_markdown_paths = [None] * num_chunks
        chunk_analyses = [None] * num_chunks

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for i, (cs, ce) in enumerate(chunks):
                futures.append(executor.submit(
                    process_local_chunk_task,
                    i,
                    num_chunks,
                    downloaded_video_path,
                    video_info['title'],
                    video_output_dir,
                    cs,
                    ce,
                    quick_summary_mode,
                    smart_extraction,
                    screenshot_density,
                    detail_level,
                    llm_provider,
                    video_url=url
                ))
            
            completed_count = 0
            for future in concurrent.futures.as_completed(futures):
                i, md_path, analysis, chunk_dir, error = future.result()
                
                if error:
                    print(f"Chunk {i} failed: {error}")
                else:
                    if md_path and os.path.exists(md_path) and chunk_dir:
                        try:
                            # Read original markdown
                            with open(md_path, "r", encoding="utf-8") as f:
                                content = f.read()
                            
                            # Find all images in chunk_dir
                            chunk_images = [f for f in os.listdir(chunk_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.webp'))]
                            
                            for img_name in chunk_images:
                                src_path = os.path.join(chunk_dir, img_name)
                                # Create unique name
                                new_img_name = f"chunk_{i+1:03d}_{img_name}"
                                dst_path = os.path.join(video_output_dir, new_img_name)
                                
                                # Move file
                                shutil.move(src_path, dst_path)
                                
                                # Update markdown content
                                content = content.replace(f"]({img_name})", f"]({new_img_name})")
                            
                            # Save updated markdown to MAIN dir (renamed)
                            new_md_name = f"chunk_{i+1:03d}_notes.md"
                            new_md_path = os.path.join(video_output_dir, new_md_name)
                            with open(new_md_path, "w", encoding="utf-8") as f:
                                f.write(content)
                                
                            chunk_markdown_paths[i] = new_md_path
                            chunk_analyses[i] = analysis
                            
                        except Exception as e:
                            print(f"Error processing artifacts for chunk {i}: {e}")
                            chunk_markdown_paths[i] = md_path
                            chunk_analyses[i] = analysis
                    else:
                        chunk_markdown_paths[i] = md_path
                        chunk_analyses[i] = analysis
                    
                completed_count += 1
                progress = 0.10 + (0.70 * (completed_count / num_chunks))
                if progress_callback:
                    progress_callback(f"Completed chunk {i+1}/{num_chunks}", progress)

        # 3. Merge Results
        valid_md_paths = [p for p in chunk_markdown_paths if p]
        
        if not valid_md_paths:
            return None, "All chunks failed to process.", None, None
            
        if progress_callback:
            progress_callback("Merging notes...", 0.85)
            
        merger = MarkdownMerger()
        merged_content = merger.merge_markdowns(valid_md_paths, chunk_duration_minutes=CHUNK_DURATION_MINUTES)
        
        # Save Merged Markdown
        md_filename = f"{safe_title}.md"
        md_path = os.path.join(video_output_dir, md_filename)
        
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# {video_info['title']}\n\n")
            f.write(f"**Source:** {url}\n\n")
            f.write(merged_content)
            
        # 4. Generate PDF
        if progress_callback:
            progress_callback("Generating PDF...", 0.90)
            
        pdf_path = None
        try:
            exporter = PDFExporter()
            pdf_filename = f"{safe_title}.pdf"
            pdf_path = os.path.join(video_output_dir, pdf_filename)
            exporter.convert_markdown_to_pdf(md_path, pdf_path)
        except Exception as e:
            print(f"Error generating PDF: {e}")

        # 5. Export to Notion (New Integration)
        if upload_to_notion and not quick_summary_mode:
            # We need to reconstruct screenhot paths for the merged analysis
            # Since merged_content is text, we'll try to re-use the chunk analyses
            # A simpler way for MVP: Just use the first chunk's analysis or try to aggregate
            # Actually, `chunk_analyses` has the structured data we need! 
            
            # Aggregate analysis sections
            full_analysis = {"title": video_info['title'], "summary": "", "sections": []}

            # Single directory scan — group images by chunk prefix
            all_files = os.listdir(video_output_dir)
            images_by_chunk = {}
            for f in all_files:
                if f.endswith(('.jpg', '.png', '.webp')) and f.startswith('chunk_'):
                    prefix = f[:9]  # "chunk_NNN"
                    images_by_chunk.setdefault(prefix, []).append(f)

            all_screenshot_paths = []
            for i, analysis in enumerate(chunk_analyses):
                if analysis:
                    if i == 0: full_analysis["summary"] = analysis.get("summary", "")
                    full_analysis["sections"].extend(analysis.get("sections", []))

                    chunk_prefix = f"chunk_{i+1:03d}"
                    chunk_imgs = sorted([
                        os.path.join(video_output_dir, f)
                        for f in images_by_chunk.get(chunk_prefix, [])
                    ])
                    all_screenshot_paths.extend(chunk_imgs)
            
            # Call export helper
            _export_to_notion(full_analysis, all_screenshot_paths, video_output_dir, progress_callback, url)

        if progress_callback:
            progress_callback("Complete!", 1.0)
        
        elapsed_time = time.time() - process_start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
        
        return video_output_dir, None, pdf_path, time_str
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, str(e), None, None

