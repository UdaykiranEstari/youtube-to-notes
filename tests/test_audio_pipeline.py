import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.modules.downloader import YouTubeDownloader
from src.modules.audio_extractor import AudioExtractor
from src.modules.transcriber import Transcriber
from src.modules.content_analyzer import ContentAnalyzer
from src.modules.frame_extractor import FrameExtractor
import shutil
import re

def sanitize_filename(name: str) -> str:
    """Sanitizes a string to be safe for filenames."""
    safe_name = re.sub(r'[\\/*?:"<>|]', "", name)
    return safe_name.replace(" ", "_")

def test_audio_pipeline(url: str):
    print(f"Testing audio pipeline with video: {url}")
    
    # 1. Download Video
    print("1. Downloading video...")
    downloader = YouTubeDownloader("test_output")
    video_info = downloader.get_video_info(url)
    
    # Create Organized Folder
    safe_title = sanitize_filename(video_info["title"])
    video_output_dir = os.path.join("test_output", safe_title)
    if not os.path.exists(video_output_dir):
        os.makedirs(video_output_dir)
    
    # Move downloaded files
    new_video_path = os.path.join(video_output_dir, os.path.basename(video_info["video_path"]))
    shutil.move(video_info["video_path"], new_video_path)
    video_info["video_path"] = new_video_path
    
    if video_info["subtitle_path"]:
        new_subtitle_path = os.path.join(video_output_dir, os.path.basename(video_info["subtitle_path"]))
        shutil.move(video_info["subtitle_path"], new_subtitle_path)
        video_info["subtitle_path"] = new_subtitle_path

    # 2. Extract Audio
    print("2. Extracting audio...")
    audio_extractor = AudioExtractor(output_dir=video_output_dir)
    audio_path = audio_extractor.extract_audio(new_video_path)
    
    if not audio_path:
        print("Audio extraction failed!")
        return

    # 3. Transcribe Audio
    print("3. Transcribing audio (this may take a while)...")
    transcriber = Transcriber()
    transcription = transcriber.transcribe_audio(audio_path)
    
    if not transcription:
        print("Transcription failed!")
        return

    word_timestamps = transcriber.get_word_level_timestamps(transcription)
    transcript_text = transcription["text"]
    print(f"Transcription complete. Found {len(word_timestamps)} words.")
    
    # 4. Analyze Content with Inline Screenshots
    print("4. Analyzing content for inline screenshots...")
    analyzer = ContentAnalyzer()
    analysis = analyzer.analyze_transcript(transcript_text, video_info["title"], word_timestamps=word_timestamps)
    
    if not analysis:
        print("Content analysis failed!")
        return

    # 5. Extract Screenshots
    print("5. Extracting screenshots...")
    extractor = FrameExtractor(output_dir=video_output_dir)
    timestamps = []
    
    for section in analysis.get("sections", []):
        if isinstance(section.get("content"), list):
            for item in section["content"]:
                if item.get("type") == "screenshot":
                    ts = item.get("timestamp")
                    if ts:
                        timestamps.append(ts)
        else:
            print("Warning: Section content is not a list (expected inline format).")

    print(f"Found {len(timestamps)} screenshots to extract.")
    screenshot_paths = extractor.extract_frames(new_video_path, timestamps, smart_extraction=False)
    
    # 6. Generate Markdown
    print("6. Generating Markdown...")
    md_filename = f"{safe_title}_inline_test.md"
    md_path = os.path.join(video_output_dir, md_filename)
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {analysis.get('title', video_info['title'])}\n\n")
        f.write(f"**Source:** {url}\n\n")
        f.write("## Summary\n")
        f.write(f"{analysis.get('summary', 'No summary available.')}\n\n")
        
        screenshot_index = 0
        for section in analysis.get('sections', []):
            f.write(f"## {section['heading']}\n")
            
            if isinstance(section.get("content"), list):
                for item in section["content"]:
                    if item.get("type") == "text":
                        f.write(f"{item['text']}\n\n")
                    elif item.get("type") == "screenshot":
                        ts = item.get("timestamp")
                        caption = item.get("caption", f"Screenshot at {ts}")
                        if screenshot_index < len(screenshot_paths):
                            image_name = os.path.basename(screenshot_paths[screenshot_index])
                            f.write(f"![{caption}]({image_name})\n\n")
                            screenshot_index += 1
            else:
                f.write(f"{section['content']}\n\n")

    print(f"Success! Output saved to {md_path}")
    
    # Cleanup
    if os.path.exists(audio_path):
        os.remove(audio_path)
    if os.path.exists(new_video_path):
        os.remove(new_video_path)

if __name__ == "__main__":
    # Use a short video for testing
    test_url = "https://www.youtube.com/watch?v=jNQXAC9IVRw" # Me at the zoo (very short)
    test_audio_pipeline(test_url)
