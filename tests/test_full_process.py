import os
import sys
import shutil

# Add current dir to path
# Add root dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.processor import process_video

def test_full_process():
    print("Testing full process with 10 min chunks...")
    
    # User provided URL
    url = "https://youtu.be/0J2_YGuNrDo?si=JRIuYjvBI62lvIBT"
    
    # Settings
    video_quality = "Low" # Use Low for speed in test, user asked for test but didn't specify quality, Low is safer for speed.
    # Wait, user asked for "detailed notes and medium density screenshots".
    # And "export file to pdf and markdown".
    
    chunk_duration = 10
    detail_level = "Standard" # "Detailed" maps to "Standard" in app.py logic
    screenshot_density = "Medium"
    
    print(f"URL: {url}")
    print(f"Chunk Duration: {chunk_duration} min")
    
    # We need to mock the progress callback
    def progress_callback(msg, p):
        print(f"[Progress {p:.2f}]: {msg}")
        
    # Run process_video
    # Note: process_video signature:
    # url, video_quality, quick_summary_mode, smart_extraction, use_audio_transcription, 
    # screenshot_density, detail_level, upload_to_notion, start_time, end_time, chunk_duration, progress_callback
    
    try:
        folder, error, pdf_path, elapsed = process_video(
            url,
            video_quality,
            False, # quick_summary_mode
            False, # smart_extraction
            False, # use_audio_transcription
            screenshot_density,
            detail_level,
            False, # upload_to_notion
            None, # start_time
            None, # end_time
            chunk_duration,
            progress_callback
        )
        
        if error:
            print(f"FAILURE: {error}")
        else:
            print(f"SUCCESS! Elapsed: {elapsed}")
            print(f"Output Folder: {folder}")
            print(f"PDF Path: {pdf_path}")
            
            # Verify files
            if os.path.exists(folder):
                md_files = [f for f in os.listdir(folder) if f.endswith('.md')]
                print(f"Markdown files: {md_files}")
                
                if pdf_path and os.path.exists(pdf_path):
                    print(f"PDF file exists and size is {os.path.getsize(pdf_path)} bytes")
                else:
                    print("PDF file missing!")
            else:
                print("Output folder missing!")
                
    except Exception as e:
        print(f"EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_full_process()
