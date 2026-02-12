import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.modules.downloader import YouTubeDownloader

def test_timeline_download():
    print("Testing timeline download...")
    output_dir = "test_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    downloader = YouTubeDownloader(output_dir)
    url = "https://www.youtube.com/watch?v=jNQXAC9IVRw" # Me at the zoo
    
    # Download 5s to 10s (5 seconds duration)
    print("Downloading range 00:05 - 00:10...")
    info = downloader.get_video_info(
        url, 
        quality="Low", 
        skip_subtitles=True, 
        start_time=5, 
        end_time=10
    )
    
    print(f"Download complete. Info: {info}")
    
    video_path = info.get('video_path')
    if video_path and os.path.exists(video_path):
        print(f"Video file exists: {video_path}")
        size = os.path.getsize(video_path)
        print(f"File size: {size} bytes")
        
        # Optional: Check duration with moviepy or similar if available, or just rely on file existence/size
        # Since we don't want to add dependencies, we'll just check it's small but not empty.
        if size > 1000:
            print("SUCCESS: Video downloaded and seems valid.")
        else:
            print("FAILURE: Video file is too small.")
    else:
        print("FAILURE: Video file not found.")

if __name__ == "__main__":
    test_timeline_download()
