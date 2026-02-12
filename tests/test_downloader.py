import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.modules.downloader import YouTubeDownloader
import os

def test_downloader():
    print("Testing YouTubeDownloader...")
    downloader = YouTubeDownloader(output_dir="test_output")
    
    # Use a very short video or a known safe one. 
    # "Me at the zoo" is a classic test video: https://www.youtube.com/watch?v=jNQXAC9IVRw
    url = "https://www.youtube.com/watch?v=yeim9FjR0dI"
    
    try:
        info = downloader.get_video_info(url)
        print("Success!")
        print(f"Title: {info['title']}")
        print(f"Video Path: {info['video_path']}")
        print(f"Subtitle Path: {info['subtitle_path']}")
        
        if os.path.exists(info['video_path']):
            print("Video file exists.")
        else:
            print("Video file missing.")
            
    except Exception as e:
        print(f"Downloader failed: {e}")

if __name__ == "__main__":
    test_downloader()
