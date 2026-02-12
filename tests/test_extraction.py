import os
import cv2
from src.modules.frame_extractor import FrameExtractor

def test_extraction():
    # Use the existing video chunk from previous run
    dummy_video_path = "output/800+_hours_of_Learning_Claude_Code_in_8_minutes_(2026_tutorial__unknown_tricks__newest_model)/temp_chunk_001/Ffh9OeJ7yxw.webm"
    
    if not os.path.exists(dummy_video_path):
        print(f"Error: Video file not found: {dummy_video_path}")
        return
    
    # Create test output dir
    test_out_dir = "output/test_frames_manual"
    if not os.path.exists(test_out_dir):
        os.makedirs(test_out_dir)
        
    extractor = FrameExtractor(output_dir=test_out_dir)
    
    print(f"Testing on video: {dummy_video_path}")
    
    # Test timestamps (e.g., 10s and 30s)
    timestamps = [10.0, 30.0]

    print("\nTesting Direct Extraction (Smart=False)...")
    try:
        paths = extractor.extract_frames(dummy_video_path, timestamps, smart_extraction=False)
        print(f"Direct Extraction Success: {len(paths)} frames extracted.")
        for p in paths: print(f" - {p}")
        
        # Verify files exist
        for p in paths:
            if os.path.exists(p): print(f"   [OK] File exists: {os.path.basename(p)}")
            else: print(f"   [FAIL] File missing: {p}")
            
    except Exception as e:
        print(f"Direct Extraction Failed: {e}")
        import traceback
        traceback.print_exc()

    print("\nTesting Smart Extraction (Smart=True)...")
    try:
        paths = extractor.extract_frames(dummy_video_path, timestamps, smart_extraction=True)
        print(f"Smart Extraction Success: {len(paths)} frames extracted.")
        for p in paths: print(f" - {p}")
        
        # Verify files exist
        for p in paths:
            if os.path.exists(p): print(f"   [OK] File exists: {os.path.basename(p)}")
            else: print(f"   [FAIL] File missing: {p}")
            
    except Exception as e:
        print(f"Smart Extraction Failed: {e}")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    test_extraction()
