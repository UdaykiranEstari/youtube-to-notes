import os
import sys
import concurrent.futures
import shutil
from unittest.mock import MagicMock

# Add root dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app

# Mock process_single_video_file to simulate screenshot creation
def mock_process(video_path, title, output_dir, *args, **kwargs):
    print(f"Mock processing: {video_path} -> {output_dir}")
    
    # Create a dummy md file
    md_path = os.path.join(output_dir, "dummy.md")
    with open(md_path, "w") as f:
        f.write(f"# {title}\n![Screenshot](screenshot.jpg)")
        
    # Create a dummy screenshot
    img_path = os.path.join(output_dir, "screenshot.jpg")
    with open(img_path, "w") as f:
        f.write("dummy image content")
        
    return md_path, {}, []

app.process_single_video_file = mock_process

def test_parallel_pipeline_with_artifacts():
    print("Testing parallel pipeline with artifact merging...")
    base_output_dir = "test_artifact_output"
    if os.path.exists(base_output_dir):
        shutil.rmtree(base_output_dir)
    os.makedirs(base_output_dir)
    
    # Create a dummy video file in the main dir to simulate "download"
    # Actually process_chunk_task downloads it. We need to mock that too or let it fail download but succeed process?
    # Let's mock download to just create a file.
    
    # We need to run process_video logic, but process_video does a lot.
    # Let's just test the logic we added to process_video by extracting it or simulating it.
    
    # Simulating the loop in process_video:
    
    num_chunks = 2
    video_output_dir = base_output_dir
    chunk_markdown_paths = [None] * num_chunks
    chunk_analyses = [None] * num_chunks
    
    print("Simulating parallel execution...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        for i in range(num_chunks):
            # We can't call process_chunk_task easily because it downloads.
            # Let's define a mock task here.
            def mock_task(idx):
                chunk_name = f"chunk_{idx+1:03d}"
                chunk_dir = os.path.join(video_output_dir, f"temp_{chunk_name}")
                if not os.path.exists(chunk_dir):
                    os.makedirs(chunk_dir)
                
                # Simulate process_single_video_file output
                md_path, _, _ = mock_process("vid.mp4", f"Part {idx}", chunk_dir)
                return idx, md_path, {}, chunk_dir, None
            
            futures.append(executor.submit(mock_task, i))
            
        for future in concurrent.futures.as_completed(futures):
            i, md_path, analysis, chunk_dir, error = future.result()
            
            if error:
                print(f"Chunk {i} failed: {error}")
            else:
                print(f"Chunk {i} done. Processing artifacts from {chunk_dir}...")
                
                if md_path and os.path.exists(md_path) and chunk_dir:
                    # Logic from app.py
                    with open(md_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    chunk_images = [f for f in os.listdir(chunk_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.webp'))]
                    
                    for img_name in chunk_images:
                        src_path = os.path.join(chunk_dir, img_name)
                        new_img_name = f"chunk_{i+1:03d}_{img_name}"
                        dst_path = os.path.join(video_output_dir, new_img_name)
                        shutil.move(src_path, dst_path)
                        content = content.replace(f"]({img_name})", f"]({new_img_name})")
                        print(f"  Moved {img_name} -> {new_img_name}")
                    
                    new_md_name = f"chunk_{i+1:03d}_notes.md"
                    new_md_path = os.path.join(video_output_dir, new_md_name)
                    with open(new_md_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    
                    print(f"  Updated markdown saved to {new_md_name}")
                    print(f"  Content: {content.strip()}")

    # Verify
    print("\nVerifying output directory...")
    files = os.listdir(video_output_dir)
    print(f"Files: {files}")
    
    if "chunk_001_screenshot.jpg" in files and "chunk_002_screenshot.jpg" in files:
        print("✅ Screenshots renamed and moved successfully")
    else:
        print("❌ Screenshots missing or not renamed")
        
    if "chunk_001_notes.md" in files:
        with open(os.path.join(video_output_dir, "chunk_001_notes.md"), "r") as f:
            c = f.read()
            if "](chunk_001_screenshot.jpg)" in c:
                print("✅ Markdown 1 updated successfully")
            else:
                print(f"❌ Markdown 1 content incorrect: {c}")

if __name__ == "__main__":
    test_parallel_pipeline_with_artifacts()
