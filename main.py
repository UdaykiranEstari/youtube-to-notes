import os
import sys
import argparse
import re
from modules.downloader import YouTubeDownloader
from modules.frame_extractor import FrameExtractor
from modules.content_analyzer import ContentAnalyzer
from modules.notion_client import NotionClient

def sanitize_filename(name: str) -> str:
    """Sanitizes a string to be safe for filenames."""
    safe_name = re.sub(r'[\\/*?:"<>|]', "", name)
    return safe_name.replace(" ", "_")

def main():
    parser = argparse.ArgumentParser(description="YouTube to Notion Agent")
    parser.add_argument("url", help="YouTube Video URL")
    args = parser.parse_args()

    print(f"Processing video: {args.url}")

    # 1. Initial Download to get Title (to create folder)
    # We need to know the title before we can create the folder, 
    # so we'll use a temporary downloader to get info first, or just download to a temp spot?
    # Actually, yt-dlp can get info without downloading.
    
    print("Fetching video info...")
    temp_downloader = YouTubeDownloader("output") # Default temp
    try:
        # We'll use a method to just get info first if possible, but our current get_video_info downloads.
        # Let's modify the flow: Download to 'output' first, then move? 
        # Or better: Update downloader to allow fetching info first. 
        # For now, let's download to default 'output' and then move files to the new folder.
        # Actually, let's just download everything to 'output' first as before, 
        # and then organize them.
        
        video_info = temp_downloader.get_video_info(args.url)
    except Exception as e:
        print(f"Error processing video: {e}")
        return

    # Create Organized Folder
    safe_title = sanitize_filename(video_info["title"])
    video_output_dir = os.path.join("output", safe_title)
    
    if not os.path.exists(video_output_dir):
        os.makedirs(video_output_dir)
        
    # Move downloaded files to the new directory
    import shutil
    
    new_video_path = os.path.join(video_output_dir, os.path.basename(video_info["video_path"]))
    shutil.move(video_info["video_path"], new_video_path)
    video_info["video_path"] = new_video_path
    
    if video_info["subtitle_path"]:
        new_subtitle_path = os.path.join(video_output_dir, os.path.basename(video_info["subtitle_path"]))
        shutil.move(video_info["subtitle_path"], new_subtitle_path)
        video_info["subtitle_path"] = new_subtitle_path

    print(f"Files organized in: {video_output_dir}")

    # 2. Parse Transcript
    transcript_text = ""
    if video_info["subtitle_path"]:
        print(f"Parsing transcript from {video_info['subtitle_path']}...")
        transcript_text = temp_downloader.parse_vtt(video_info["subtitle_path"])
    else:
        print("No subtitle file found. Proceeding without transcript.")
        
    if not transcript_text:
        print("Error: Could not obtain transcript. Exiting.")
        return

    # 3. Analyze Content
    print("Analyzing content with Gemini...")
    analyzer = ContentAnalyzer()
    analysis = analyzer.analyze_transcript(transcript_text, video_info["title"])
    
    if not analysis:
        print("Failed to analyze content.")
        return

    # 4. Extract Screenshots
    print("Extracting screenshots...")
    # Initialize extractor with the specific output directory
    extractor = FrameExtractor(output_dir=video_output_dir)
    timestamps = []
    for section in analysis.get("sections", []):
        ts_list = section.get("timestamps", [])
        if section.get("timestamp"):
            ts_list.append(section.get("timestamp"))
            
        for ts in ts_list:
            if ts:
                timestamps.append(ts)
            
    screenshot_paths = extractor.extract_frames(video_info["video_path"], timestamps)
    print(f"Extracted {len(screenshot_paths)} screenshots.")

    # 5. Generate Markdown (Always)
    print("Generating Markdown notes...")
    generate_markdown(video_info, analysis, screenshot_paths, args.url, video_output_dir)

    # 6. Create Notion Page (Optional)
    notion_api_key = os.getenv("NOTION_API_KEY")
    notion_page_id = os.getenv("NOTION_PAGE_ID")
    
    if notion_api_key and notion_page_id and notion_api_key != "your_notion_api_key" and notion_page_id != "your_parent_page_id":
        print("Creating Notion page...")
        notion = NotionClient()
        
        blocks = []
        
        # Add Summary
        blocks.append(notion.create_text_block("Summary", "heading_2"))
        blocks.append(notion.create_text_block(analysis.get("summary", "No summary available.")))
        
        # Add Sections
        for section in analysis.get("sections", []):
            blocks.append(notion.create_text_block(section["heading"], "heading_2"))
            blocks.append(notion.create_text_block(section["content"]))
            
            # Add Images
            ts_list = section.get("timestamps", [])
            if section.get("timestamp"):
                ts_list.append(section.get("timestamp"))
                
            for ts in ts_list:
                if ts:
                    blocks.append(notion.create_text_block(f"[Screenshot at {ts}] - Image upload not implemented in MVP", "callout"))

        page_url = notion.create_page(analysis.get("title", video_info["title"]), blocks)
        
        if page_url:
            print(f"Success! Notion page created: {page_url}")
        else:
            print("Failed to create Notion page.")
    else:
        print("Skipping Notion page creation (credentials missing or placeholders).")

    # 7. Cleanup
    if os.path.exists(video_info["video_path"]):
        print(f"Cleaning up video file: {video_info['video_path']}")
        os.remove(video_info["video_path"])

def generate_markdown(video_info, analysis, screenshot_paths, source_url, output_dir):
    # Use the sanitized title for filename
    safe_title = sanitize_filename(analysis.get('title', video_info['title']))
    
    md_filename = f"{safe_title}.md"
    md_path = os.path.join(output_dir, md_filename)
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {analysis.get('title', video_info['title'])}\n\n")
        f.write(f"**Source:** {source_url}\n\n")
        
        f.write("## Summary\n")
        f.write(f"{analysis.get('summary', 'No summary available.')}\n\n")
        
        screenshot_index = 0
        for section in analysis.get('sections', []):
            f.write(f"## {section['heading']}\n")
            f.write(f"{section['content']}\n\n")
            
            ts_list = section.get("timestamps", [])
            if section.get("timestamp"):
                ts_list.append(section.get("timestamp"))
            
            for ts in ts_list:
                if ts:
                    # Use relative path for markdown (just filename since they are in same dir)
                    if screenshot_index < len(screenshot_paths):
                        image_name = os.path.basename(screenshot_paths[screenshot_index])
                        f.write(f"![Screenshot at {ts}]({image_name})\n\n")
                        screenshot_index += 1
    
    print(f"Success! Markdown notes saved to: {md_path}")

if __name__ == "__main__":
    main()
