import streamlit as st
import os
import sys
import tempfile
from pathlib import Path
import subprocess
import shutil
import time
import yt_dlp

# Add the current directory to the path (optional if running from root, but good for safety)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline.processor import process_video, process_local_video
from src.modules.exporter import Exporter
from src.modules.llm_providers import get_provider_choices, get_provider_key, get_models_for_provider, get_default_model, PROVIDER_CONFIG
from src.utils.common import sanitize_filename, make_timestamp_clickable, parse_time_string


import concurrent.futures
import re


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_video_preview(url: str):
    """
    Fetch video metadata (thumbnail, title, duration) without downloading.
    Cached with TTL of 1 hour.
    """
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'skip_download': True,
            'extract_flat': False,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
            # Get best thumbnail URL
            thumbnails = info.get('thumbnails', [])
            thumbnail_url = None
            
            # Look for maxresdefault or high quality thumbnail
            for thumb in reversed(thumbnails):  # Higher quality usually at end
                if thumb.get('url'):
                    thumbnail_url = thumb['url']
                    # Prefer larger thumbnails
                    if thumb.get('width', 0) >= 1280:
                        break
            
            # Fallback to standard YouTube thumbnail URL
            if not thumbnail_url:
                video_id = info.get('id')
                thumbnail_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
            
            duration = info.get('duration', 0)
            mins, secs = divmod(duration, 60)
            hours, mins = divmod(mins, 60)
            
            if hours > 0:
                duration_str = f"{int(hours)}:{int(mins):02d}:{int(secs):02d}"
            else:
                duration_str = f"{int(mins)}:{int(secs):02d}"
            
            return {
                'title': info.get('title', 'Unknown Title'),
                'thumbnail_url': thumbnail_url,
                'duration': duration,
                'duration_str': duration_str,
                'channel': info.get('channel', info.get('uploader', 'Unknown Channel')),
                'view_count': info.get('view_count'),
            }
    except Exception as e:
        print(f"Error fetching video preview: {e}")
        return None

def main():
    st.set_page_config(
        page_title="YouTube to Notes",
        page_icon="📝",
        layout="wide"
    )
    
    # Custom CSS for 'Never Too Small' inspired design
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Newsreader:ital,wght@0,400;0,600;1,400&display=swap');
        
        /* Global Font & Colors - Exclude Sidebar */
        html, body, [class*="css"] {
            font-family: 'DM Sans', sans-serif;
            color: #2B2B2B;
            background-color: #FFFFFF;
        }
        
        /* Reset Sidebar to default font */
        [data-testid="stSidebar"] {
            font-family: sans-serif !important;
        }
        [data-testid="stSidebar"] [class*="css"] {
            font-family: sans-serif !important;
        }
        
        /* Target Streamlit Markdown Containers specifically (Main Content Only) */
        .main [data-testid="stMarkdownContainer"] p, 
        .main [data-testid="stMarkdownContainer"] li, 
        .main [data-testid="stMarkdownContainer"] div {
            font-family: 'DM Sans', sans-serif !important;
            font-size: 1.05rem !important;
            line-height: 1.7 !important;
            color: #333 !important;
        }
        
        /* Headings */
        h1, h2, h3, h4, h5, h6, 
        .main [data-testid="stMarkdownContainer"] h1, 
        .main [data-testid="stMarkdownContainer"] h2, 
        .main [data-testid="stMarkdownContainer"] h3 {
            font-family: 'Newsreader', serif !important;
            color: #2B2B2B !important;
        }
        
        h1, .main [data-testid="stMarkdownContainer"] h1 {
            font-weight: 600;
            letter-spacing: -0.02em;
            font-size: 2.2rem !important;
            margin-top: 1.5em !important;
        }
        
        h2, .main [data-testid="stMarkdownContainer"] h2 {
            font-weight: 500;
            font-size: 1.8rem !important;
            margin-top: 1.2em !important;
        }
        
        h3, .main [data-testid="stMarkdownContainer"] h3 {
            font-family: 'DM Sans', sans-serif !important;
            font-weight: 700;
            font-size: 1.2rem !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 1em !important;
        }
        
        /* Button Styling - Aligned with Input */
        .stButton>button {
            width: 100%;
            border-radius: 4px;
            font-weight: 500;
            font-family: 'DM Sans', sans-serif;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding-top: 0.55rem;
            padding-bottom: 0.55rem;
            margin-top: 2px; /* Fine-tune alignment */
            transition: all 0.2s;
            border: 1px solid #E0E0E0;
        }
        
        /* Center Main Container */
        .block-container {
            max_width: 800px;
            padding-top: 3rem;
        }
        
        /* Remove Streamlit's default top padding/margin if needed */
        .main .block-container {
            padding-top: 2rem;
        }

        </style>
    """, unsafe_allow_html=True)
    
    # Header Image
    if os.path.exists("assets/maxresdefault.jpg"):
        st.image("assets/maxresdefault.jpg", width=200)
    
    # Layout: Title + Settings Button
    col_header, col_settings = st.columns([5, 1])
    with col_header:
        st.title("YouTube to Notes")
        st.markdown("Transform YouTube videos into detailed, editorial-style notes.")

    with col_settings:
        st.write("") # Spacer
        st.write("") # Spacer

        # Settings Popover
        with st.popover("⚙️ Settings", use_container_width=True):
            # --- LLM Provider (Top) ---
            st.markdown("### 🤖 AI Provider")
            col_provider, col_model = st.columns(2)
            with col_provider:
                llm_provider_display = st.selectbox(
                    "Provider",
                    get_provider_choices(),
                    index=0,
                    help="AI provider for content analysis"
                )

            llm_provider_key = get_provider_key(llm_provider_display)
            provider_config = PROVIDER_CONFIG.get(llm_provider_key, {})
            available_models = get_models_for_provider(llm_provider_key)
            default_model = get_default_model(llm_provider_key)
            default_idx = available_models.index(default_model) if default_model in available_models else 0

            with col_model:
                llm_model = st.selectbox(
                    "Model",
                    available_models,
                    index=default_idx,
                    help="Model for analysis"
                )

            llm_api_key = None
            if provider_config.get("requires_api_key", False):
                llm_api_key = st.text_input(
                    "API Key",
                    type="password",
                    placeholder="Enter API key...",
                    help=f"Required for {llm_provider_display}"
                )

            st.divider()

            # --- Video Settings ---
            st.markdown("### 📹 Video")
            col_quality, col_chunk = st.columns(2)
            with col_quality:
                video_quality = st.selectbox(
                    "Quality",
                    ["High (1440p)", "Medium (1080p)", "Low (720p)"],
                    index=1,
                    help="Higher = better screenshots, larger files"
                )
            with col_chunk:
                chunk_duration = st.selectbox(
                    "Chunk (min)",
                    [10, 15, 20, 30],
                    index=3,
                    help="Split long videos for parallel processing"
                )

            col_start, col_end = st.columns(2)
            with col_start:
                start_time_input = st.text_input("Start", placeholder="00:00", help="MM:SS or H:MM:SS")
            with col_end:
                end_time_input = st.text_input("End", placeholder="End", help="Leave empty for full video")

            st.divider()

            # --- Notes Options ---
            st.markdown("### 📝 Notes")
            col_detail, col_density = st.columns(2)
            with col_detail:
                detail_level = st.selectbox(
                    "Detail Level",
                    ["Standard", "Detailed", "More Detailed"],
                    index=1,
                    help="Standard: brief | Detailed: comprehensive | More: exhaustive"
                )
            with col_density:
                screenshot_density = st.selectbox(
                    "Screenshots",
                    ["Low (3-5)", "Medium (8-12)", "High (15-20)"],
                    index=1,
                    help="Screenshots per section"
                )

            quick_summary_mode = st.checkbox(
                "⚡ Quick Summary Only",
                value=False,
                help="Fast mode: title + summary only, no screenshots"
            )

            st.divider()

            # --- Advanced Options ---
            st.markdown("### ⚙️ Advanced")
            col_adv1, col_adv2 = st.columns(2)
            with col_adv1:
                smart_extraction = st.checkbox(
                    "Smart Frames",
                    value=False,
                    help="Select sharpest frame per timestamp",
                    disabled=quick_summary_mode
                )
            with col_adv2:
                use_audio_transcription = st.checkbox(
                    "Audio Transcription",
                    value=False,
                    help="Use Whisper for accurate timestamps",
                    disabled=quick_summary_mode
                )

            upload_to_notion = st.checkbox(
                "📤 Upload to Notion",
                value=False,
                help="Auto-upload notes to Notion after generation"
            )

    # Sidebar - History Only
    st.sidebar.title("History")
    st.sidebar.info("Select past videos to view notes.")


    # st.sidebar.divider()
    

    output_dir = Path("output")
    if output_dir.exists():
        folders = [f for f in output_dir.iterdir() if f.is_dir()]
        if folders:
            selected_folder = st.sidebar.selectbox(
                "Select a video",
                folders,
                format_func=lambda x: x.name.replace("_", " ")
            )
            if st.sidebar.button("Load Notes"):
                st.session_state.current_folder = selected_folder
            


        else:
            st.sidebar.info("No notes yet. Process a video to get started!")
    
    st.sidebar.divider()

    # --- Local Video Upload ---
    st.sidebar.subheader("Upload Video")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a local video file",
        type=["mp4", "mkv", "webm", "mov", "avi"],
        label_visibility="collapsed"
    )
    upload_title = st.sidebar.text_input(
        "Video title (optional)",
        placeholder="Defaults to filename",
        label_visibility="collapsed"
    )
    upload_btn = st.sidebar.button("Create Notes from Upload", type="primary")

    st.sidebar.divider()

    # Main Input Section (Clean & Minimal)
    col1, col2 = st.columns([3, 1], vertical_alignment="bottom")
    with col1:
        url = st.text_input(
            "YouTube URL",
            placeholder="Paste YouTube URL here...",
            label_visibility="collapsed"
        )
    with col2:
        process_btn = st.button("Create Notes", type="primary")
    
    # Video Preview - Show thumbnail when URL is entered
    if url and ('youtube.com' in url or 'youtu.be' in url):
        with st.spinner("Loading preview..."):
            preview = fetch_video_preview(url)
        
        if preview:
            # Create a nice preview card
            preview_col1, preview_col2 = st.columns([1, 2])
            
            with preview_col1:
                st.image(preview['thumbnail_url'], use_container_width=True)
            
            with preview_col2:
                st.markdown(f"### {preview['title']}")
                st.caption(f"🎬 {preview['channel']} • ⏱️ {preview['duration_str']}")
                
                if preview.get('view_count'):
                    view_count = preview['view_count']
                    if view_count >= 1_000_000:
                        view_str = f"{view_count / 1_000_000:.1f}M views"
                    elif view_count >= 1_000:
                        view_str = f"{view_count / 1_000:.1f}K views"
                    else:
                        view_str = f"{view_count} views"
                    st.caption(f"👁️ {view_str}")
            
            st.divider()
        
    def render_download_buttons(folder_path):
        """Helper to render download buttons inside the status container."""
        st.markdown("---")
        st.caption("📥 **Downloads**")
        dl_col1, dl_col2, dl_col3 = st.columns(3)
        
        # PDF Download
        pdf_files = list(folder_path.glob("*.pdf"))
        expected_pdf_name = f"{folder_path.name}.pdf"
        pdf_file = folder_path / expected_pdf_name
        if not pdf_file.exists() and pdf_files:
            pdf_file = pdf_files[0]
            
        with dl_col1:
            if pdf_file and pdf_file.exists():
                with open(pdf_file, "rb") as f:
                    st.download_button(
                        label="📄 PDF",
                        data=f,
                        file_name=pdf_file.name,
                        mime="application/pdf",
                        key=f"dl_pdf_{folder_path.name}_{int(time.time())}" # Unique key
                    )
            else:
                st.button("📄 PDF", disabled=True, key=f"dl_pdf_dis_{folder_path.name}")
        
        # Find Markdown file (similar logic to PDF)
        md_files = list(folder_path.glob("*.md"))
        # Expected name might match folder, otherwise take first
        expected_md_name = f"{folder_path.name}.md"
        md_file = folder_path / expected_md_name
        if not md_file.exists() and md_files:
            md_file = md_files[0]
        
        # HTML Download
        with dl_col2:
            html_path = folder_path / f"{md_file.stem}.html"
            if not html_path.exists() and md_file and md_file.exists():
                try:
                    exporter = Exporter(str(folder_path))
                    html_path = Path(exporter.export_to_html(str(md_file)))
                except Exception:
                    pass
            
            if html_path.exists():
                with open(html_path, "r", encoding="utf-8") as f:
                    st.download_button(
                        label="🌐 HTML",
                        data=f.read(),
                        file_name=html_path.name,
                        mime="text/html",
                        key=f"dl_html_{folder_path.name}_{int(time.time())}"
                    )
            else:
                st.button("🌐 HTML", disabled=True, key=f"dl_html_dis_{folder_path.name}")
                 
        # Text Download (Plain text)
        with dl_col3:
            txt_path = folder_path / f"{md_file.stem}.txt"
            if not txt_path.exists() and md_file and md_file.exists():
                 # Create txt from md
                 with open(md_file, "r", encoding="utf-8") as f: content = f.read()
                 with open(txt_path, "w", encoding="utf-8") as f: f.write(content)

            if txt_path.exists():
                with open(txt_path, "r", encoding="utf-8") as f:
                    text_content = f.read()
                st.download_button(
                    label="📄 Text",
                    data=text_content,
                    file_name=txt_path.name,
                    mime="text/plain",
                    key=f"dl_txt_{folder_path.name}_{int(time.time())}"
                )
            else:
                 st.button("📄 Text", disabled=True, key=f"dl_txt_dis_{folder_path.name}")

    # Determine which processing mode to use
    is_youtube_process = process_btn and url
    is_upload_process = upload_btn and uploaded_file

    # Save uploaded file to temp location if processing an upload
    _temp_video_path = None
    if is_upload_process:
        tmp_dir = tempfile.mkdtemp(prefix="yt_notes_upload_")
        _temp_video_path = os.path.join(tmp_dir, uploaded_file.name)
        with open(_temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    # Process video (YouTube URL or local upload)
    if is_youtube_process or is_upload_process:
        # Session state reset
        st.session_state.processing_logs = []
        st.session_state.processing_complete = False
        st.session_state.status_label = "Processing..."
        st.session_state.status_state = "running"

        with st.status("Processing...", expanded=True) as status:
            log_container = st.container()

            def update_progress(msg, progress=None):
                log_container.write(f"- {msg}")
                st.session_state.processing_logs.append(f"- {msg}")

            # Maps for settings
            density_map = {
                "Low (3-5)": "Low",
                "Medium (8-12)": "Medium",
                "High (15-20)": "High"
            }
            detail_map = {
                "Standard": "Brief",
                "Detailed": "Standard",
                "More Detailed": "More Detailed"
            }
            quality_map = {
                "High (1440p)": "High",
                "Medium (1080p)": "Medium",
                "Low (720p)": "Low"
            }

            if is_upload_process:
                # Process local video file
                folder, error, pdf_path, elapsed_time = process_local_video(
                    _temp_video_path,
                    video_title=upload_title or None,
                    quick_summary_mode=quick_summary_mode,
                    smart_extraction=smart_extraction,
                    screenshot_density=density_map[screenshot_density],
                    detail_level=detail_map[detail_level],
                    upload_to_notion=upload_to_notion,
                    chunk_duration=chunk_duration,
                    progress_callback=update_progress,
                    llm_provider_key=llm_provider_key,
                    llm_api_key=llm_api_key,
                    llm_model=llm_model
                )
            else:
                # Process YouTube URL
                folder, error, pdf_path, elapsed_time = process_video(
                    url,
                    quality_map[video_quality],
                    quick_summary_mode,
                    smart_extraction,
                    use_audio_transcription,
                    density_map[screenshot_density],
                    detail_level=detail_map[detail_level],
                    upload_to_notion=upload_to_notion,
                    start_time=start_time_input,
                    end_time=end_time_input,
                    chunk_duration=chunk_duration,
                    progress_callback=update_progress,
                    llm_provider_key=llm_provider_key,
                    llm_api_key=llm_api_key,
                    llm_model=llm_model
                )

            if error:
                status.update(label="Error", state="error", expanded=True)
                st.error(f"Error: {error}")

                st.session_state.status_label = "Error"
                st.session_state.status_state = "error"
                st.session_state.processing_complete = True
            else:
                status.update(label=f"Complete ({elapsed_time})", state="complete", expanded=True)
                st.success(f"Notes created in {elapsed_time}")

                st.session_state.status_label = f"Complete ({elapsed_time})"
                st.session_state.status_state = "complete"
                st.session_state.processing_complete = True
                st.session_state.current_folder = Path(folder)

                render_download_buttons(Path(folder))

    # Persistent Progress Container
    elif hasattr(st.session_state, 'processing_logs') and st.session_state.processing_logs:
        # If we have logs, show the status container
        # Use the saved label and state
        with st.status(st.session_state.get('status_label', "Processing Log"), expanded=False, state=st.session_state.get('status_state', "complete")):
            for log in st.session_state.processing_logs:
                st.write(log)
            
            # If complete and we have a folder, show buttons
            if st.session_state.get('processing_complete') and hasattr(st.session_state, 'current_folder') and st.session_state.current_folder:
                 render_download_buttons(st.session_state.current_folder)
    
    # Display notes
    if hasattr(st.session_state, 'current_folder') and st.session_state.current_folder:
        folder = st.session_state.current_folder
        
        # Find markdown file
        expected_md_name = f"{folder.name}.md"
        md_file = folder / expected_md_name
        
        if not md_file.exists():
            md_files = list(folder.glob("*.md"))
            main_files = [f for f in md_files if "(Part " not in f.name]
            if main_files: md_file = main_files[0]
            elif md_files: md_file = md_files[0]
            else: md_file = None
        
        if md_file and md_file.exists():
            # Read and display markdown content

            
            st.divider()
            
            # Display Video Thumbnail (Hero Image)
            thumbnail_files = list(folder.glob("*.jpg")) + list(folder.glob("*.webp"))
            # Filter out screenshots (which usually have timestamps in names or are in a subfolder, 
            # but here they are in the same folder. We assume the thumbnail is the one matching the video ID/Title pattern 
            # or simply the largest non-screenshot image. 
            # Simpler approach: yt-dlp saves thumbnail as [video_id].jpg. 
            # Screenshots are named [video_title]_screenshot_[timestamp].jpg.
            # So we look for a file that DOES NOT contain "_screenshot_".
            
            hero_thumbnail = None
            
            # Priority 1: Look for the high-res thumbnail we explicitly downloaded
            for img in thumbnail_files:
                if "_maxres" in img.name:
                    hero_thumbnail = img
                    break
            
            # Priority 2: Fallback to any non-screenshot image (standard yt-dlp thumb)
            if not hero_thumbnail:
                for img in thumbnail_files:
                    if "_screenshot_" not in img.name and "_maxres" not in img.name:
                        hero_thumbnail = img
                        break
            
            if hero_thumbnail:
                st.image(str(hero_thumbnail), width='stretch')

            # Read and display markdown with images
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse content into blocks (Text or Image)
            blocks = []
            current_text_block = []
            
            lines = content.split('\n')
            i = 0
            while i < len(lines):
                line = lines[i]
                if line.strip().startswith('!['):
                    # Flush text
                    if current_text_block:
                        blocks.append({"type": "text", "content": '\n'.join(current_text_block)})
                        current_text_block = []
                    
                    # Parse image

                    # Use greedy match for caption to handle nested brackets like [MM:SS]
                    match = re.search(r'\!\[(.*)\]\((.+)\)', line)
                    if match:
                        alt = match.group(1)
                        path = match.group(2)
                        full_path = folder / path
                        blocks.append({"type": "image", "path": full_path, "caption": alt})
                else:
                    current_text_block.append(line)
                i += 1
            
            if current_text_block:
                blocks.append({"type": "text", "content": '\n'.join(current_text_block)})
            
            # Render blocks with Mosaic Layout
            i = 0
            while i < len(blocks):
                block = blocks[i]
                
                if block["type"] == "text":
                    st.markdown(block["content"])
                    i += 1
                elif block["type"] == "image":
                    # Collect consecutive images
                    image_group = [block]
                    j = i + 1
                    while j < len(blocks):
                        next_block = blocks[j]
                        if next_block["type"] == "image":
                            image_group.append(next_block)
                            j += 1
                        elif next_block["type"] == "text" and not next_block["content"].strip():
                            # Skip empty text blocks
                            j += 1
                        else:
                            break
                    
                    # Render the group
                    count = len(image_group)
                    
                    if count == 1:
                        # Single image -> Full width
                        img = image_group[0]
                        if img["path"].exists():
                            st.image(str(img["path"]), caption=img["caption"])
                            
                    elif count == 2:
                        # Two images -> Side by side (50/50)
                        c1, c2 = st.columns(2)
                        with c1:
                            if image_group[0]["path"].exists():
                                st.image(str(image_group[0]["path"]), caption=image_group[0]["caption"])
                        with c2:
                            if image_group[1]["path"].exists():
                                st.image(str(image_group[1]["path"]), caption=image_group[1]["caption"])
                                
                    else:
                        # 3+ images -> Mosaic (First is Hero, rest are Grid)
                        # Hero
                        if image_group[0]["path"].exists():
                            st.image(str(image_group[0]["path"]), caption=image_group[0]["caption"], width='stretch')
                        
                        # Grid for the rest
                        remaining = image_group[1:]
                        # Create rows of 2
                        for k in range(0, len(remaining), 2):
                            batch = remaining[k:k+2]
                            cols = st.columns(len(batch))
                            for idx, col in enumerate(cols):
                                with col:
                                    if batch[idx]["path"].exists():
                                        st.image(str(batch[idx]["path"]), caption=batch[idx]["caption"])

                    # Advance index
                    i = j

if __name__ == "__main__":
    main()
