# YouTube to Notes

Transform YouTube videos into beautiful, editorial-style notes with AI-powered analysis and intelligent screenshot extraction.

## ✨ Features

### Core Capabilities
- **AI-Powered Analysis**: Uses Gemini AI to analyze video content and generate structured, comprehensive notes
- **Smart Screenshot Extraction**: Automatically captures relevant screenshots at key moments with sharpness detection
- **Multiple Transcription Methods**: 
  - YouTube auto-generated subtitles (VTT)
  - Audio transcription with word-level timestamps for precise screenshot placement
- **Rich Markdown Output**: Generated notes with embedded screenshots in an elegant, readable format
- **PDF Export**: Automatically converts notes to beautifully formatted PDF documents
- **Notion Integration**: Optional upload to your Notion workspace

### Advanced Options

#### Screenshot Density
Control the number of screenshots captured per section:
- **Low**: 3-5 screenshots per section
- **Medium**: 8-12 screenshots per section (default)
- **High**: 15-20 screenshots per section

#### Note Detail Level
Choose the depth and verbosity of generated notes:
- **Brief**: High-level summaries and key points
- **Standard**: Balanced coverage with essential details (default)
- **Detailed**: Comprehensive, textbook-style explanations

#### Smart Frame Extraction
Analyzes multiple frames around each timestamp to select the sharpest, clearest image. While slower, it significantly improves screenshot quality.

#### Audio Transcription
Extracts and transcribes audio for word-level timestamps, enabling precise inline screenshot placement within the content. More accurate than VTT subtitles but takes longer to process.

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd youtube_notion_agent
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt  # Note: You may need to create requirements.txt
```

### Running the Application

#### Streamlit Web UI (Recommended)
```bash
streamlit run app.py
```
The UI will open in your browser at `http://localhost:8501`

#### Command Line Interface
```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

## 🎯 How to Use

### Web Interface

1. **Enter YouTube URL**: Paste any YouTube video URL in the input field
2. **Configure Options** (Sidebar):
   - Toggle smart frame extraction for better quality screenshots
   - Enable audio transcription for precise screenshot placement
   - Adjust screenshot density (Low/Medium/High)
   - Set note detail level (Brief/Standard/Detailed)
   - Enable Notion upload (requires configuration)
3. **Process Video**: Click "Create Notes" and wait for processing to complete
4. **View Results**: Browse generated notes with embedded images
5. **Download PDF**: Get a formatted PDF version of your notes
6. **Access Previous Notes**: Load previously processed videos from the sidebar

### What Happens During Processing

1. **Fetching video info** - Downloads video metadata and content
2. **Parsing transcript** - Extracts text from subtitles or transcribes audio
3. **Analyzing content with AI** - Generates structured notes and identifies key moments
4. **Extracting screenshots** - Captures high-quality screenshots at identified timestamps
5. **Generating markdown notes** - Creates formatted document with embedded images
6. **Generating PDF** - Converts markdown to a beautifully formatted PDF

## 📁 Output Structure

All processed videos are organized in dedicated folders:

```
output/
  └── Video_Title/
      ├── Video_Title.md           # Markdown notes with embedded images
      ├── Video_Title.pdf          # PDF version of notes
      ├── videoId_maxres.jpg       # Video thumbnail
      ├── videoId_screenshot_0001.jpg  # Screenshots
      ├── videoId_screenshot_0002.jpg
      └── ...
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Required for AI analysis
GOOGLE_API_KEY=your_google_api_key_here

# Optional: For Notion integration
NOTION_API_KEY=your_notion_api_key
NOTION_PAGE_ID=your_parent_page_id
```

### Notion Integration Setup

1. Create a Notion integration at [notion.so/my-integrations](https://www.notion.so/my-integrations)
2. Copy the Internal Integration Token to `NOTION_API_KEY`
3. Share a Notion page with your integration
4. Copy the page ID from the URL to `NOTION_PAGE_ID`
5. Enable "Upload to Notion" in the sidebar when processing videos

## 🎨 UI Design

The Streamlit interface features a clean, editorial-inspired design:
- Custom typography using DM Sans and Newsreader fonts
- Elegant image layouts with mosaic arrangements for multiple screenshots
- Responsive design optimized for readability
- Minimalist controls aligned with modern web standards

## 💡 Tips

- **First run may take longer** as the AI analyzes the content thoroughly
- **Screenshots are automatically optimized** for sharpness and relevance
- **Video files are deleted after processing** to save disk space (notes and screenshots are retained)
- **Use audio transcription** for technical videos where precise timing matters
- **Adjust screenshot density** based on video content (higher for tutorials, lower for talks)
- **All previous notes are preserved** and accessible via the sidebar

## 🛠️ Tech Stack

- **Streamlit**: Web interface
- **Google Gemini AI**: Content analysis and note generation
- **OpenCV**: Video processing and frame extraction
- **yt-dlp**: YouTube video and subtitle download
- **Whisper** (optional): Audio transcription
- **Notion SDK**: Notion integration
- **WeasyPrint**: PDF generation

## 📝 Project Structure

```
youtube_notion_agent/
├── app.py                    # Streamlit web interface
├── main.py                   # CLI interface
├── modules/
│   ├── downloader.py         # YouTube download logic
│   ├── frame_extractor.py    # Screenshot extraction
│   ├── content_analyzer.py   # AI analysis with Gemini
│   ├── audio_extractor.py    # Audio extraction from video
│   ├── transcriber.py        # Audio transcription
│   ├── notion_client.py      # Notion API integration
│   └── pdf_exporter.py       # PDF generation
├── output/                   # Processed videos folder
└── Assests/                  # Application assets

```

## 🔄 Version History

Current enhancements and improvements are tracked in `Notes.todo`.

## 📄 License

[Add your license information here]
