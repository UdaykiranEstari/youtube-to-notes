Architecture
============

Pipeline Flow
-------------

The main processing pipeline is orchestrated by
:mod:`src.pipeline.processor`. Each video goes through the following stages:

1. **Video Download** — ``YouTubeDownloader`` fetches the video, subtitles, and
   thumbnail via yt-dlp.
2. **Chunking** — Long videos are split into parallel-processable chunks
   (default 30 minutes each).
3. **Transcription** — Either VTT subtitles or Whisper audio transcription
   provides the transcript.
4. **AI Analysis** — The selected LLM provider analyzes the transcript and
   generates structured notes with screenshot timestamps.
5. **Frame Extraction** — ``FrameExtractor`` captures screenshots at
   AI-specified timestamps using OpenCV.
6. **Markdown Generation** — Notes are assembled in an editorial format with
   inline images.
7. **Merging** — ``MarkdownMerger`` combines chunk outputs into a single
   document.
8. **Export** — ``PDFExporter`` generates a PDF; optionally ``NotionClient``
   uploads to Notion.

Parallel Chunk Processing
--------------------------

For videos longer than the chunk threshold (default 30 min), the pipeline:

- Splits the video into segments using ``VideoSplitter``
- Processes each chunk concurrently via ``ThreadPoolExecutor``
- Each chunk independently downloads its segment, runs AI analysis, and
  extracts screenshots
- Artifacts are moved to the main output folder with chunk prefixes
- ``MarkdownMerger`` combines all chunk markdown files into the final document

LLM Provider Abstraction
-------------------------

:mod:`src.modules.llm_providers` provides a unified interface across multiple
LLM backends. The ``create_provider()`` factory function returns a provider
instance for any supported backend:

- **Vertex AI** — GCP-authenticated Gemini (default)
- **Google AI** — API-key-based Gemini
- **OpenAI** — GPT-4o
- **Anthropic** — Claude 3.5 Sonnet

All providers expose the same ``generate()`` method, allowing the rest of the
pipeline to remain backend-agnostic.

Module Responsibilities
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Module
     - Purpose
   * - ``src.pipeline.processor``
     - Main orchestration, parallel chunk processing
   * - ``src.modules.content_analyzer``
     - LLM prompt engineering, JSON response parsing/repair
   * - ``src.modules.frame_extractor``
     - OpenCV screenshot extraction with optional sharpness detection
   * - ``src.modules.downloader``
     - yt-dlp wrapper for video/subtitle/thumbnail download
   * - ``src.modules.transcriber``
     - Whisper integration for audio transcription
   * - ``src.modules.audio_extractor``
     - Audio stream extraction from video files
   * - ``src.modules.video_splitter``
     - Splits long videos into chunks for parallel processing
   * - ``src.modules.markdown_merger``
     - Combines chunk markdown files into final document
   * - ``src.modules.exporter``
     - HTML/PDF export utilities
   * - ``src.modules.pdf_exporter``
     - PDF generation from markdown
   * - ``src.modules.llm_providers``
     - Unified LLM provider interface and factory
   * - ``src.modules.notion_client``
     - Notion API integration for uploading notes
   * - ``src.modules.progress_tracker``
     - Processing progress tracking for the UI
   * - ``src.utils.common``
     - Shared helpers (filename sanitization, timestamp parsing)
