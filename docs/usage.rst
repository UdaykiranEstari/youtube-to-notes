Usage
=====

Web UI
------

Launch the Streamlit interface:

.. code-block:: bash

   streamlit run app.py

1. **Enter a YouTube URL** in the input field, or upload a local video file.
2. **Configure options** in the sidebar (see below).
3. Click **Create Notes** and wait for processing to complete.
4. Browse the generated notes with embedded screenshots.
5. Download the PDF or access previously processed videos from the sidebar.

CLI
---

.. code-block:: bash

   python main.py "https://www.youtube.com/watch?v=VIDEO_ID"

The CLI processes the video and writes output to the ``output/`` directory.

Configuration Options
---------------------

Screenshot Density
^^^^^^^^^^^^^^^^^^

Controls the number of screenshots captured per section:

- **Low** — 3--5 screenshots per section
- **Medium** — 8--12 screenshots per section (default)
- **High** — 15--20 screenshots per section

Note Detail Level
^^^^^^^^^^^^^^^^^

Controls the depth and verbosity of generated notes:

- **Brief** — High-level summaries and key points
- **Standard** — Balanced coverage with essential details (default)
- **Detailed** — Comprehensive, textbook-style explanations

Smart Frame Extraction
^^^^^^^^^^^^^^^^^^^^^^

When enabled, the extractor analyzes multiple frames around each timestamp and
selects the sharpest image. This improves screenshot quality at the cost of
longer processing time.

Audio Transcription
^^^^^^^^^^^^^^^^^^^

Extracts and transcribes audio for word-level timestamps, enabling precise
inline screenshot placement. More accurate than VTT subtitles but slower.

LLM Providers
^^^^^^^^^^^^^

Select your preferred AI provider in the sidebar. Supported providers:

- **Vertex AI** (Gemini) — default, requires GCP authentication
- **Google AI** (Gemini) — uses ``GOOGLE_API_KEY``
- **OpenAI** — ``gpt-4o``
- **Anthropic** — ``claude-3-5-sonnet-latest``

Local Video Upload
^^^^^^^^^^^^^^^^^^

Instead of a YouTube URL, you can upload a local video file directly through
the Web UI. The pipeline processes local files using the same analysis and
screenshot extraction steps.

Output Structure
----------------

All processed videos produce a dedicated folder under ``output/``:

.. code-block:: text

   output/
     └── Video_Title/
         ├── Video_Title.md           # Final merged markdown
         ├── Video_Title.pdf          # PDF export
         ├── chunk_001_notes.md       # Individual chunk notes
         ├── chunk_001_*.jpg          # Chunk screenshots
         └── videoId_maxres.jpg       # Thumbnail
