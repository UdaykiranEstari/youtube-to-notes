Getting Started
===============

Prerequisites
-------------

- **Python 3.9+**
- **ffmpeg** — required for video/audio processing. Install via your package
  manager:

  .. code-block:: bash

     # macOS
     brew install ffmpeg

     # Ubuntu / Debian
     sudo apt install ffmpeg

     # Windows (via choco)
     choco install ffmpeg

Installation
------------

1. Clone the repository:

   .. code-block:: bash

      git clone <repository-url>
      cd youtube_notion_agent

2. Create and activate a virtual environment:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate   # Windows: venv\Scripts\activate

3. Install dependencies:

   .. code-block:: bash

      pip install -r requirements.txt

Environment Variables
---------------------

Create a ``.env`` file in the project root with the credentials for your chosen
LLM provider:

.. code-block:: bash

   # Vertex AI (default provider)
   GOOGLE_CLOUD_PROJECT=your_project_id
   GOOGLE_CLOUD_LOCATION=us-central1

   # Google AI (alternative)
   GOOGLE_API_KEY=your_key

   # OpenAI
   OPENAI_API_KEY=your_key

   # Anthropic
   ANTHROPIC_API_KEY=your_key

   # Optional: Notion integration
   NOTION_API_KEY=your_notion_api_key
   NOTION_PAGE_ID=your_parent_page_id

You only need to configure **one** LLM provider. Vertex AI is the default; set
``GOOGLE_API_KEY`` for the simplest setup using Google AI.

Notion Integration
^^^^^^^^^^^^^^^^^^

To enable Notion upload:

1. Create a Notion integration at https://www.notion.so/my-integrations
2. Copy the Internal Integration Token to ``NOTION_API_KEY``
3. Share a Notion page with your integration
4. Copy the page ID from the URL to ``NOTION_PAGE_ID``

Quick Start
-----------

**Streamlit Web UI** (recommended):

.. code-block:: bash

   streamlit run app.py

The UI opens in your browser at ``http://localhost:8501``.

**Command-line interface**:

.. code-block:: bash

   python main.py "https://www.youtube.com/watch?v=VIDEO_ID"
