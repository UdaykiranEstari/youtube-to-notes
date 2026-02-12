import os
import sys

# Add project root to path so autodoc can find src/
sys.path.insert(0, os.path.abspath(".."))

project = "YouTube to Notes"
copyright = "2024, YouTube to Notes Contributors"
author = "YouTube to Notes Contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

# Napoleon settings (Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_mock_imports = [
    "streamlit",
    "yt_dlp",
    "whisper",
    "google",
    "vertexai",
    "openai",
    "anthropic",
    "cv2",
    "notion_client",
    "ffmpeg",
    "faster_whisper",
    "xhtml2pdf",
    "pypandoc",
    "markdown",
    "dotenv",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_logo = "../assets/maxresdefault.jpg"

html_theme_options = {
    "github_url": "https://github.com/your-org/youtube_notion_agent",
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "logo": {
        "text": "YouTube to Notes",
    },
}
