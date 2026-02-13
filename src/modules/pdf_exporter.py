"""Markdown-to-PDF conversion with image embedding.

Converts a markdown notes file into a styled full-width PDF using
`xhtml2pdf <https://github.com/xhtml2pdf/xhtml2pdf>`_.  Images are rendered
full-width with captions below, following an editorial magazine layout
inspired by NeverTooSmall.
"""

import re
import markdown
from xhtml2pdf import pisa
import os

# Pre-compiled regex patterns for PDF image processing
_IMG_TAG_RE = re.compile(r'<img[^>]+>')
_IMG_ALT_RE = re.compile(r'alt="([^"]+)"')
_IMG_SRC_RE = re.compile(r'src="([^"]+)"')

class PDFExporter:
    """Generates styled PDF documents from markdown notes.

    Images are rendered full-width with captions beneath, using Inter as the
    sole typeface to match an editorial magazine aesthetic.
    """

    def __init__(self):
        pass

    def convert_markdown_to_pdf(self, markdown_path: str, output_pdf_path: str):
        """Convert a markdown file to a styled PDF with embedded images.

        Args:
            markdown_path: Path to the source ``.md`` file.
            output_pdf_path: Destination path for the generated PDF.

        Returns:
            The *output_pdf_path* on success.

        Raises:
            FileNotFoundError: If *markdown_path* does not exist.
            Exception: If xhtml2pdf reports an error during generation.
        """
        if not os.path.exists(markdown_path):
            raise FileNotFoundError(f"Markdown file not found: {markdown_path}")

        with open(markdown_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        # Convert Markdown to HTML
        html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])

        # Add visible captions below full-width images
        html_content = self._add_image_captions(html_content)

        # Add basic styling
        styled_html = self._add_styling(html_content, os.path.dirname(markdown_path))

        # Generate PDF
        with open(output_pdf_path, "wb") as pdf_file:
            pisa_status = pisa.CreatePDF(
                styled_html,
                dest=pdf_file,
                encoding='utf-8'
            )

        if pisa_status.err:
            raise Exception(f"PDF generation failed: {pisa_status.err}")

        return output_pdf_path

    def _add_image_captions(self, html_content: str) -> str:
        """Add visible captions below full-width images using their alt text.

        Args:
            html_content: HTML string to process.

        Returns:
            HTML with ``<p class="caption">`` elements inserted after each
            ``<img>`` that has non-empty alt text.
        """
        def add_caption(match):
            img_tag = match.group(0)
            alt_match = _IMG_ALT_RE.search(img_tag)

            if alt_match and alt_match.group(1).strip():
                caption = alt_match.group(1)
                caption = caption.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')

                return f'''{img_tag}
<p class="caption">{caption}</p>'''

            return img_tag

        return _IMG_TAG_RE.sub(add_caption, html_content)

    def _add_styling(self, html_content: str, base_path: str) -> str:
        """Wrap HTML content in a full document with CSS and absolute image paths.

        Uses a full-width page layout with Inter as the sole typeface,
        matching the NeverTooSmall editorial aesthetic: consistent font
        across all headings and body, generous whitespace, and full-width
        images.

        Args:
            html_content: Inner HTML body content.
            base_path: Directory used to resolve relative image ``src``
                attributes to absolute paths.

        Returns:
            Complete ``<!DOCTYPE html>`` document string ready for PDF
            rendering.
        """
        base_path = os.path.abspath(base_path)

        def replace_path(match):
            src = match.group(1)
            if src.startswith(('http://', 'https://', 'file://', '/')):
                return f'src="{src}"'

            abs_path = os.path.join(base_path, src)
            return f'src="{abs_path}"'

        html_content = _IMG_SRC_RE.sub(replace_path, html_content)

        css = """
        <style>
            @page {
                size: 11in 17in;
                margin: 1.5cm 2cm;
            }
            body {
                font-family: Inter, "Helvetica Neue", Helvetica, Arial, sans-serif;
                font-size: 11pt;
                line-height: 1.4;
                color: #2B2B2B;
                -webkit-font-smoothing: antialiased;
            }

            /* --- Headings: same font family, differentiated by size/weight --- */
            h1 {
                font-family: Inter, "Helvetica Neue", Helvetica, Arial, sans-serif;
                font-size: 24pt;
                font-weight: 600;
                color: #1a1a1a;
                margin-top: 0;
                margin-bottom: 24px;
                padding-bottom: 16px;
                border-bottom: 1px solid #e0e0e0;
                letter-spacing: -0.02em;
            }
            h2 {
                font-family: Inter, "Helvetica Neue", Helvetica, Arial, sans-serif;
                font-size: 18pt;
                font-weight: 500;
                color: #1a1a1a;
                margin-top: 48px;
                margin-bottom: 16px;
                letter-spacing: -0.01em;
            }
            h3 {
                font-family: Inter, "Helvetica Neue", Helvetica, Arial, sans-serif;
                font-size: 14pt;
                font-weight: 500;
                color: #333;
                margin-top: 32px;
                margin-bottom: 12px;
            }

            /* --- Body text --- */
            p {
                margin-bottom: 1.4em;
                text-align: left;
            }

            /* --- Full-width images with generous spacing --- */
            img {
                width: 100%;
                height: auto;
                margin: 48px 0 8px 0;
                border-radius: 0;
            }

            /* --- Captions: smaller, muted --- */
            p.caption {
                font-family: Inter, "Helvetica Neue", Helvetica, Arial, sans-serif;
                font-size: 9pt;
                line-height: 1.4;
                color: #888;
                text-align: left;
                margin-top: 4px;
                margin-bottom: 48px;
                font-weight: 400;
            }

            /* --- Tables --- */
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 24px 0;
            }
            td, th {
                vertical-align: top;
                padding: 6px 8px;
                border-bottom: 1px solid #eee;
                font-size: 10pt;
            }

            /* --- Code --- */
            code {
                background-color: #f5f5f5;
                padding: 2px 5px;
                border-radius: 3px;
                font-family: "SF Mono", Menlo, Courier, monospace;
                font-size: 9pt;
            }
            pre {
                background-color: #f5f5f5;
                padding: 16px;
                border-radius: 4px;
                overflow-x: auto;
                font-family: "SF Mono", Menlo, Courier, monospace;
                font-size: 9pt;
                margin: 24px 0;
                line-height: 1.5;
            }

            /* --- Blockquotes --- */
            blockquote {
                border-left: 2px solid #ccc;
                padding-left: 16px;
                color: #555;
                font-style: normal;
                margin: 24px 0;
            }

            strong {
                color: #1a1a1a;
                font-weight: 600;
            }

            /* --- Source link --- */
            a {
                color: #2B2B2B;
                text-decoration: underline;
            }
        </style>
        """

        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            {css}
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """

        return full_html
