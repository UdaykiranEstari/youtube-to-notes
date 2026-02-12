"""Markdown-to-PDF conversion with image embedding.

Converts a markdown notes file into a styled A3 PDF using
`xhtml2pdf <https://github.com/xhtml2pdf/xhtml2pdf>`_.  Consecutive images
are automatically arranged in a two-column layout and alt-text captions are
rendered below each image.
"""

import markdown
from xhtml2pdf import pisa
import os

class PDFExporter:
    """Generates styled PDF documents from markdown notes.

    Images referenced in the markdown are resolved to absolute paths and
    embedded.  Consecutive images are arranged in a two-column table layout
    and alt-text captions are rendered beneath each image.
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

        # Process images to create 2-column layout
        html_content = self._process_html_images(html_content)

        # Add visible captions below images
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

    def _process_html_images(self, html_content: str) -> str:
        """Group consecutive image paragraphs into a two-column table layout.

        Args:
            html_content: HTML string to process.

        Returns:
            Modified HTML with paired images wrapped in a ``<table>``.
        """
        import re
        
        pattern = r'(<p>\s*<img[^>]+>\s*</p>)\s*(<p>\s*<img[^>]+>\s*</p>)'

        def replace_pair(match):
            img1_p = match.group(1)
            img2_p = match.group(2)
            
            try:
                img1 = re.search(r'<img[^>]+>', img1_p).group(0)
                img2 = re.search(r'<img[^>]+>', img2_p).group(0)
                
                return f"""
                <table style="width: 100%; border: none; margin-bottom: 20px;">
                  <tr>
                    <td style="width: 47%; padding-right: 0; vertical-align: top; border: none;">{img1}</td>
                    <td style="width: 6%; border: none;"></td> <!-- Spacer -->
                    <td style="width: 47%; padding-left: 0; vertical-align: top; border: none;">{img2}</td>
                  </tr>
                </table>
                """
            except:
                return match.group(0)

        return re.sub(pattern, replace_pair, html_content)

    def _add_image_captions(self, html_content: str) -> str:
        """Add visible captions below images using their alt text.

        Args:
            html_content: HTML string to process.

        Returns:
            HTML with ``<p class="caption">`` elements inserted after each
            ``<img>`` that has non-empty alt text.
        """
        import re
        
        def add_caption(match):
            img_tag = match.group(0)
            alt_match = re.search(r'alt="([^"]+)"', img_tag)
            
            if alt_match and alt_match.group(1).strip():
                caption = alt_match.group(1)
                # Unescape HTML entities in caption
                caption = caption.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
                
                # Return img + caption paragraph with centered, smaller text
                return f'''{img_tag}
<p class="caption" style="text-align: center; font-size: 9pt; color: #666; margin-top: 5px; margin-bottom: 15px; font-style: italic;">{caption}</p>'''
            
            return img_tag
        
        # Find all img tags and add captions
        return re.sub(r'<img[^>]+>', add_caption, html_content)


    def _add_styling(self, html_content: str, base_path: str) -> str:
        """Wrap HTML content in a full document with CSS and absolute image paths.

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

        import re
        html_content = re.sub(r'src="([^"]+)"', replace_path, html_content)

        css = """
        <style>
            @page {
                size: A3;
                margin: 2cm;
            }
            body {
                font-family: Helvetica, sans-serif;
                font-size: 12pt;
                line-height: 1.5;
                color: #2B2B2B;
            }
            h1 {
                font-family: "Times New Roman", serif;
                font-size: 26pt;
                font-weight: bold;
                color: #000000;
                border-bottom: 1px solid #ddd;
                padding-bottom: 15px;
                margin-top: 0;
                margin-bottom: 20px;
            }
            h2 {
                font-family: "Times New Roman", serif;
                font-size: 20pt;
                font-weight: bold;
                color: #000000;
                margin-top: 30px;
                margin-bottom: 15px;
            }
            h3 {
                font-family: "Times New Roman", serif;
                font-size: 16pt;
                font-weight: bold;
                color: #444;
                margin-top: 20px;
                margin-bottom: 10px;
            }
            p {
                margin-bottom: 12px;
                text-align: justify;
            }
            img {
                max-width: 100%;
                height: auto;
                margin: 0;
                border-radius: 2px;
            }
            /* Table styling for image grid */
            table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }
            td {
                vertical-align: top;
                padding: 0;
            }
            
            code {
                background-color: #f5f5f5;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: Courier, monospace;
                font-size: 10pt;
            }
            pre {
                background-color: #f5f5f5;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
                font-family: Courier, monospace;
                font-size: 9pt;
                margin: 20px 0;
            }
            blockquote {
                border-left: 3px solid #ccc;
                padding-left: 15px;
                color: #666;
                font-style: italic;
                margin: 20px 0;
            }
            strong {
                color: #000;
                font-weight: bold;
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
