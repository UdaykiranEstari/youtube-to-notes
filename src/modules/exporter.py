"""
Multi-format exporter for video notes.
Exports Markdown to HTML, TXT, and DOCX formats.
"""
import os
import re
from typing import Optional

class Exporter:
    """Multi-format exporter for markdown notes.

    Supports conversion to HTML, plain text, and DOCX.

    Args:
        output_dir: Directory where exported files are written.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
    
    def export_to_html(self, md_path: str) -> Optional[str]:
        """Convert markdown to a self-contained styled HTML file.

        Args:
            md_path: Path to the source markdown file.

        Returns:
            Path to the generated ``.html`` file, or *None* if the
            ``markdown`` package is not installed.
        """
        try:
            import markdown
        except ImportError:
            print("markdown package not installed. Run: pip install markdown")
            return None
        
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()
        
        # Convert markdown to HTML
        html_body = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
        
        # Wrap in styled HTML document
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Notes</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 2rem;
            line-height: 1.6;
            color: #333;
        }}
        h1 {{ color: #2B2B2B; border-bottom: 2px solid #eee; padding-bottom: 0.5rem; }}
        h2 {{ color: #444; margin-top: 2rem; }}
        img {{ max-width: 100%; height: auto; border-radius: 8px; margin: 1rem 0; }}
        a {{ color: #0066cc; }}
        code {{ background: #f5f5f5; padding: 2px 6px; border-radius: 3px; }}
        blockquote {{ border-left: 4px solid #ddd; margin: 1rem 0; padding-left: 1rem; color: #666; }}
    </style>
</head>
<body>
{html_body}
</body>
</html>"""
        
        # Save HTML
        base_name = os.path.splitext(os.path.basename(md_path))[0]
        html_path = os.path.join(self.output_dir, f"{base_name}.html")
        
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        return html_path
    
    def export_to_txt(self, md_path: str) -> Optional[str]:
        """Convert markdown to plain text, stripping images and formatting.

        Args:
            md_path: Path to the source markdown file.

        Returns:
            Path to the generated ``.txt`` file.
        """
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()
        
        # Remove images
        txt_content = re.sub(r'!\[.*?\]\(.*?\)', '', md_content)
        
        # Remove links but keep text
        txt_content = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', txt_content)
        
        # Remove markdown formatting
        txt_content = re.sub(r'^#+\s*', '', txt_content, flags=re.MULTILINE)  # Headers
        txt_content = re.sub(r'\*\*(.*?)\*\*', r'\1', txt_content)  # Bold
        txt_content = re.sub(r'\*(.*?)\*', r'\1', txt_content)  # Italic
        txt_content = re.sub(r'`(.*?)`', r'\1', txt_content)  # Code
        
        # Clean up extra whitespace
        txt_content = re.sub(r'\n{3,}', '\n\n', txt_content)
        
        # Save TXT
        base_name = os.path.splitext(os.path.basename(md_path))[0]
        txt_path = os.path.join(self.output_dir, f"{base_name}.txt")
        
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(txt_content.strip())
        
        return txt_path
    
    def export_to_docx(self, md_path: str) -> Optional[str]:
        """Convert markdown to DOCX using pypandoc.

        Args:
            md_path: Path to the source markdown file.

        Returns:
            Path to the generated ``.docx`` file, or *None* if pypandoc is
            not installed or conversion fails.
        """
        try:
            import pypandoc
        except ImportError:
            print("pypandoc not installed. Run: pip install pypandoc")
            return None
        
        base_name = os.path.splitext(os.path.basename(md_path))[0]
        docx_path = os.path.join(self.output_dir, f"{base_name}.docx")
        
        try:
            pypandoc.convert_file(md_path, 'docx', outputfile=docx_path)
            return docx_path
        except Exception as e:
            print(f"DOCX export failed: {e}")
            return None
