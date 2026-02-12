import markdown
import re

md_content = """
# Header

Text paragraph.

![Image 1](img1.jpg)

![Image 2](img2.jpg)

Text in between.

![Image 3](img3.jpg)
![Image 4](img4.jpg)
![Image 5](img5.jpg)

End.
"""

html = markdown.markdown(md_content)
print("--- Original HTML ---")
print(html)

# Regex to find two consecutive paragraphs that contain only an image
# Pattern explanation:
# <p>\s*<img[^>]+>\s*</p>  matches a paragraph with a single image
# \s*                      matches whitespace (newlines) between paragraphs
# (...)                    groups for replacement
pattern = r'(<p>\s*<img[^>]+>\s*</p>)\s*(<p>\s*<img[^>]+>\s*</p>)'

def replace_pair(match):
    img1_p = match.group(1)
    img2_p = match.group(2)
    
    # Extract img tags
    img1 = re.search(r'<img[^>]+>', img1_p).group(0)
    img2 = re.search(r'<img[^>]+>', img2_p).group(0)
    
    return f"""
<table style="width: 100%; border: none; margin-bottom: 20px;">
  <tr>
    <td style="width: 48%; padding-right: 2%; vertical-align: top; border: none;">{img1}</td>
    <td style="width: 48%; padding-left: 2%; vertical-align: top; border: none;">{img2}</td>
  </tr>
</table>
"""

# Apply regex
new_html = re.sub(pattern, replace_pair, html)

print("\n--- Transformed HTML ---")
print(new_html)
