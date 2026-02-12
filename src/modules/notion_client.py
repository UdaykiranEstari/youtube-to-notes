"""Notion API integration for uploading generated notes.

Creates Notion pages with text blocks and externally-hosted images.  Images
are uploaded to Imgur first, then embedded via external URL because the
Notion API does not support direct file uploads to page content.
"""

import os
import requests
from notion_client import Client
from typing import List, Dict, Any, Optional

class NotionClient:
    """Client for creating Notion pages from generated video notes.

    Reads ``NOTION_API_KEY`` and ``NOTION_PAGE_ID`` from the environment.
    Images are first uploaded to Imgur (anonymous) because the Notion API
    does not support inline file uploads, then embedded via external URL.

    Attributes:
        client: Authenticated ``notion_client.Client`` instance, or *None*
            if the API key is missing.
    """

    def __init__(self):
        self.api_key = os.getenv("NOTION_API_KEY")
        self.parent_page_id = os.getenv("NOTION_PAGE_ID")
        # Client ID for Imgur (Anonymous upload)
        # Using a public/shared client ID or environment variable
        self.imgur_client_id = os.getenv("IMGUR_CLIENT_ID", "d3d252cb7ec55ff") # Default generic ID or use env
        
        if not self.api_key:
            print("Warning: NOTION_API_KEY not found in environment variables. Notion export will fail.")
        if not self.parent_page_id:
            print("Warning: NOTION_PAGE_ID not found in environment variables. Notion export will fail.")
            
        if self.api_key:
            self.client = Client(auth=self.api_key)
        else:
            self.client = None

    def create_page(self, title: str, content_blocks: List[Dict[str, Any]]) -> str:
        """Create a child page under the configured parent page.

        Handles Notion's 100-block-per-request limit by batching
        automatically.

        Args:
            title: Page title.
            content_blocks: List of Notion block dicts (see
                :meth:`create_text_block` and :meth:`create_image_block`).

        Returns:
            URL of the newly created Notion page, or an empty string on
            failure.
        """
        if not self.client:
            return ""
            
        try:
            # Notion has a limit of 100 children per request. 
            # We need to batch if there are more.
            
            # 1. Create page with first batch (or just title if blocks are many)
            initial_blocks = content_blocks[:100]
            remaining_blocks = content_blocks[100:]
            
            new_page = self.client.pages.create(
                parent={"page_id": self.parent_page_id},
                properties={
                    "title": {
                        "title": [
                            {
                                "text": {
                                    "content": title
                                }
                            }
                        ]
                    }
                },
                children=initial_blocks
            )
            
            page_id = new_page["id"]
            
            # 2. Append remaining blocks in batches of 100
            while remaining_blocks:
                batch = remaining_blocks[:100]
                remaining_blocks = remaining_blocks[100:]
                
                self.client.blocks.children.append(
                    block_id=page_id,
                    children=batch
                )
                
            return new_page["url"]
        except Exception as e:
            print(f"Error creating Notion page: {e}")
            return ""

    def create_text_block(self, content: str, style: str = "paragraph") -> Dict[str, Any]:
        """Build a Notion rich-text block dict.

        Content longer than 2 000 characters is automatically truncated to
        stay within the Notion API limit.

        Args:
            content: Plain text content for the block.
            style: Block type — ``"paragraph"``, ``"heading_1"``,
                ``"heading_2"``, ``"heading_3"``,
                ``"bulleted_list_item"``, ``"callout"``, or ``"quote"``.

        Returns:
            Notion block dict ready for inclusion in :meth:`create_page`.
        """
        # Truncate content if it exceeds Notion's limit (2000 chars)
        if len(content) > 2000:
            content = content[:1997] + "..."
            
        block = {
            "object": "block",
            "type": style,
            style: {
                "rich_text": [
                    {
                        "type": "text",
                        "text": {
                            "content": content
                        }
                    }
                ]
            }
        }
        
        # Callout needs an icon
        if style == "callout":
            block[style]["icon"] = {"emoji": "💡"}
            
        return block

    def create_image_block(self, image_url: str) -> Dict[str, Any]:
        """Build a Notion external-image block dict.

        Args:
            image_url: Publicly accessible URL of the image.

        Returns:
            Notion image block dict, or *None* if *image_url* is falsy.
        """
        if not image_url:
            return None
            
        return {
            "object": "block",
            "type": "image",
            "image": {
                "type": "external",
                "external": {
                    "url": image_url
                }
            }
        }
        
    def upload_image_to_imgur(self, image_path: str) -> Optional[str]:
        """Upload an image to Imgur (anonymous) and return its public URL.

        Args:
            image_path: Local path to the image file.

        Returns:
            Public Imgur URL string, or *None* on failure.
        """
        if not os.path.exists(image_path):
            return None
            
        headers = {'Authorization': f'Client-ID {self.imgur_client_id}'}
        
        try:
            with open(image_path, 'rb') as img:
                response = requests.post(
                    'https://api.imgur.com/3/image',
                    headers=headers,
                    files={'image': img}
                )
                
            if response.status_code == 200:
                data = response.json()
                return data['data']['link']
            else:
                print(f"Imgur upload failed: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"Error uploading to Imgur: {e}")
            return None

if __name__ == "__main__":
    pass
