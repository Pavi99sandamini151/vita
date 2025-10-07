from atlassian import Confluence
import os
from typing import List, Dict

class ConfluenceService:
    def __init__(self):
        self.confluence = Confluence(
            url=os.getenv('CONFLUENCE_URL'),
            username=os.getenv('CONFLUENCE_USERNAME'),
            password=os.getenv('CONFLUENCE_API_TOKEN')
        )
        self.space_key = os.getenv('CONFLUENCE_SPACE_KEY')

    def get_space_content(self) -> List[Dict]:
        pages = []
        start = 0
        limit = 100
        
        while True:
            results = self.confluence.get_all_pages_from_space(
                self.space_key,
                start=start,
                limit=limit,
                expand='body.storage'
            )
            
            if not results:
                break
                
            pages.extend([{
                'id': page['id'],
                'title': page['title'],
                'content': page['body']['storage']['value']
            } for page in results])
            
            start += limit
            
        return pages