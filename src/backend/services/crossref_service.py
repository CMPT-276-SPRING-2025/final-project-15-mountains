import requests
import time
import json
import logging
import re

logger = logging.getLogger(__name__)

# --- New: CrossRef Service ---
class CrossRefService:
    BASE_URL = "https://api.crossref.org"

    def __init__(self, config, email=None):
        self.config = config
        self.email = email or self.config.get('OPENALEX_EMAIL') # Reuse email for politeness
        if not self.email:
            logger.warning("Email not set for CrossRef, using default. Set OPENALEX_EMAIL for polite API usage.")
            self.email = 'rba137@sfu.ca' # Default if not set
        self.headers = {'User-Agent': f'Factify/1.0 (mailto:{self.email})'}
        self.timeout = 30 # Increased timeout for potentially larger/slower requests

    def search_works_by_keyword(self, keywords, rows=10, offset=0):
        """Search CrossRef works using keyword query with pagination support."""
        # Join keywords for search query if it's a list
        search_query = keywords if isinstance(keywords, str) else " ".join(keywords)

        # Use the configured max results per source. CrossRef limit is 1000 but often unstable.
        # Capping at a safer limit like 200 might be wise, adjust if needed.
        rows = min(rows, 200) # Cap at 200, was 100. Can be increased further but test stability.

        params = {
            'query': search_query,
            'rows': rows,
            'offset': offset,  # Add offset parameter for pagination
            'select': 'DOI,title,author,abstract,published-print,published-online,created,is-referenced-by-count'  # Added citation count field
        }
        logger.info(f"Querying CrossRef using 'query' param: {search_query} with rows={rows}, offset={offset} (Abstracts filtered post-retrieval)")
        try:
            response = requests.get(
                f"{self.BASE_URL}/works",
                params=params,
                headers=self.headers,
                timeout=self.timeout
            )
            
            # Handle rate limiting or errors
            if response.status_code == 429:
                logger.warning("CrossRef rate limit hit, sleeping for 2 seconds.")
                time.sleep(2)
                return self.search_works_by_keyword(keywords, rows, offset)
                
            # Handle other error responses
            if response.status_code >= 400:
                logger.warning(f"CrossRef returned error {response.status_code}. Trying with smaller batch size.")
                if rows > 25:
                    # Retry with a smaller batch size
                    return self.search_works_by_keyword(keywords, 25, offset)
                else:
                    logger.error(f"CrossRef error {response.status_code} even with small batch size: {response.text}")
                    return None
                    
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"CrossRef API request failed: {e}")
            return None
        except json.JSONDecodeError as e:
             logger.error(f"Failed to decode CrossRef JSON response: {e}")
             return None

    def process_results(self, results_json):
        """Processes CrossRef JSON results into a standardized format."""
        processed = []
        if not results_json or 'message' not in results_json or 'items' not in results_json['message']:
            return processed

        for item in results_json['message'].get('items', []):
            # --- Filter for abstract existence AFTER retrieval ---
            abstract = item.get('abstract', '').strip()
            title_list = item.get('title', [])
            title = ". ".join(title_list) if title_list else 'Untitled'
            
            # --- NEW: Check for Retraction Keyword --- 
            # Look for explicit retraction markers in title or abstract
            if title.lower().strip().startswith('[retracted]') or \
               abstract.lower().strip().startswith('[retracted]'):
                doi = item.get('DOI', 'Unknown DOI')
                logger.warning(f"Skipping likely retracted CrossRef article based on keyword: DOI {doi}")
                continue # Skip this item
            # --- End Retraction Keyword Check ---
            
            # Clean JATS XML tags from abstract
            abstract = re.sub(r'</?jats:[^>]+>', '', abstract)
            abstract = re.sub(r'</?[^>]+>', '', abstract)
            
            # Skip if abstract is empty or too short after cleaning
            if not abstract or len(abstract) < 50:
                continue

            # Extract authors
            authors_list = []
            if item.get('author'):
                authors_list = [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in item['author']]
            authors = ", ".join(filter(None, authors_list)) # Join non-empty names

             # Extract publication date (can be complex)
            pub_date_parts = item.get('published-print', {}).get('date-parts', [[]])[0] or \
                             item.get('published-online', {}).get('date-parts', [[]])[0] or \
                             item.get('created', {}).get('date-parts', [[]])[0]
            pub_date = "-".join(map(str, pub_date_parts)) if pub_date_parts else None

            # Extract citation count
            citation_count = item.get('is-referenced-by-count', 0)

            # Accept all studies with abstracts, regardless of citation count
            processed.append({
                "doi": item.get('DOI'),
                "title": title,
                "authors": authors,
                "pub_date": pub_date,
                "abstract": abstract, # Use cleaned abstract
                "source_api": "crossref",
                "citation_count": citation_count
            })
        return processed
# --- End CrossRef Service --- 