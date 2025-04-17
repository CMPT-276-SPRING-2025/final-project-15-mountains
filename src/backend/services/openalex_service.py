import requests
import time
import json
import logging

logger = logging.getLogger(__name__)

# --- Modified OpenAlex Service ---
class OpenAlexService:
    BASE_URL = "https://api.openalex.org"

    def __init__(self, config, email=None):
        self.config = config
        self.email = email or self.config.get('OPENALEX_EMAIL')
        # Ensure email is provided for User-Agent politeness
        if not self.email:
            logger.warning("OPENALEX_EMAIL not set, using default. Please set it for polite API usage.")
            self.email = 'rba137@sfu.ca' # Default if not set
        self.headers = {'User-Agent': f'Factify/1.0 (mailto:{self.email})'}
        self.timeout = 20 # Increased timeout for larger requests

    def _reconstruct_abstract_from_inverted_index(self, abstract_inverted_index):
        # (Keep the existing implementation - unchanged)
        if not abstract_inverted_index:
            return ""
        max_position = 0
        for positions in abstract_inverted_index.values():
            if positions and max(positions) > max_position:
                max_position = max(positions)
        words = [""] * (max_position + 1)
        for word, positions in abstract_inverted_index.items():
            for position in positions:
                words[position] = word
        return " ".join(words)

    def search_works_by_keyword(self, keywords, per_page=10, page=1):
        """Search works using keyword search with pagination support."""
        # Join keywords for search query if it's a list
        search_query = keywords if isinstance(keywords, str) else " ".join(keywords)

        # Use the configured max results per source. Be mindful of potential API limits.
        # OpenAlex official limit seems to be 200. Higher values might cause errors.
        per_page = min(per_page, 200) # Cap at 200 based on OpenAlex docs, was 100

        # Filter only for abstracts for now
        openalex_filter = 'has_abstract:true' # Removed cited_by_count filter

        params = {
            'search': search_query,
            'per-page': per_page,
            'page': page,  # Add page parameter for pagination
            'filter': openalex_filter, # Use the defined filter
            'select': 'id,doi,title,authorships,publication_date,abstract_inverted_index,primary_location,cited_by_count' # Add cited_by_count
        }
        logger.info(f"Querying OpenAlex: {search_query} with per_page={per_page}, page={page}, filter='{openalex_filter}'")
        try:
            response = requests.get(
                f"{self.BASE_URL}/works",
                params=params,
                headers=self.headers,
                timeout=self.timeout
            )

            # Handle rate limiting
            if response.status_code == 429:
                logger.warning("OpenAlex rate limit hit, sleeping for 2 seconds.")
                time.sleep(2)
                return self.search_works_by_keyword(keywords, per_page, page) # Retry

            # Handle forbidden errors
            if response.status_code == 403:
                logger.warning("OpenAlex returned 403 Forbidden. Trying with smaller batch size.")
                if per_page > 25:
                    # Retry with a much smaller batch size
                    return self.search_works_by_keyword(keywords, 25, page)
                else:
                    # If we're already using a small batch size, it's some other issue
                    logger.error(f"OpenAlex 403 Forbidden error even with small batch size: {response.text}")
                    return None

            response.raise_for_status() # Raise HTTP errors
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenAlex API request failed: {e}")
            return None
        except json.JSONDecodeError as e:
             logger.error(f"Failed to decode OpenAlex JSON response: {e}")
             return None

    def process_results(self, results_json):
        """Processes OpenAlex JSON results into a standardized format."""
        processed = []
        if not results_json or 'results' not in results_json:
            return processed

        for paper in results_json.get('results', []):
            # --- Check for Retraction ---
            # OpenAlex might use `is_retracted` or status fields, adjust based on observed data
            # Assuming a field like `is_retracted` or `publication_status` exists
            if paper.get('is_retracted', False) or paper.get('publication_status', '').lower() == 'retracted':
                 doi = paper.get('doi', 'Unknown DOI')
                 logger.warning(f"Skipping retracted OpenAlex article: DOI {doi}")
                 continue # Skip this article
            # --- End Check for Retraction ---

            # Extract abstract
            abstract = ""
            if paper.get('abstract_inverted_index'):
                try:
                    abstract = self._reconstruct_abstract_from_inverted_index(paper.get('abstract_inverted_index'))
                except Exception as e:
                    logger.warning(f"Error reconstructing abstract for OpenAlex ID {paper.get('id')}: {e}")

            # Skip if abstract is empty or too short
            if not abstract or len(abstract) < 50:
                continue

            # Extract authors
            authors = ", ".join([a.get('author', {}).get('display_name', '') for a in paper.get('authorships', []) if a.get('author')])

            # Extract citation count
            citation_count = paper.get('cited_by_count', 0)

            # Accept all studies with abstracts, regardless of citation count
            processed.append({
                "doi": paper.get('doi'),
                "title": paper.get('title', 'Untitled'),
                "authors": authors,
                "pub_date": paper.get('publication_date'),
                "abstract": abstract,
                "source_api": "openalex",
                "citation_count": citation_count
            })
        return processed
# --- End OpenAlex Service --- 