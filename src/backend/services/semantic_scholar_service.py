import requests
import time
import json
import logging

logger = logging.getLogger(__name__)

# --- New: Semantic Scholar Service ---
class SemanticScholarService:
    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, config):
        self.config = config
        # Semantic Scholar doesn't strictly require an API key for basic search,

        self.headers = {'User-Agent': f'Factify/1.0 (mailto:{self.config.get("OPENALEX_EMAIL", "rba137@sfu.ca")})'} # Reuse email for politeness
        self.timeout = 30 # Increased timeout

    def search_works_by_keyword(self, keywords, limit=10, offset=0):
        """Search Semantic Scholar works using keyword query with pagination."""
        search_query = keywords if isinstance(keywords, str) else " ".join(keywords)
        # Use configured limit, but respect the API's 100-per-request limit
        max_per_page = 100
        # The limit parameter now represents the total limit for this function call
        # We'll still use max_per_page for individual API requests
        total_limit_requested = limit # Rename for clarity
        current_offset = offset # Use the provided offset
        all_results_data = []
        total_reported_by_api = None # Store the total number of results reported by the API
        retrieved_this_call = 0 # Track results retrieved in this specific call

        # Define the fields we want to retrieve - using correct field names
        fields = 'externalIds,title,authors,year,abstract,citationCount,publicationDate,journal'

        logger.info(f"Querying Semantic Scholar: '{search_query}' starting at offset={current_offset}, aiming for up to {total_limit_requested} results (in batches of {max_per_page})")

        # Loop to fetch pages until the limit for *this call* is reached or no more results
        while retrieved_this_call < total_limit_requested:
            # Determine how many to request in this batch
            request_limit = min(max_per_page, total_limit_requested - retrieved_this_call)
            if request_limit <= 0:
                 break # Should not happen, but safety check

            params = {
                'query': search_query,
                'limit': request_limit,
                'fields': fields,
                'offset': current_offset
            }
            logger.info(f"  - Requesting batch: limit={request_limit}, offset={current_offset}")

            try:
                response = requests.get(
                    f"{self.BASE_URL}/paper/search",
                    params=params,
                    headers=self.headers,
                    timeout=self.timeout
                )

                # Handle rate limiting (HTTP 429)
                if response.status_code == 429:
                    logger.warning("Semantic Scholar rate limit hit, sleeping for 5 seconds.") # Longer sleep
                    time.sleep(5)
                    continue # Retry the same request after waiting

                # Handle other potential errors
                if response.status_code >= 400:
                    logger.error(f"Semantic Scholar API request failed with status {response.status_code} at offset {current_offset}: {response.text}")
                    # Decide whether to stop or continue trying next pages
                    break # Stop pagination on error

                response.raise_for_status() # Raise HTTP errors for other codes (e.g., 5xx)
                page_data = response.json()

                # Check if 'data' exists and is a list
                if 'data' not in page_data or not isinstance(page_data['data'], list):
                    logger.warning(f"Semantic Scholar response missing 'data' list or is not a list at offset {current_offset}.")
                    break # Stop if data format is unexpected

                # Store the total reported by the API on the first request (if offset was 0)
                if current_offset == offset and total_reported_by_api is None: # Only log total on the very first page of a sequence
                    total_reported_by_api = page_data.get('total', 0)
                    logger.info(f"  - Semantic Scholar API reports {total_reported_by_api} total potential results for the query.")

                current_results = page_data['data']
                all_results_data.extend(current_results)
                num_in_page = len(current_results)
                retrieved_this_call += num_in_page
                current_offset += num_in_page # Increment offset for the next potential loop iteration

                logger.info(f"  - Retrieved {num_in_page} results in this batch. Total retrieved this call: {retrieved_this_call}")

                # Stop if the API returns fewer results than requested (means we reached the end for this query)
                if num_in_page < request_limit:
                    logger.info(f"  - Reached end of Semantic Scholar results (requested {request_limit}, got {num_in_page}).")
                    break
                # Stop if API reports a total and we have fetched at least that many starting from the *initial* offset
                if total_reported_by_api is not None and (current_offset - offset) >= total_reported_by_api:
                    logger.info(f"  - Fetched {current_offset - offset} results, meeting or exceeding API reported total ({total_reported_by_api}).")
                    break

            except requests.exceptions.Timeout:
                logger.error(f"Semantic Scholar API request timed out at offset {current_offset}. Stopping pagination.")
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"Semantic Scholar API request failed at offset {current_offset}: {e}. Stopping pagination.")
                break
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode Semantic Scholar JSON response at offset {current_offset}: {e}. Stopping pagination.")
                break
            except Exception as e:
                logger.error(f"Unexpected error during Semantic Scholar pagination at offset {current_offset}: {e}")
                break # Stop on unexpected errors

        logger.info(f"Finished Semantic Scholar batch call (started at offset {offset}). Total results collected: {len(all_results_data)}")
        # Return the aggregated results in the expected format for process_results
        # The 'total' key here might be less meaningful as it represents results from this specific call
        return {'data': all_results_data, 'total': len(all_results_data), 'next_offset': current_offset}

    def process_results(self, results_json):
        """Processes Semantic Scholar JSON results into a standardized format."""
        processed = []
        initial_count = 0
        if results_json and 'data' in results_json and isinstance(results_json['data'], list):
            initial_count = len(results_json['data'])
        else:
            # Also check for 'total' and log if 0 results were found
            if results_json and results_json.get('total', 0) == 0:
                logger.info("Semantic Scholar API reported 0 results for the query.")
            else:
                logger.warning(f"Invalid or empty Semantic Scholar results received: {results_json}")
            return processed

        logger.debug(f"Processing {initial_count} raw results from Semantic Scholar.")

        for item in results_json.get('data', []):
            # --- Check for Retraction --- 
            # Semantic Scholar might indicate retractions in `publicationTypes` or a dedicated field
            # Checking for 'Retraction' in publicationTypes as a common indicator
            publication_types = item.get('publicationTypes', [])
            is_retracted = False
            if isinstance(publication_types, list):
                is_retracted = any(pt.lower() == 'retraction' for pt in publication_types)
                
            if is_retracted:
                doi = item.get('externalIds', {}).get('DOI', 'Unknown DOI')
                logger.warning(f"Skipping retracted Semantic Scholar article: DOI {doi}")
                continue # Skip this article
            # --- End Check for Retraction ---
            
            # Skip if abstract is missing or too short
            abstract = item.get('abstract', '')
            if not abstract or len(abstract) < 50:
                # logger.debug(f"Skipping Semantic Scholar item (ID: {item.get('paperId', 'N/A')}) due to missing/short abstract.")
                continue

            # Extract DOI from externalIds
            external_ids = item.get('externalIds', {})
            doi = external_ids.get('DOI')

            # Extract authors - handle the new author structure
            authors = []
            for author in item.get('authors', []):
                if isinstance(author, dict) and author.get('name'):
                    authors.append(author['name'])
            authors_str = ", ".join(authors)

            # Extract publication date (prefer 'publicationDate', fallback to 'year')
            pub_date = item.get('publicationDate') or str(item.get('year')) if item.get('year') else None

            # Extract citation count
            citation_count = item.get('citationCount', 0)

            # Accept all studies with abstracts, regardless of citation count
            processed.append({
                "doi": doi,
                "title": item.get('title', 'Untitled'),
                "authors": authors_str,
                "pub_date": pub_date,
                "abstract": abstract,
                "source_api": "semantic_scholar",
                "citation_count": citation_count
            })
            
        logger.debug(f"Finished processing Semantic Scholar results. Kept {len(processed)} out of {initial_count}.")
        return processed
# --- End Semantic Scholar Service --- 