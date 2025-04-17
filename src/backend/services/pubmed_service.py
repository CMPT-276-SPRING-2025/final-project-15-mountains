import requests
import time
import json
import logging
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

# --- New: PubMed Service ---
class PubMedService:
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self, config, clean_abstract_func, email=None):
        self.config = config
        self.clean_abstract_func = clean_abstract_func # Store the function
        self.email = email or self.config.get('OPENALEX_EMAIL') # Reuse email for politeness
        if not self.email:
            logger.warning("Email not set for PubMed, using default. Set OPENALEX_EMAIL for polite API usage.")
            self.email = 'rba137@sfu.ca' # Default if not set
        self.headers = {'User-Agent': f'Factify/1.0 (mailto:{self.email})'}
        self.timeout = 30 # Increased timeout
        # Rate limiting: NCBI allows 3 requests/second without API key, 10/second with key.
        # We'll add a small delay between requests.
        self.request_delay = 0.5 # Increased delay to 0.5 seconds (was 0.4)

    def _fetch_article_details(self, pmids):
        """Fetches detailed article information for a list of PMIDs with retry logic."""
        if not pmids:
            return []

        pmid_str = ",".join(pmids)
        fetch_url = f"{self.BASE_URL}/efetch.fcgi"
        params = {
            'db': 'pubmed',
            'id': pmid_str,
            'retmode': 'xml',
            'rettype': 'abstract',
            'email': self.email
        }
        logger.info(f"Fetching PubMed details for {len(pmids)} PMIDs...")

        max_retries = 2
        retry_delay = 1.5 # Seconds to wait before retrying after a 429

        for attempt in range(max_retries + 1):
            try:
                # Ensure delay *before* each attempt
                if attempt > 0:
                    logger.warning(f"Rate limit hit (429). Retrying PubMed EFetch in {retry_delay}s... (Attempt {attempt}/{max_retries})")
                    time.sleep(retry_delay)
                else:
                     # Still apply base delay before the first attempt of a fetch
                     time.sleep(self.request_delay)

                response = requests.get(fetch_url, params=params, headers=self.headers, timeout=self.timeout)

                if response.status_code == 429:
                    # If it's the last attempt, raise the error, otherwise loop will handle delay/retry
                    if attempt == max_retries:
                        response.raise_for_status() # Raise the 429 error
                    else:
                        continue # Go to the next attempt after delay
                
                # Log other non-200 status codes before raising
                if response.status_code != 200:
                    logger.error(f"PubMed EFetch returned non-200 status: {response.status_code} for PMIDs starting with {pmids[0]}")
                    # Optionally log response text for debugging, be careful with large responses
                    # logger.debug(f"PubMed EFetch error response text: {response.text[:500]}")
                    response.raise_for_status() # Raise other HTTP errors immediately
                
                logger.debug(f"PubMed EFetch successful for PMIDs starting with {pmids[0]}")
                return response.text # Return XML content on success

            except requests.exceptions.RequestException as e:
                # If it's the last attempt or not a 429 error, log and return None
                if attempt == max_retries or (response and response.status_code != 429):
                     logger.error(f"PubMed EFetch request failed after {attempt+1} attempt(s): {e}")
                     return None
                 # Otherwise, the loop will retry for 429

        return None # Should not be reached if loop logic is correct, but as fallback

    def _parse_pubmed_xml(self, xml_content):
        """Parses the XML from EFetch into a standardized list of dictionaries."""
        processed = []
        initial_count = 0
        if not xml_content:
            logger.warning("Received empty XML content for PubMed parsing.")
            return processed

        try:
            root = ET.fromstring(xml_content)
            articles = root.findall('.//PubmedArticle')
            initial_count = len(articles)
            logger.debug(f"Processing {initial_count} raw articles found in PubMed XML.")
            
            for article in articles:
                medline_citation = article.find('.//MedlineCitation')
                if medline_citation is None:
                    # logger.debug("Skipping article due to missing MedlineCitation element.")
                    continue

                # --- Check for Retraction --- 
                pub_status_elem = medline_citation.find('Article/PublicationStatus')
                if pub_status_elem is not None and pub_status_elem.text == 'Retracted Publication':
                    pmid_elem = medline_citation.find('PMID')
                    pmid = pmid_elem.text if pmid_elem is not None else 'Unknown PMID'
                    logger.warning(f"Skipping retracted PubMed article: PMID {pmid}")
                    continue # Skip this article
                # --- End Check for Retraction ---

                pmid_elem = medline_citation.find('PMID')
                pmid = pmid_elem.text if pmid_elem is not None else None

                article_elem = medline_citation.find('Article')
                if article_elem is None:
                    # logger.debug(f"Skipping article PMID {pmid} due to missing Article element.")
                    continue

                title_elem = article_elem.find('ArticleTitle')
                title = title_elem.text if title_elem is not None else 'Untitled'

                abstract_elem = article_elem.find('.//Abstract/AbstractText')
                abstract = abstract_elem.text if abstract_elem is not None else ''
                
                 # Skip if abstract is missing or too short
                if not abstract or len(abstract) < 50:
                    # logger.debug(f"Skipping article PMID {pmid} due to missing/short abstract.")
                    continue
                    
                # Clean the abstract using the passed function
                abstract = self.clean_abstract_func(abstract)
                
                # Extract authors
                authors_list = []
                author_list_elem = article_elem.find('AuthorList')
                if author_list_elem is not None:
                    for author in author_list_elem.findall('Author'):
                        last_name_elem = author.find('LastName')
                        fore_name_elem = author.find('ForeName')
                        initials_elem = author.find('Initials')
                        name_parts = []
                        if fore_name_elem is not None and fore_name_elem.text:
                             name_parts.append(fore_name_elem.text)
                        elif initials_elem is not None and initials_elem.text:
                             name_parts.append(initials_elem.text + '.') # Add dot for initials
                        if last_name_elem is not None and last_name_elem.text:
                             name_parts.append(last_name_elem.text)
                        if name_parts:
                            authors_list.append(" ".join(name_parts))
                authors = ", ".join(authors_list)

                # Extract publication date (simplified - prefers journal issue pub date)
                pub_date = None
                journal_issue = article_elem.find('.//Journal/JournalIssue')
                if journal_issue is not None:
                    pub_date_elem = journal_issue.find('PubDate')
                    if pub_date_elem is not None:
                        year_elem = pub_date_elem.find('Year')
                        month_elem = pub_date_elem.find('Month')
                        day_elem = pub_date_elem.find('Day')
                        date_parts = []
                        if year_elem is not None and year_elem.text:
                            date_parts.append(year_elem.text)
                            if month_elem is not None and month_elem.text:
                                date_parts.append(month_elem.text.zfill(2)) # Pad month
                                if day_elem is not None and day_elem.text:
                                    date_parts.append(day_elem.text.zfill(2)) # Pad day
                        pub_date = "-".join(date_parts)

                # Extract DOI if available
                doi = None
                doi_elem = article_elem.find(".//ELocationID[@EIdType='doi']") or article_elem.find(".//ArticleId[@IdType='doi']")
                if doi_elem is not None:
                    doi = doi_elem.text
                    
                # Citation count is not directly available via EFetch search results.
                # Set to 0 as a placeholder.
                citation_count = 0

                processed.append({
                    "doi": doi,
                    "title": title,
                    "authors": authors,
                    "pub_date": pub_date,
                    "abstract": abstract,
                    "source_api": "pubmed",
                    "citation_count": citation_count,
                    "pmid": pmid # Include PMID for potential future use
                })
        except ET.ParseError as e:
            logger.error(f"Error parsing PubMed XML: {e}")
            # Optionally log the problematic XML content (first few hundred chars)
            # logger.debug(f"Problematic PubMed XML content (start): {xml_content[:500]}")
        except Exception as e:
            logger.error(f"Unexpected error during PubMed XML processing: {e}")

        logger.debug(f"Finished processing PubMed XML. Kept {len(processed)} out of {initial_count} articles.")
        return processed

    def search_works_by_keyword(self, keywords, retmax=10, retstart=0):
        """Search PubMed PMIDs using keyword query via ESearch with pagination support."""
        search_query = keywords if isinstance(keywords, str) else " ".join(keywords)
        retmax = min(retmax, 500) # Keep a reasonable cap on ESearch batch size

        esearch_url = f"{self.BASE_URL}/esearch.fcgi"
        params = {
            'db': 'pubmed',
            'term': search_query,
            'retmax': retmax,
            'retstart': retstart,
            'retmode': 'json',
            'sort': 'relevance',
            'email': self.email
        }
        logger.info(f"Querying PubMed ESearch PMIDs: '{search_query}' with retmax={retmax}, retstart={retstart}")
        try:
            time.sleep(self.request_delay) # Adhere to rate limits
            response = requests.get(esearch_url, params=params, headers=self.headers, timeout=self.timeout)
            # Add simple logging for the request URL
            logger.debug(f"PubMed ESearch Request URL: {response.url}")
            response.raise_for_status() # Raise HTTP errors
            esearch_results = response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"PubMed ESearch request failed: {e}")
            return None # Return None on failure
        except json.JSONDecodeError as e:
             logger.error(f"Failed to decode PubMed ESearch JSON response: {e}")
             return None # Return None on failure

        if 'esearchresult' not in esearch_results or 'idlist' not in esearch_results['esearchresult']:
            logger.warning("PubMed ESearch returned no results or unexpected format.")
            return [] # Return empty list if no IDs found

        pmids = esearch_results['esearchresult']['idlist']
        logger.info(f"PubMed ESearch found {len(pmids)} PMIDs for offset {retstart}.")
        return pmids # Return only the list of PMIDs

    # process_results is integrated into search_works_by_keyword via _parse_pubmed_xml
# --- End PubMed Service --- 