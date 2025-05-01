import time
import concurrent.futures
import logging
import math
import re
import numpy as np
from sqlalchemy.exc import SQLAlchemyError
from psycopg2 import errors

# Import necessary components from the new structure
from .vector_store_service import VectorStoreService
from models import Study
from utils.helpers import num_tokens_from_string

logger = logging.getLogger(__name__)

# Helper function to safely convert NumPy types for JSON serialization
def safe_float(value):
    if isinstance(value, (np.float32, np.float64)):
        return float(value)
    return value

class RAGVerificationService:
    def __init__(self, gemini_service, openalex_service, crossref_service, semantic_scholar_service, pubmed_service, db_session, config, genai_module):
        """Initialize the RAG Verification Service.

        Args:
            gemini_service: Instance of GeminiService
            openalex_service: Instance of OpenAlexService
            crossref_service: Instance of CrossRefService
            semantic_scholar_service: Instance of SemanticScholarService
            pubmed_service: Instance of PubMedService
            db_session: SQLAlchemy database session
            config: Flask application configuration object
            genai_module: Google GenerativeAI module reference
        """
        self.gemini = gemini_service
        self.openalex = openalex_service
        self.crossref = crossref_service
        self.semantic_scholar = semantic_scholar_service
        self.pubmed = pubmed_service
        self.db = db_session
        self.config = config
        # Initialize vector store service with necessary dependencies
        self.vector_store = VectorStoreService(
            db_session,
            config,
            genai_module,
            config.get('GOOGLE_API_KEY') # Pass API key from config
        )

    def process_claim_request(self, claim):
        """Orchestrates the RAG workflow with improved study retrieval and logging."""
        start_time = time.time()
        logger.info(f"Starting RAG verification for claim: '{claim}'")

        # Initialize grand totals for this request
        grand_total_studies_embedded = 0
        grand_total_tokens_embedded = 0

        # Count tokens in the claim
        claim_tokens = num_tokens_from_string(claim)
        logger.info(f"Claim has {claim_tokens} tokens.")

        # 1. Preprocess Claim
        if not self.gemini:
            logger.error("Gemini service not initialized. Cannot preprocess claim.")
            return {"error": "Gemini service unavailable.", "status": "failed"}

        preprocessing_result = self.gemini.preprocess_claim(claim)
        keywords = preprocessing_result.get("keywords", [])
        synonyms = preprocessing_result.get("synonyms", [])
        search_query = preprocessing_result.get("boolean_query")
        category = preprocessing_result.get("category", "unknown")

        # Fallback logic if boolean query is bad
        if not search_query or len(search_query) < 5:
            logger.warning(f"Generated boolean query '{search_query}' seems invalid. Falling back.")
            all_terms = list(set(keywords + synonyms))
            if all_terms:
                search_query = " AND ".join(f'"{term}"' for term in all_terms if term) # Quote terms/phrases
                logger.info(f"Using fallback search query from keywords/synonyms: '{search_query}'")
            else:
                search_query = claim
                logger.info(f"Using original claim as fallback search query: '{search_query}'")

        if not search_query:
            logger.warning("No search query could be generated, cannot retrieve evidence.")
            return {"error": "Could not generate a search query from claim.", "status": "failed"}

        logger.info(f"Using Search Query: '{search_query}', Keywords: {keywords}, Synonyms: {synonyms}, Category: {category}")

        # 2. Define API Retrieval Configuration
        target_studies_per_api = 300

        openalex_limit = max(self.config['OPENALEX_MAX_RESULTS'], target_studies_per_api)
        crossref_limit = max(self.config['CROSSREF_MAX_RESULTS'], target_studies_per_api)
        semantic_scholar_limit = max(self.config['SEMANTIC_SCHOLAR_MAX_RESULTS'], target_studies_per_api)
        pubmed_limit = max(self.config['PUBMED_MAX_RESULTS'], target_studies_per_api)
        vector_search_db_top_k = self.config['RAG_TOP_K'] * 2
        logger.info(f"DB Vector Search K: {vector_search_db_top_k}")

        logger.info(f"Target study counts - OpenAlex: {openalex_limit}, CrossRef: {crossref_limit}, " +
                   f"Semantic Scholar: {semantic_scholar_limit}, PubMed: {pubmed_limit}")

        api_studies_data = []
        openalex_studies = []
        crossref_studies = []
        semantic_scholar_studies = []
        pubmed_studies = []

        # 3. Retrieve Evidence from APIs
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # --- Define API retrieval functions (get_openalex_studies, etc.) ---
            # These functions will use self.openalex, self.crossref, etc.
            # and self.config for limits.
            # IMPORTANT: Make sure these internal functions are correctly defined
            # based on the logic previously in app.py.bak

            # Example structure for one function (adapt for others):
            def get_openalex_studies():
                results = []
                page = 1
                per_page = 200
                while len(results) < openalex_limit and page <= 5:
                    try:
                        logger.info(f"Querying OpenAlex (page {page}) using query: {search_query}")
                        data = self.openalex.search_works_by_keyword(search_query, per_page=per_page, page=page)
                        if not data or not data.get('results'): break
                        batch = self.openalex.process_results(data)
                        if not batch:
                            if data and data.get('results'): logger.warning(f"OpenAlex API had results but processing yielded none on page {page}.")
                            else: break
                        batch = [s for s in batch if s.get('abstract') and len(s.get('abstract', '')) >= 50]
                        logger.info(f"OpenAlex page {page}: Found {len(batch)} usable studies")
                        results.extend(batch)
                        logger.info(f"Total OpenAlex studies so far: {len(results)}/{openalex_limit}")
                        if len(batch) < per_page and data.get('meta', {}).get('count', 0) < per_page: break
                        page += 1
                    except Exception as e:
                        logger.error(f"Error retrieving data from OpenAlex (page {page}): {e}")
                        break
                logger.info(f"FINAL OpenAlex retrieval: {len(results)} studies")
                return results

            def get_crossref_studies():
                results = []
                rows = 200
                offset = 0
                while len(results) < crossref_limit and offset < 1000:
                    try:
                        logger.info(f"Querying CrossRef (offset {offset}, rows {rows}) using query: {search_query}")
                        data = self.crossref.search_works_by_keyword(search_query, rows=rows, offset=offset)
                        if not data or 'message' not in data or 'items' not in data['message'] or not data['message']['items']: break
                        num_retrieved_api = len(data['message']['items'])
                        batch = self.crossref.process_results(data)
                        if not batch:
                            logger.info(f"No processed results from CrossRef at offset {offset} (retrieved {num_retrieved_api} raw)")
                            if num_retrieved_api > 0: offset += rows; continue
                            else: break
                        batch = [s for s in batch if s.get('abstract') and len(s.get('abstract', '')) >= 50]
                        logger.info(f"CrossRef offset {offset}: Retrieved {num_retrieved_api} raw, Processed {len(batch)} usable studies")
                        results.extend(batch)
                        logger.info(f"Total CrossRef studies so far: {len(results)}/{crossref_limit}")
                        if num_retrieved_api < rows: break
                        offset += rows
                    except Exception as e:
                        logger.error(f"Error retrieving data from CrossRef (offset {offset}): {e}")
                        break
                logger.info(f"FINAL CrossRef retrieval: {len(results)} studies")
                return results

            def get_semantic_scholar_studies():
                results = []
                limit_per_call = 100
                current_offset = 0
                max_total_offset = 1000
                while len(results) < semantic_scholar_limit and current_offset < max_total_offset:
                    try:
                        logger.info(f"Querying Semantic Scholar (offset {current_offset}, limit {limit_per_call}) using query: {search_query}")
                        data = self.semantic_scholar.search_works_by_keyword(search_query, limit=limit_per_call, offset=current_offset)
                        if not data or 'data' not in data: logger.error(f"Semantic Scholar call failed at offset {current_offset}"); break
                        num_retrieved_api = len(data.get('data', []))
                        next_offset = data.get('next_offset')
                        batch = self.semantic_scholar.process_results(data)
                        if batch:
                            logger.info(f"Semantic Scholar offset {current_offset}: Retrieved {num_retrieved_api} raw, Processed {len(batch)} usable studies")
                            results.extend(batch)
                            logger.info(f"Total Semantic Scholar studies so far: {len(results)}/{semantic_scholar_limit}")
                        else:
                            logger.info(f"No processed usable studies from Semantic Scholar at offset {current_offset} (Retrieved {num_retrieved_api} raw).")
                        if num_retrieved_api < limit_per_call or next_offset is None or next_offset <= current_offset: break
                        current_offset = next_offset
                        time.sleep(1)
                    except Exception as e:
                        logger.error(f"Error retrieving data from Semantic Scholar (offset {current_offset}): {e}")
                        if "rate limit" in str(e).lower(): time.sleep(5)
                        else: break
                logger.info(f"FINAL Semantic Scholar retrieval: {len(results)} studies")
                return results

            def get_pubmed_studies():
                results = []
                retmax_esearch = 100
                retstart = 0
                pmid_batch_size_efetch = 100
                pubmed_search_term = search_query
                logger.info(f"Using PubMed search term: {pubmed_search_term}")
                while len(results) < pubmed_limit and retstart < 1000:
                    try:
                        logger.info(f"Querying PubMed ESearch (retstart {retstart}, retmax {retmax_esearch}): {pubmed_search_term}")
                        pmids_in_window = self.pubmed.search_works_by_keyword(pubmed_search_term, retmax=retmax_esearch, retstart=retstart)
                        if pmids_in_window is None: logger.error(f"PubMed ESearch failed at retstart {retstart}"); break
                        if not pmids_in_window: logger.info(f"PubMed ESearch returned no more PMIDs at retstart {retstart}"); break
                        num_pmids_found = len(pmids_in_window)
                        logger.info(f"PubMed ESearch found {num_pmids_found} PMIDs at retstart {retstart}")
                        processed_count_in_window = 0
                        for i in range(0, num_pmids_found, pmid_batch_size_efetch):
                            batch_pmids = pmids_in_window[i:i+pmid_batch_size_efetch]
                            if not batch_pmids: continue
                            logger.info(f"Fetching details for PubMed EFetch batch ({len(batch_pmids)} PMIDs from retstart {retstart})...")
                            xml_content = self.pubmed._fetch_article_details(batch_pmids)
                            if xml_content:
                                batch_details = self.pubmed._parse_pubmed_xml(xml_content)
                                if batch_details:
                                    results.extend(batch_details)
                                    processed_count_in_window += len(batch_details)
                                    logger.info(f"  Processed {len(batch_details)} usable studies from EFetch batch. Total PubMed: {len(results)}/{pubmed_limit}")
                                else: logger.info(f"  No usable studies found after processing EFetch batch.")
                            else: logger.warning(f"  Failed to fetch details for PubMed EFetch batch (PMIDs starting with {batch_pmids[0]}).")
                            if i + pmid_batch_size_efetch < num_pmids_found: time.sleep(self.pubmed.request_delay / 2)
                        logger.info(f"Completed processing for ESearch window retstart {retstart}. Found {processed_count_in_window} usable studies.")
                        if num_pmids_found < retmax_esearch: logger.info(f"PubMed ESearch returned fewer PMIDs ({num_pmids_found}) than requested ({retmax_esearch}), indicating end."); break
                        retstart += retmax_esearch
                        if len(results) >= pubmed_limit: logger.info(f"Reached target PubMed limit ({pubmed_limit})."); break
                    except Exception as e:
                        logger.error(f"Error retrieving data from PubMed (retstart {retstart}): {e}")
                        break
                logger.info(f"FINAL PubMed retrieval: {len(results)} studies")
                return results
            # --- End Define API retrieval functions ---

            # Submit tasks
            future_openalex = executor.submit(get_openalex_studies)
            future_crossref = executor.submit(get_crossref_studies)
            future_semantic = executor.submit(get_semantic_scholar_studies)
            future_pubmed = executor.submit(get_pubmed_studies)

            # Collect results
            try: openalex_studies = future_openalex.result(); logger.info(f"Retrieved {len(openalex_studies)} usable from OpenAlex")
            except Exception as e: logger.error(f"Error in OpenAlex retrieval: {e}")
            try: crossref_studies = future_crossref.result(); logger.info(f"Retrieved {len(crossref_studies)} usable from CrossRef")
            except Exception as e: logger.error(f"Error in CrossRef retrieval: {e}")
            try: semantic_scholar_studies = future_semantic.result(); logger.info(f"Retrieved {len(semantic_scholar_studies)} usable from SemSch")
            except Exception as e: logger.error(f"Error in SemSch retrieval: {e}")
            try: pubmed_studies = future_pubmed.result(); logger.info(f"Retrieved {len(pubmed_studies)} usable from PubMed")
            except Exception as e: logger.error(f"Error in PubMed retrieval: {e}")

        # 4. Combine API studies
        api_studies_data = openalex_studies + crossref_studies + semantic_scholar_studies + pubmed_studies
        logger.info(f"TOTAL STUDIES WITH ABSTRACTS FROM APIs: {len(api_studies_data)}")
        logger.info(f"  - OpenAlex: {len(openalex_studies)}, CrossRef: {len(crossref_studies)}, SemSch: {len(semantic_scholar_studies)}, PubMed: {len(pubmed_studies)}")

        # 4b. Explicit Vector Search Against DB
        logger.info(f"Performing explicit vector search against database (top_k={vector_search_db_top_k})...")
        vector_studies_data = []
        try:
            db_vector_results = self.vector_store.find_relevant_studies(claim, top_k=vector_search_db_top_k)
            if db_vector_results:
                logger.info(f"Retrieved {len(db_vector_results)} studies from DB vector search.")
                for study in db_vector_results:
                    vector_studies_data.append({
                        "doi": study.doi, "title": study.title, "authors": study.authors,
                        "pub_date": study.pub_date, "abstract": study.abstract,
                        "source_api": study.source_api + "_db_vector",
                        "citation_count": study.citation_count or 0,
                    })
            else: logger.info("DB vector search returned no results.")
        except Exception as e: logger.error(f"Error during explicit DB vector search: {e}")

        # 5. Combine API and DB Vector Search results
        combined_studies_data = api_studies_data + vector_studies_data
        logger.info(f"TOTAL STUDIES BEFORE DEDUPLICATION (API + DB Vector): {len(combined_studies_data)}")

        # 6. Deduplicate studies
        seen_dois = set()
        seen_titles = {}
        unique_studies_data = []
        duplicates_removed_count = 0
        for study_data in combined_studies_data:
            doi = study_data.get('doi')
            title = study_data.get('title')
            doi_norm = doi.lower().strip() if doi else None
            title_lower = title.lower().strip() if title else None
            is_duplicate = False
            if doi_norm:
                if doi_norm in seen_dois: is_duplicate = True
                else:
                    seen_dois.add(doi_norm)
                    if title_lower: seen_titles[title_lower] = doi_norm
            elif title_lower:
                if title_lower in seen_titles: is_duplicate = True
                else: seen_titles[title_lower] = None
            if not is_duplicate: unique_studies_data.append(study_data)
            else: duplicates_removed_count += 1
        logger.info(f"TOTAL UNIQUE STUDIES AFTER DEDUPLICATION: {len(unique_studies_data)} (Removed {duplicates_removed_count})")

        # 7. Limit total studies
        max_evidence = self.config['MAX_EVIDENCE_TO_STORE']
        studies_to_process_data = unique_studies_data[:max_evidence]
        logger.info(f"Using {len(studies_to_process_data)} studies for analysis (max limit: {max_evidence})")

        if not studies_to_process_data:
            logger.warning("No usable evidence found after retrieval, DB search, and filtering.")
            return {
                "claim": claim, "verdict": "Inconclusive",
                "reasoning": "No relevant academic studies could be retrieved.",
                "detailed_reasoning": "No relevant academic studies with abstracts could be retrieved and processed.",
                "simplified_reasoning": "No relevant academic studies could be found.",
                "accuracy_score": 0.0, "evidence": [],
                "keywords_used": keywords, "category": category,
                "processing_time_seconds": round(time.time() - start_time, 2)
            }

        # 8. Store Evidence & Embed NEW Studies
        stored_studies = []
        new_studies_to_create_data = []
        existing_study_objects = []
        success_count = 0
        error_count = 0

        try:
            # Find existing studies
            study_dois_to_check = [s.get('doi').lower() for s in studies_to_process_data if s.get('doi')]
            existing_dois_in_db_map = {}
            if study_dois_to_check:
                existing_studies_in_db = self.db.query(Study).filter(Study.doi.in_(study_dois_to_check)).all()
                # Build enhanced DOI map (logic adapted from app.py.bak)
                for study in existing_studies_in_db:
                     if not study.doi: continue
                     raw_doi = study.doi
                     norm_doi = raw_doi.lower().strip()
                     existing_dois_in_db_map[norm_doi] = study
                     if norm_doi.startswith('https://doi.org/'):
                         plain_doi = norm_doi[16:]
                         existing_dois_in_db_map[plain_doi] = study
                     elif not norm_doi.startswith('https://doi.org/'):
                         prefixed_doi = 'https://doi.org/' + norm_doi
                         existing_dois_in_db_map[prefixed_doi] = study
                     # ... (include other variations as needed from app.py.bak) ...
                logger.info(f"Found {len(existing_studies_in_db)} of selected studies already in DB.")
                existing_study_objects = list(set(existing_dois_in_db_map.values()))
                logger.info(f"Tracking {len(existing_study_objects)} unique existing study objects.")

            # Separate new study data
            logger.info("Separating new studies from existing ones...")
            new_duplicates_found = 0
            for study_data in studies_to_process_data:
                doi = study_data.get('doi')
                doi_norm = doi.lower().strip() if doi else None
                is_duplicate = False
                if doi_norm:
                    # Use the enhanced map for checking
                    if doi_norm in existing_dois_in_db_map: is_duplicate = True
                    # ... (check other variations against the map) ...
                if is_duplicate: new_duplicates_found += 1; continue
                else: new_studies_to_create_data.append(study_data)
            if new_duplicates_found > 0: logger.info(f"Enhanced DOI check found {new_duplicates_found} more duplicates.")
            logger.info(f"Identified {len(new_studies_to_create_data)} potential new studies.")

            # 9. CONCURRENT EMBEDDING FOR NEW STUDIES
            new_study_objects_with_embeddings = []
            if new_studies_to_create_data:
                texts_to_embed = []
                original_indices_map = {}
                studies_needing_embedding_count = 0
                for idx, study_data in enumerate(new_studies_to_create_data):
                    if study_data.get('abstract'):
                        texts_to_embed.append(study_data['abstract'])
                        original_indices_map[len(texts_to_embed) - 1] = idx
                        studies_needing_embedding_count += 1

                logger.info(f"Starting concurrent embedding for {studies_needing_embedding_count} new studies...")
                embeddings = [None] * len(new_studies_to_create_data)
                if texts_to_embed:
                    embedding_start_time = time.time()
                    try:
                        batch_embeddings = self.vector_store.get_embedding_for_text(
                            texts_to_embed, task_type="retrieval_document"
                        )
                        if batch_embeddings and len(batch_embeddings) == len(texts_to_embed):
                            for embed_idx, embedding in enumerate(batch_embeddings):
                                original_data_idx = original_indices_map.get(embed_idx)
                                if original_data_idx is not None:
                                    if embedding:
                                        embeddings[original_data_idx] = embedding
                                        grand_total_studies_embedded += 1
                                    else: logger.warning(f"Embedding failed for new study data at index {original_data_idx}.")
                        else: logger.error(f"Concurrent embedding failed or returned mismatched results.")
                    except Exception as embed_err: logger.error(f"Error during concurrent embedding: {embed_err}")
                    embedding_duration = time.time() - embedding_start_time
                    estimated_tokens = sum(num_tokens_from_string(txt) for txt, emb in zip(texts_to_embed, batch_embeddings or []) if emb)
                    grand_total_tokens_embedded = estimated_tokens
                    logger.info(f"Embedding finished in {embedding_duration:.2f}s. Embedded: {grand_total_studies_embedded}/{studies_needing_embedding_count}. Tokens: {estimated_tokens}")

                # Deduplicate new study data before object creation (using logic from app.py.bak)
                unique_new_study_data_map = {}
                deduplicated_new_studies_data = []
                original_embeddings_map = {idx: emb for idx, emb in enumerate(embeddings)}
                new_data_duplicates_removed = 0
                for idx, study_data in enumerate(new_studies_to_create_data):
                    doi = study_data.get('doi')
                    doi_norm = doi.lower().strip() if doi else None
                    title = study_data.get('title')
                    title_lower = title.lower().strip() if title else None
                    is_duplicate = False
                    if doi_norm:
                        if doi_norm in unique_new_study_data_map: is_duplicate = True
                        else: unique_new_study_data_map[doi_norm] = idx
                    elif title_lower:
                        if title_lower in unique_new_study_data_map: is_duplicate = True
                        else: unique_new_study_data_map[title_lower] = idx
                    if not is_duplicate:
                        study_data['_original_embedding'] = original_embeddings_map.get(idx)
                        deduplicated_new_studies_data.append(study_data)
                    else: new_data_duplicates_removed += 1
                logger.info(f"Removed {new_data_duplicates_removed} duplicates from new study list. Processing {len(deduplicated_new_studies_data)} unique new studies.")

                # Create Study objects
                logger.info("Creating new Study objects...")
                for study_data in deduplicated_new_studies_data:
                    try:
                        source_api_cleaned = study_data.get('source_api', '').replace('_db_vector', '')
                        doi_raw = study_data.get('doi')
                        doi_cleaned = doi_raw.strip() if doi_raw else None
                        study_obj = Study(
                            doi=doi_cleaned, title=study_data.get('title'),
                            authors=study_data.get('authors'), pub_date=study_data.get('pub_date'),
                            abstract=study_data.get('abstract'), source_api=source_api_cleaned,
                            citation_count=study_data.get('citation_count', 0),
                            embedding=study_data['_original_embedding']
                        )
                        new_study_objects_with_embeddings.append(study_obj)
                        success_count += 1
                    except Exception as e: logger.error(f"Error creating Study object (DOI: {study_data.get('doi')}): {e}"); error_count += 1

            # 10. ADD AND COMMIT NEW STUDIES
            if new_study_objects_with_embeddings:
                logger.info(f"Adding {len(new_study_objects_with_embeddings)} new studies to DB session...")
                try:
                    self.db.add_all(new_study_objects_with_embeddings)
                    logger.info("Committing new studies...")
                    commit_start_time = time.time()
                    self.db.commit()
                    commit_duration = time.time() - commit_start_time
                    logger.info(f"Successfully committed {len(new_study_objects_with_embeddings)} new studies in {commit_duration:.2f}s.")
                except SQLAlchemyError as e:
                    self.db.rollback()
                    error_msg = str(e)
                    logger.error(f"Database error committing new studies: {error_msg}. Rolling back.")
                    # Fallback logic (batch/individual commits) adapted from app.py.bak
                    if "UniqueViolation" in error_msg and "doi" in error_msg:
                        logger.warning("UniqueViolation detected. Trying smaller batches...")
                        # --- Implement batch/individual commit fallback here ---
                        # This involves querying existing DOIs, iterating in batches,
                        # trying commits, handling errors, and potentially committing one by one.
                        # See the detailed logic in app.py.bak lines 1078-1184
                        # Remember to update success_count, error_count, and
                        # new_study_objects_with_embeddings based on the outcome.
                        # Placeholder for the complex fallback logic:
                        logger.error("Batch commit fallback logic not fully implemented in this refactored version.")
                        error_count += len(new_study_objects_with_embeddings) # Count all as errors for now
                        new_study_objects_with_embeddings = [] # Clear list as commit failed

                    else: # Non-unique constraint error
                         logger.error(f"Database error not related to DOI duplicates: {error_msg}")
                         error_count += len(new_study_objects_with_embeddings)
                         new_study_objects_with_embeddings = []
                except Exception as e:
                    self.db.rollback()
                    logger.error(f"Unexpected error committing new studies: {e}. Rolling back.")
                    error_count += len(new_study_objects_with_embeddings)
                    new_study_objects_with_embeddings = []

            # 11. GATHER FINAL LIST FOR RANKING
            stored_studies = existing_study_objects + new_study_objects_with_embeddings
            logger.info(f"Total studies available for ranking: {len(stored_studies)} ({len(existing_study_objects)} existing, {len(new_study_objects_with_embeddings)} new)")
            logger.info(f"DB/PRE-RANKING SUMMARY: Processed {success_count} data, {error_count} errors.")
            logger.info(f"EMBEDDING SUMMARY: Embedded {grand_total_studies_embedded} new studies, {grand_total_tokens_embedded} tokens (est).")

        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error during evidence storage/embedding phase: {e}")
            return {"error": "Database error storing/embedding evidence.", "status": "failed"}
        except Exception as e:
            self.db.rollback()
            logger.error(f"Unexpected error during evidence storage/embedding phase: {e}")
            return {"error": f"Unexpected error: {str(e)}", "status": "failed"}

        # 12. Rank Studies
        logger.info(f"Ranking {len(stored_studies)} studies for RAG analysis...")
        claim_embedding_np = None
        try:
            claim_embedding_list = self.vector_store.get_embedding_for_text([claim], task_type="retrieval_query")
            if not claim_embedding_list or claim_embedding_list[0] is None:
                logger.error("Failed to generate claim embedding for ranking.")
                return {"error": "Failed to generate claim embedding.", "status": "failed"}

            claim_embedding = claim_embedding_list[0]
            expected_dimension = self.config['EMBEDDING_DIMENSION']
            if len(claim_embedding) != expected_dimension:
                 logger.error(f"Claim embedding dim mismatch ({len(claim_embedding)} vs {expected_dimension}). Cannot rank.")
                 return {"error": "Claim embedding dimension mismatch.", "status": "failed"}
            claim_embedding_np = np.array(claim_embedding).astype('float32')
        except Exception as e:
            logger.error(f"Error generating claim embedding: {e}")
            return {"error": "Failed to generate claim embedding.", "status": "failed"}

        ranked_studies_with_scores = []
        studies_with_valid_embeddings_count = 0
        for study in stored_studies:
            relevance_score = 0.0
            study_embedding_np = None
            study_has_valid_embedding = False
            if study.embedding is not None and len(study.embedding) == expected_dimension:
                study_embedding_np = np.array(study.embedding).astype('float32')
                study_has_valid_embedding = True
                studies_with_valid_embeddings_count += 1

            if claim_embedding_np is not None and study_has_valid_embedding:
                try:
                    dot_product = np.dot(claim_embedding_np, study_embedding_np)
                    norm_claim = np.linalg.norm(claim_embedding_np)
                    norm_study = np.linalg.norm(study_embedding_np)
                    if norm_claim > 0 and norm_study > 0:
                        similarity = dot_product / (norm_claim * norm_study)
                        relevance_score = max(0.0, min(1.0, similarity))
                    else: logger.warning(f"Zero norm for claim or study {study.doi or study.id}.")
                except Exception as e: logger.warning(f"Error calculating relevance for study {study.doi or study.id}: {e}")

            citation_count = study.citation_count if study.citation_count is not None else 0
            try: credibility_score = math.log10(citation_count + 1)
            except ValueError: credibility_score = 0.0; logger.warning(f"Invalid citation count {study.citation_count} for study {study.doi or study.id}.")

            relevance_weight = 0.7 if study_has_valid_embedding and relevance_score > 0 else 0.1
            credibility_weight = 1.0 - relevance_weight
            combined_score = (relevance_weight * relevance_score) + (credibility_weight * credibility_score)
            ranked_studies_with_scores.append((study, combined_score, relevance_score, credibility_score))

        ranked_studies_with_scores.sort(key=lambda item: item[1], reverse=True)
        top_k = self.config['RAG_TOP_K']
        top_ranked_studies_with_scores = ranked_studies_with_scores[:top_k]
        top_ranked_studies = [study for study, score, rel, cred in top_ranked_studies_with_scores]
        relevant_chunks = [study.abstract for study in top_ranked_studies if study.abstract]

        logger.info(f"Selected top {len(top_ranked_studies)} ranked studies for LLM analysis from {len(stored_studies)}.")
        # Log score distribution (optional)

        if not relevant_chunks:
             logger.warning("No abstracts available from top ranked studies for RAG.")
             return {
                "claim": claim, "verdict": "Inconclusive",
                "reasoning": "Relevant studies found, but none had abstracts.",
                "detailed_reasoning": "Relevant studies were found, but none had abstracts suitable for analysis.",
                "simplified_reasoning": "Could not find study details.",
                "accuracy_score": 0.0, "evidence": [],
                "keywords_used": keywords, "category": category,
                "processing_time_seconds": round(time.time() - start_time, 2)
             }

        # 13. Analyze with LLM
        if not self.gemini:
            logger.error("Gemini service not available for final analysis.")
            return {"error": "Gemini service unavailable for analysis.", "status": "failed"}
        analysis_result = self.gemini.analyze_with_rag(claim, relevant_chunks)

        # 14. Format and Return Output
        evidence_details = []
        for idx, (study, score, rel, cred) in enumerate(top_ranked_studies_with_scores):
            if study.abstract:
                evidence_details.append({
                    "id": idx + 1, "title": study.title, "abstract": study.abstract,
                    "authors": study.authors, "doi": study.doi, "pub_date": study.pub_date,
                    "source_api": study.source_api, "citation_count": study.citation_count or 0,
                    "combined_score": round(safe_float(score), 4),
                    "relevance_score": round(safe_float(rel), 4),
                    "credibility_score": round(safe_float(cred), 4)
                })

        total_chunk_tokens = sum(num_tokens_from_string(chunk) for chunk in relevant_chunks)
        avg_token_per_chunk = total_chunk_tokens / len(relevant_chunks) if relevant_chunks else 0
        total_process_time = time.time() - start_time

        # Print summary logs (adapted from app.py.bak)
        logger.info("="*80)
        logger.info("📊 CLAIM VERIFICATION SUMMARY 📊")
        # ... (add detailed logging as in app.py.bak lines 1421-1451) ...
        logger.info(f"  • Verdict: {analysis_result.get('verdict', 'Error')}")
        logger.info(f"  • Accuracy Score: {analysis_result.get('accuracy_score', 0.0)}")
        logger.info(f"  • Processing Time: {total_process_time:.2f} seconds")
        logger.info("="*80)

        final_response = {
            "claim": claim,
            "verdict": analysis_result.get("verdict", "Error"),
            "reasoning": analysis_result.get("reasoning", "Analysis failed."),
            "detailed_reasoning": analysis_result.get("detailed_reasoning", analysis_result.get("reasoning", "Analysis failed.")),
            "simplified_reasoning": analysis_result.get("simplified_reasoning", analysis_result.get("reasoning", "Analysis failed.")),
            "accuracy_score": analysis_result.get("accuracy_score", 0.0),
            "evidence": evidence_details,
            "keywords_used": keywords,
            "synonyms_identified": synonyms,
            "search_query_used": search_query,
            "category": category,
            "processing_time_seconds": round(total_process_time, 2)
        }

        return final_response