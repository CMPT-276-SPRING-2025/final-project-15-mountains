# app.py
import os
import concurrent.futures
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json
import time
import google.generativeai as genai
from dotenv import load_dotenv
import logging
import datetime
import re  # Add this at the top with other imports
import socket  # For DNS resolution test
import numpy as np # Add numpy import
import math # For log scaling
import xml.etree.ElementTree as ET # Add this for PubMed XML parsing
import tiktoken # Add tiktoken import

# Set Tokenizer Parallelism to avoid fork issues (can be 'true' or 'false')
# Setting to 'false' is often safer in web server environments
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Restrict FAISS threading to avoid resource issues
os.environ["OMP_NUM_THREADS"] = "4"  # Limit OpenMP threads used by FAISS

# --- New Imports ---
import sqlite3
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, Date, MetaData, Table, text, or_ # Add or_ import
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError
# --- End New Imports ---
# --- New pgvector Import ---
from pgvector.sqlalchemy import Vector
# --- End pgvector Import ---
# --- Import for UniqueViolation ---
from psycopg2 import errors
# --- End Import ---

# --- Import Services ---
from services.openalex_service import OpenAlexService
from services.crossref_service import CrossRefService
from services.semantic_scholar_service import SemanticScholarService
from services.pubmed_service import PubMedService
from services.gemini_service import GeminiService
# --- End Import Services ---

# Configure logging
logging.basicConfig(level=logging.INFO)
# Add this line to reduce SQLAlchemy logging noise on errors
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# --- Database Config ---
app.config.update(
    SECRET_KEY=os.getenv('SECRET_KEY', 'factify-dev-key'),
    DEBUG=os.getenv('FLASK_DEBUG', 'False') == 'True',
    GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY'),
    OPENALEX_EMAIL=os.getenv('OPENALEX_EMAIL', 'rba137@sfu.ca'),
    # --- Database Config ---
    DATABASE_URL=os.getenv('DATABASE_URL', 'postgresql://postgres:password@localhost:5432/postgres'),
    EMBEDDING_MODEL_API = "models/text-embedding-004", # Using Gemini embedding API model
    # --- THIS VALUE WILL BE DYNAMICALLY UPDATED BASED ON SCHEMA DETECTION ---
    EMBEDDING_DIMENSION = 768, # Default dimension for text-embedding-004
    # Replace single limit with specific API limits
    OPENALEX_MAX_RESULTS=int(os.getenv('OPENALEX_MAX_RESULTS', '200')),
    CROSSREF_MAX_RESULTS=int(os.getenv('CROSSREF_MAX_RESULTS', '1000')),
    SEMANTIC_SCHOLAR_MAX_RESULTS=int(os.getenv('SEMANTIC_SCHOLAR_MAX_RESULTS', '100')),
    PUBMED_MAX_RESULTS=int(os.getenv('PUBMED_MAX_RESULTS', '200')), # Max results for PubMed ESearch
    # Keep these settings
    MAX_EVIDENCE_TO_STORE=int(os.getenv('MAX_EVIDENCE_TO_STORE', '100')), # Reduced default from 800 to 100
    RAG_TOP_K=int(os.getenv('RAG_TOP_K', '20')), # Number of chunks for RAG analysis
)

# --- Initialize Gemini API
gemini_api_key = app.config.get('GOOGLE_API_KEY')
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)

    gemini_model = genai.GenerativeModel('gemini-2.0-flash') # Using 2.0 Flash as the model.
else:
    print("WARNING: GOOGLE_API_KEY not set. Gemini features will not work.")
    gemini_model = None

# --- New: Database Setup ---
Base = declarative_base()

# Log the database URL (without password)
database_url = app.config['DATABASE_URL']
logger.info(f"Using database connection: {database_url.split('@')[0].split(':')[0]}:****@{database_url.split('@')[1] if '@' in database_url else 'localhost'}")

# Test DNS resolution for database host
try:
    hostname = database_url.split('@')[1].split('/')[0].split(':')[0]
    logger.info(f"Testing DNS resolution for: {hostname}")
    resolved_ip = socket.gethostbyname(hostname)
    logger.info(f"Successfully resolved {hostname} to {resolved_ip}")
except socket.gaierror as e:
    logger.error(f"DNS resolution error for {hostname}: {e}")
    logger.error("Please check your network connection, DNS settings, or if the hostname is correct")
    # Provide helpful suggestions for Supabase issues
    if 'supabase.co' in database_url:
        logger.error("SUPABASE CONNECTION TROUBLESHOOTING:")
        logger.error("1. Check if your DATABASE_URL is correctly formatted.")
        logger.error("2. For direct connections, use: postgresql://postgres:password@db.YOUR-PROJECT-REF.supabase.co:5432/postgres")
        logger.error("3. For pooler connections, use: postgresql://postgres.YOUR-PROJECT-REF:password@aws-0-REGION.pooler.supabase.com:5432/postgres")
        logger.error("4. Ensure your project reference ID in the connection string is correct.")
        logger.error("5. Verify your Supabase database is active in the Supabase dashboard.")
    # Don't exit here, let SQLAlchemy handle the connection error

# Handle potential SSL requirement for PostgreSQL connection
if database_url.startswith('postgresql'):
    # Force SSL mode for Supabase connections
    if 'supabase.co' in database_url:
        logger.info("Supabase connection detected, setting SSL requirements")
        if '?' not in database_url:
            database_url += "?sslmode=require"
        elif 'sslmode=' not in database_url:
            database_url += "&sslmode=require"
        
try:
    # Create the engine with more detailed error handling
    logger.info(f"Creating database engine...")
    engine = create_engine(
        database_url, 
        pool_pre_ping=True,  # Add health checks for connections
        connect_args={
            # Longer timeout for slow connections
            'connect_timeout': 30
        } if database_url.startswith('postgresql') else {}
    )
    
    # Test the connection
    logger.info("Testing database connection...")
    with engine.connect() as connection:
        # Simple query to test connectivity
        connection.execute(text("SELECT 1"))
        logger.info("Database connection test successful!")
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
except Exception as e:
    logger.error(f"Failed to create database engine or test connection: {e}")
    raise

class Study(Base):
    __tablename__ = "studies"
    id = Column(Integer, primary_key=True, index=True)
    claim_id = Column(Integer, index=True) # Link study to a specific claim request if needed
    doi = Column(String, unique=True, index=True)
    title = Column(Text)
    authors = Column(Text, nullable=True) # Store as JSON string or delimited
    pub_date = Column(String, nullable=True) # Store as string for flexibility
    abstract = Column(Text, nullable=True)
    source_api = Column(String) # 'crossref' or 'openalex'
    retrieved_at = Column(Date, default=datetime.date.today)
    citation_count = Column(Integer, nullable=True, default=0) # Add citation count column
    # --- Add pgvector column ---
    embedding = Column(Vector(app.config['EMBEDDING_DIMENSION']), nullable=True)
    # --- End Add pgvector column ---
    # relevance_score = Column(Float, nullable=True) # Add if calculated

# Create tables if they don't exist (better to use Alembic migrations)
logger.info("Creating database tables if they don't exist...")
try:
    # Make sure the vector extension is available before creating tables
    with engine.connect() as conn:
        try:
             logger.info("Checking if vector extension is enabled...")
             conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
             conn.commit()
             logger.info("Vector extension check complete.")
        except Exception as ext_e:
             logger.warning(f"Could not automatically enable pgvector extension: {ext_e}. Please ensure it is enabled manually in your Supabase dashboard.")

    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully!")
except Exception as e:
    logger.error(f"Failed to create database tables: {e}")
    raise

# Modified to work with both SQLite and PostgreSQL
def add_column_if_not_exists():
    try:
        # Get a connection and detect database type
        # This function is now simplified as create_all handles column additions
        # if the model definition changes. We'll primarily use it for the index.
        # We assume PostgreSQL with pgvector. If SQLite is needed, this requires more complex logic.
        with engine.connect() as conn:
             dialect = engine.dialect.name
             logger.info(f"Database dialect detected: {dialect}")
             if dialect == 'postgresql':
                 logger.info("Checking/Adding columns and vector index for PostgreSQL...")
                 # Check/Add citation_count (SQLAlchemy's create_all should handle this, but belt-and-suspenders)
                 try:
                     conn.execute(text("ALTER TABLE studies ADD COLUMN IF NOT EXISTS citation_count INTEGER DEFAULT 0"))
                     conn.commit() # Commit after ALTER TABLE
                     logger.info("Checked/Added citation_count column.")
                 except Exception as e:
                     logger.warning(f"Could not add citation_count column (might already exist or other issue): {e}")
                     conn.rollback()

                 # Explicitly check and add the embedding column if it doesn't exist
                 # Check if the embedding column exists using information_schema
                 column_check_query = text("""
                     SELECT EXISTS (
                         SELECT 1
                         FROM information_schema.columns
                         WHERE table_schema = 'public' -- Adjust if using a different schema
                         AND table_name = 'studies'
                         AND column_name = 'embedding'
                     );
                 """)
                 embedding_column_exists = conn.execute(column_check_query).scalar()

                 if not embedding_column_exists:
                    logger.info("'embedding' column not found. Attempting to add it...")
                    try:
                        # Ensure vector extension is available
                        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                        # Add the column with the correct dimension
                        conn.execute(text(f"ALTER TABLE studies ADD COLUMN embedding vector({app.config['EMBEDDING_DIMENSION']})"))
                        conn.commit()
                        logger.info(f"Successfully added 'embedding' column with dimension {app.config['EMBEDDING_DIMENSION']}.")
                    except Exception as add_col_e:
                        logger.error(f"Failed to add 'embedding' column: {add_col_e}")
                        conn.rollback()
                 else:
                     logger.info("'embedding' column already exists (checked via information_schema).")

                 # Create HNSW index for faster vector similarity search (Recommended for performance)
                 # You can adjust 'm' and 'ef_construction' based on recall/performance needs.
                 # Index types: https://github.com/pgvector/pgvector#indexing
                 # Using cosine distance (<=>) as it's common for sentence embeddings.
                 # Use L2 distance (<->) if your model/task suits it better.
                 # Use inner product (<#>) for max inner product search.
                 index_name = "idx_studies_embedding_cosine"
                 index_check_query = text(f"SELECT 1 FROM pg_indexes WHERE indexname = '{index_name}'")
                 index_exists = conn.execute(index_check_query).scalar()

                 if not index_exists:
                     logger.info(f"Creating HNSW index '{index_name}' on studies(embedding)... This may take time.")
                     try:
                         # Ensure vector extension is loaded in this session
                         conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                         # Create the index using cosine distance
                         # Adjust lists_or_m based on expected data size and performance needs
                         # For HNSW: m=16, ef_construction=64 are common starting points
                         conn.execute(text(f"""
                             CREATE INDEX {index_name} ON studies
                             USING hnsw (embedding vector_cosine_ops)
                             WITH (m = 16, ef_construction = 64)
                         """))
                         conn.commit() # Commit after CREATE INDEX
                         logger.info(f"Successfully created HNSW index '{index_name}'.")
                     except Exception as e:
                         logger.error(f"Failed to create HNSW index '{index_name}': {e}")
                         conn.rollback() # Rollback on error
                 else:
                      logger.info(f"HNSW index '{index_name}' already exists.")
             else:
                 logger.warning(f"Database dialect '{dialect}' detected. Automatic schema migration for vector column/index is only implemented for PostgreSQL.")

        logger.info("Database schema and index check complete.")
    except Exception as e:
        logger.error(f"Error checking or updating database schema/index: {e}")

# Perform the schema check/update
add_column_if_not_exists()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
# --- End Database Setup ---

# Add the token counting function back
def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string using OpenAI's tiktoken."""
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens
    except Exception as e:
        logger.error(f"Error counting tokens: {e}")
        # Return a rough estimate if tiktoken fails
        return len(string.split()) * 1.3  # Rough approximation

# --- New: Vector Store Service using pgvector ---
class VectorStoreService:
    def __init__(self, db_session):
        self.db = db_session

    def get_embedding_for_text(self, texts: list[str], task_type="retrieval_document") -> list[list[float] | None]:
        """Generates embeddings for a list of texts using Gemini API and adapts dimensions."""
        if not gemini_api_key:
            logger.error("GOOGLE_API_KEY not set. Cannot generate embeddings.")
            # Return None for each text if key is missing
            return [None] * len(texts)
        if not texts:
            logger.warning("Attempted to embed an empty list of texts.")
            return []

        # Truncate individual texts if too long
        MAX_CHARS = 20000
        processed_texts = []
        original_indices = [] # Keep track of which texts were valid
        for i, text in enumerate(texts):
            if not text:
                logger.warning(f"Skipping empty text at index {i}.")
                continue # Skip empty strings
            if len(text) > MAX_CHARS:
                logger.warning(f"Text at index {i} too long ({len(text)} chars), truncating to {MAX_CHARS} chars.")
                processed_texts.append(text[:MAX_CHARS])
            else:
                processed_texts.append(text)
            original_indices.append(i) # Store the original index of the valid text
            
        if not processed_texts:
             logger.warning("No valid texts left after preprocessing for embedding.")
             return [None] * len(texts)

        try:
            # Gemini API call with the batch of processed texts
            # Assuming genai.embed_content handles a list in the 'content' field
            result = genai.embed_content(
                model=app.config['EMBEDDING_MODEL_API'],
                content=processed_texts, # Send the list of texts
                task_type=task_type # Specify task type for the batch
            )

            # Extract the list of embeddings
            batch_embeddings = result.get('embedding', [])

            if len(batch_embeddings) != len(processed_texts):
                 logger.error(f"Mismatch between number of texts sent ({len(processed_texts)}) and embeddings received ({len(batch_embeddings)}). Returning None for all.")
                 # Create a result list with None for all original texts
                 final_embeddings = [None] * len(texts)
                 return final_embeddings

            # Adapt dimensions for each embedding in the batch
            expected_dimension = app.config['EMBEDDING_DIMENSION']
            adapted_embeddings_map = {}
            for i, embedding in enumerate(batch_embeddings):
                if not isinstance(embedding, list):
                     logger.warning(f"Received non-list embedding at index {i}, skipping.")
                     adapted_embeddings_map[original_indices[i]] = None
                     continue
                     
                actual_dimension = len(embedding)
                adapted_embedding = embedding
                if actual_dimension != expected_dimension:
                    logger.debug(f"Adapting embedding {i} from dimension {actual_dimension} to {expected_dimension}")
                    if actual_dimension > expected_dimension:
                        adapted_embedding = embedding[:expected_dimension]
                    else:
                        adapted_embedding = embedding + [0.0] * (expected_dimension - actual_dimension)
                
                adapted_embeddings_map[original_indices[i]] = adapted_embedding
            
            # Create the final list, placing embeddings at original indices, None elsewhere
            final_embeddings = [adapted_embeddings_map.get(i) for i in range(len(texts))]
            return final_embeddings

        except Exception as e:
            logger.error(f"Error generating batch embeddings using Gemini API: {e}")
            # Handle specific API errors (rate limits, etc.) if needed
            # Return None for all original texts on batch failure
            return [None] * len(texts)

    def find_relevant_studies(self, claim_text, top_k):
        """
        Finds top_k relevant studies using vector similarity search.
        Uses the single-text version of get_embedding_for_text for the claim.
        """
        if not gemini_api_key:
             logger.error("Gemini API key not available. Cannot perform vector search.")
             return []

        try:
            # 1. Embed the single claim text
            logger.info("Generating embedding for claim using Gemini API...")
            # Use the single-text embedding logic here for the claim
            # We need a way to get a single embedding. Let's assume we might need
            # a separate method or adapt the batch one for single use.
            # For now, let's call the batch method with a single item list.
            claim_embedding_list = self.get_embedding_for_text([claim_text], task_type="retrieval_query")
            
            if not claim_embedding_list or claim_embedding_list[0] is None:
                 logger.error("Failed to generate embedding for the claim.")
                 return []
            
            claim_embedding = claim_embedding_list[0]
            claim_embedding_np = np.array(claim_embedding).astype('float32')

            # 2. Perform vector similarity search
            logger.info(f"Searching database for top {top_k} studies...")
            relevant_studies = (
                self.db.query(Study)
                .filter(Study.embedding != None)
                .order_by(Study.embedding.cosine_distance(claim_embedding_np)) # Use the numpy array
                .limit(top_k)
                .all()
            )

            if not relevant_studies:
                logger.warning("Vector search returned no relevant studies.")
                return []

            logger.info(f"Retrieved {len(relevant_studies)} relevant studies from DB vector search.")
            return relevant_studies

        except Exception as e:
            logger.error(f"Error during vector search in database: {e}")
            return []

# --- End Vector Store Service ---

def clean_abstract(text):
    """Clean JATS XML tags and other formatting from abstract text."""
    if not text:
        return ""
    
    # Remove JATS XML tags
    text = re.sub(r'</?jats:[^>]+>', '', text)
    # Remove any remaining XML-like tags
    text = re.sub(r'</?[^>]+>', '', text)
    # Remove 'Abstract:' or 'Abstract' at the start
    text = re.sub(r'^(?:Abstract:?\s*)', '', text, flags=re.IGNORECASE)
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# --- New RAG Verification Service ---
class RAGVerificationService:
    def __init__(self, gemini_service, openalex_service, crossref_service, semantic_scholar_service, pubmed_service, db_session):
        self.gemini = gemini_service
        self.openalex = openalex_service
        self.crossref = crossref_service
        self.semantic_scholar = semantic_scholar_service
        self.pubmed = pubmed_service
        self.db = db_session
        self.vector_store = VectorStoreService(db_session) # Initialize vector store service

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
        preprocessing_result = self.gemini.preprocess_claim(claim)
        keywords = preprocessing_result.get("keywords", [])
        synonyms = preprocessing_result.get("synonyms", []) # Get synonyms
        # Use the generated boolean query as the primary search term
        search_query = preprocessing_result.get("boolean_query")
        category = preprocessing_result.get("category", "unknown")

        # Fallback logic if boolean query is bad
        if not search_query or len(search_query) < 5: # Check if query seems too short/invalid
            logger.warning(f"Generated boolean query '{search_query}' seems invalid. Falling back.")
            # Combine keywords and synonyms for a fallback query
            all_terms = list(set(keywords + synonyms)) # Unique terms
            if all_terms:
                search_query = " AND ".join(f'\\"{{term}}\\"' for term in all_terms if term) # Quote terms/phrases
                logger.info(f"Using fallback search query from keywords/synonyms: '{search_query}'")
            else:
                search_query = claim # Last resort fallback
                logger.info(f"Using original claim as fallback search query: '{search_query}'")


        if not search_query:
            logger.warning("No search query could be generated, cannot retrieve evidence.")
            return {"error": "Could not generate a search query from claim.", "status": "failed"}

        logger.info(f"Using Search Query: '{search_query}', Keywords: {keywords}, Synonyms: {synonyms}, Category: {category}") # Log synonyms too

        # 2. Define API Retrieval Configuration
        target_studies_per_api = 300  # Target minimum studies per API source

        # Get limits from config but respect minimum target
        openalex_limit = max(app.config['OPENALEX_MAX_RESULTS'], target_studies_per_api)
        crossref_limit = max(app.config['CROSSREF_MAX_RESULTS'], target_studies_per_api)
        semantic_scholar_limit = max(app.config['SEMANTIC_SCHOLAR_MAX_RESULTS'], target_studies_per_api)
        pubmed_limit = max(app.config['PUBMED_MAX_RESULTS'], target_studies_per_api)
        # --- New: Top K for DB vector search ---
        vector_search_db_top_k = app.config['RAG_TOP_K'] * 2 # Fetch more candidates from DB
        logger.info(f"DB Vector Search K: {vector_search_db_top_k}")
        # --- End New ---

        logger.info(f"Target study counts - OpenAlex: {openalex_limit}, CrossRef: {crossref_limit}, " +
                   f"Semantic Scholar: {semantic_scholar_limit}, PubMed: {pubmed_limit}")

        # Initialize results containers
        # all_studies_data = [] # This will be populated after API calls
        api_studies_data = [] # Store results *only* from APIs first
        openalex_studies = []
        crossref_studies = []
        semantic_scholar_studies = []
        pubmed_studies = []

        # 3. Retrieve Evidence from APIs - Using ThreadPoolExecutor for concurrent API calls
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit search tasks with pagination support using the generated search_query
            # OpenAlex (supports 200 per page max)
            def get_openalex_studies():
                results = []
                page = 1
                per_page = 200  # OpenAlex max per page
                while len(results) < openalex_limit and page <= 5:  # Limit to 5 pages max
                    try:
                        logger.info(f"Querying OpenAlex (page {page}) using query: {search_query}")
                        # Use the generated search_query
                        data = self.openalex.search_works_by_keyword(search_query, per_page=per_page, page=page)
                        if not data or not data.get('results'):
                            logger.info(f"OpenAlex returned no more results on page {page}")
                            break

                        batch = self.openalex.process_results(data)
                        if not batch:
                            logger.info(f"No processed results from OpenAlex on page {page}")
                            # Check if API returned results but processing failed
                            if data and data.get('results'):
                                logger.warning(f"OpenAlex API had results but processing yielded none on page {page}.")
                                # Potentially break here or just continue to next page? Continue for now.
                            else:
                                break # Break if API itself returned nothing

                        # Filter for abstracts (already done in process_results, but keep for clarity)
                        batch = [s for s in batch if s.get('abstract') and len(s.get('abstract', '')) >= 50]
                        logger.info(f"OpenAlex page {page}: Found {len(batch)} studies with usable abstracts")

                        results.extend(batch)
                        logger.info(f"Total OpenAlex studies with abstracts so far: {len(results)}/{openalex_limit}")

                        if len(batch) < per_page and data.get('meta', {}).get('count', 0) < per_page:  # More robust check for end
                             logger.info(f"OpenAlex likely finished (returned {len(batch)} < {per_page})")
                             break

                        page += 1
                    except Exception as e:
                        logger.error(f"Error retrieving data from OpenAlex (page {page}): {e}")
                        break

                logger.info(f"FINAL OpenAlex retrieval: {len(results)} studies with abstracts")
                return results

            # CrossRef (supports pagination)
            def get_crossref_studies():
                results = []
                rows = 200  # Increased batch size closer to API cap
                offset = 0
                while len(results) < crossref_limit and offset < 1000:  # Limit to 1000 results max
                    try:
                        logger.info(f"Querying CrossRef (offset {offset}, rows {rows}) using query: {search_query}")
                        # Use the generated search_query
                        data = self.crossref.search_works_by_keyword(search_query, rows=rows, offset=offset)
                        if not data or 'message' not in data or 'items' not in data['message'] or not data['message']['items']:
                            logger.info(f"CrossRef returned no more results at offset {offset}")
                            break

                        num_retrieved_api = len(data['message']['items'])
                        batch = self.crossref.process_results(data)
                        if not batch:
                             logger.info(f"No processed results from CrossRef at offset {offset} (retrieved {num_retrieved_api} raw)")
                             # Still increment offset if API returned something raw
                             if num_retrieved_api > 0:
                                 offset += rows
                                 continue
                             else:
                                 break # Break if API returned nothing raw

                        # Filter for abstracts (already done, keep for clarity)
                        batch = [s for s in batch if s.get('abstract') and len(s.get('abstract', '')) >= 50]
                        logger.info(f"CrossRef offset {offset}: Retrieved {num_retrieved_api} raw, Processed {len(batch)} studies with usable abstracts")

                        results.extend(batch)
                        logger.info(f"Total CrossRef studies with abstracts so far: {len(results)}/{crossref_limit}")

                        # Check if fewer items were returned than requested by API itself
                        if num_retrieved_api < rows:
                            logger.info(f"CrossRef API returned fewer items ({num_retrieved_api}) than requested ({rows}), indicating end of results.")
                            break

                        offset += rows
                    except Exception as e:
                        logger.error(f"Error retrieving data from CrossRef (offset {offset}): {e}")
                        break

                logger.info(f"FINAL CrossRef retrieval: {len(results)} studies with abstracts")
                return results

            # Semantic Scholar (supports pagination)
            def get_semantic_scholar_studies():
                results = []
                limit_per_call = 100  # Semantic Scholar max per API request
                current_offset = 0
                max_total_offset = 1000 # Safety limit for total results to check

                while len(results) < semantic_scholar_limit and current_offset < max_total_offset:
                    try:
                        logger.info(f"Querying Semantic Scholar (offset {current_offset}, limit {limit_per_call}) using query: {search_query}")
                        # Use the generated search_query
                        data = self.semantic_scholar.search_works_by_keyword(search_query, limit=limit_per_call, offset=current_offset)

                        # Check if the API call failed or returned unexpected data
                        if not data or 'data' not in data:
                            logger.error(f"Semantic Scholar call failed or returned invalid data at offset {current_offset}")
                            break # Stop if the API call itself fails

                        num_retrieved_api = len(data.get('data', []))
                        next_offset = data.get('next_offset') # Get the next offset from the service method return

                        # Process the results returned from this specific API call
                        batch = self.semantic_scholar.process_results(data)

                        if batch:
                            # Abstract filtering is already done in process_results
                            logger.info(f"Semantic Scholar offset {current_offset}: Retrieved {num_retrieved_api} raw, Processed {len(batch)} usable studies")
                            results.extend(batch)
                            logger.info(f"Total Semantic Scholar studies with abstracts so far: {len(results)}/{semantic_scholar_limit}")
                        else:
                            logger.info(f"No processed usable studies from Semantic Scholar at offset {current_offset} (Retrieved {num_retrieved_api} raw).")

                        # Decide whether to continue based on the API response
                        if num_retrieved_api < limit_per_call or next_offset is None or next_offset <= current_offset:
                            if num_retrieved_api < limit_per_call:
                                logger.info(f"Semantic Scholar API returned fewer items ({num_retrieved_api}) than requested ({limit_per_call}), indicating end of results.")
                            else:
                                logger.info(f"Stopping Semantic Scholar pagination loop (next_offset: {next_offset}, current_offset: {current_offset}).")
                            break

                        # Update offset for the next loop iteration
                        current_offset = next_offset
                        time.sleep(1)  # Avoid rate limiting

                    except Exception as e:
                        logger.error(f"Error retrieving data from Semantic Scholar (offset {current_offset}): {e}")
                        if "rate limit" in str(e).lower():
                            logger.warning("Semantic Scholar rate limit hit, waiting 5 seconds...")
                            time.sleep(5)
                            # Continue the loop to retry after waiting
                        else:
                            break # Break on other exceptions

                logger.info(f"FINAL Semantic Scholar retrieval: {len(results)} studies with abstracts")
                return results

            # PubMed (Rewritten to handle ESearch pagination correctly)
            def get_pubmed_studies():
                results = []
                retmax_esearch = 100  # ESearch batch size
                retstart = 0
                pmid_batch_size_efetch = 100 # EFetch batch size

                # --- PubMed Specific Query Adaptation ---
                # Use the generated search_query, but potentially adapt for PubMed syntax if needed
                # PubMed often benefits from explicit field tags like [Title/Abstract]
                # We'll assume the boolean_query from Gemini might be good enough,
                # but one could add logic here to reformat it (e.g., add [tiab])
                pubmed_search_term = search_query
                logger.info(f"Using PubMed search term: {pubmed_search_term}")
                # --- End PubMed Adaptation ---

                while len(results) < pubmed_limit and retstart < 1000: # Limit ESearch offset
                    try:
                        # 1. ESearch for PMIDs in the current window
                        logger.info(f"Querying PubMed ESearch (retstart {retstart}, retmax {retmax_esearch}): {pubmed_search_term}")
                        # Use the adapted pubmed_search_term
                        pmids_in_window = self.pubmed.search_works_by_keyword(pubmed_search_term, retmax=retmax_esearch, retstart=retstart)

                        if pmids_in_window is None: # Indicates an error occurred during ESearch
                            logger.error(f"PubMed ESearch failed at retstart {retstart}. Stopping PubMed retrieval.")
                            break

                        if not pmids_in_window:
                            logger.info(f"PubMed ESearch returned no more PMIDs at retstart {retstart}")
                            break # No more PMIDs found

                        num_pmids_found = len(pmids_in_window)
                        logger.info(f"PubMed ESearch found {num_pmids_found} PMIDs at retstart {retstart}")

                        # 2. Fetch and Process details for these PMIDs in batches
                        processed_count_in_window = 0
                        for i in range(0, num_pmids_found, pmid_batch_size_efetch):
                            batch_pmids = pmids_in_window[i:i+pmid_batch_size_efetch]
                            if not batch_pmids:
                                continue

                            logger.info(f"Fetching details for PubMed EFetch batch ({len(batch_pmids)} PMIDs from ESearch retstart {retstart})...")
                            xml_content = self.pubmed._fetch_article_details(batch_pmids)

                            if xml_content:
                                batch_details = self.pubmed._parse_pubmed_xml(xml_content)
                                # Abstract filtering is already done in _parse_pubmed_xml
                                if batch_details:
                                    results.extend(batch_details)
                                    processed_count_in_window += len(batch_details)
                                    logger.info(f"  Processed {len(batch_details)} usable studies from EFetch batch. Total PubMed: {len(results)}/{pubmed_limit}")
                                else:
                                    logger.info(f"  No usable studies found after processing EFetch batch.")
                            else:
                                logger.warning(f"  Failed to fetch details for PubMed EFetch batch (PMIDs starting with {batch_pmids[0]}).")

                            # Add a small delay between EFetch batches
                            if i + pmid_batch_size_efetch < num_pmids_found:
                                time.sleep(self.pubmed.request_delay / 2)

                        logger.info(f"Completed processing for ESearch window retstart {retstart}. Found {processed_count_in_window} usable studies.")

                        # Check if ESearch returned fewer PMIDs than requested, indicating end
                        if num_pmids_found < retmax_esearch:
                            logger.info(f"PubMed ESearch returned fewer PMIDs ({num_pmids_found}) than requested ({retmax_esearch}), indicating end of results.")
                            break

                        # Increment ESearch offset
                        retstart += retmax_esearch

                        # Check if we've already reached the overall limit
                        if len(results) >= pubmed_limit:
                            logger.info(f"Reached target PubMed limit ({pubmed_limit}). Stopping retrieval.")
                            break

                    except Exception as e:
                        logger.error(f"Error retrieving data from PubMed (retstart {retstart}): {e}")
                        break # Stop on error

                logger.info(f"FINAL PubMed retrieval: {len(results)} studies with abstracts")
                return results

            # Submit all API tasks concurrently
            future_openalex = executor.submit(get_openalex_studies)
            future_crossref = executor.submit(get_crossref_studies)
            future_semantic = executor.submit(get_semantic_scholar_studies)
            future_pubmed = executor.submit(get_pubmed_studies)

            # Collect results from APIs
            try:
                openalex_studies = future_openalex.result()
                logger.info(f"Retrieved {len(openalex_studies)} usable studies from OpenAlex")
            except Exception as e:
                logger.error(f"Error in OpenAlex retrieval process: {e}")

            try:
                crossref_studies = future_crossref.result()
                logger.info(f"Retrieved {len(crossref_studies)} usable studies from CrossRef")
            except Exception as e:
                logger.error(f"Error in CrossRef retrieval process: {e}")

            try:
                semantic_scholar_studies = future_semantic.result()
                logger.info(f"Retrieved {len(semantic_scholar_studies)} usable studies from Semantic Scholar")
            except Exception as e:
                logger.error(f"Error in Semantic Scholar retrieval process: {e}")

            try:
                pubmed_studies = future_pubmed.result()
                logger.info(f"Retrieved {len(pubmed_studies)} usable studies from PubMed")
            except Exception as e:
                logger.error(f"Error in PubMed retrieval process: {e}")

        # 4. Combine all studies from APIs
        api_studies_data = openalex_studies + crossref_studies + semantic_scholar_studies + pubmed_studies
        logger.info(f"TOTAL STUDIES WITH ABSTRACTS FROM APIs: {len(api_studies_data)}")
        logger.info(f"  - OpenAlex: {len(openalex_studies)}, CrossRef: {len(crossref_studies)}, SemSch: {len(semantic_scholar_studies)}, PubMed: {len(pubmed_studies)}")

        # --- New Step 4b: Explicit Vector Search Against DB ---
        logger.info(f"Performing explicit vector search against database (top_k={vector_search_db_top_k})...")
        vector_studies_data = []
        try:
            # Use the find_relevant_studies method from VectorStoreService
            db_vector_results = self.vector_store.find_relevant_studies(claim, top_k=vector_search_db_top_k)

            if db_vector_results:
                logger.info(f"Retrieved {len(db_vector_results)} studies from DB vector search.")
                # Convert Study objects back to dictionary format to match API results format for consistency
                for study in db_vector_results:
                    vector_studies_data.append({
                        "doi": study.doi,
                        "title": study.title,
                        "authors": study.authors,
                        "pub_date": study.pub_date,
                        "abstract": study.abstract,
                        "source_api": study.source_api + "_db_vector", # Mark source
                        "citation_count": study.citation_count or 0,
                        # We don't need to include the embedding itself here
                    })
            else:
                logger.info("DB vector search returned no results.")

        except Exception as e:
            logger.error(f"Error during explicit DB vector search: {e}")
            # Continue without these results if the search fails

        # --- End New Step 4b ---


        # 5. Combine API results and DB Vector Search results
        combined_studies_data = api_studies_data + vector_studies_data
        logger.info(f"TOTAL STUDIES BEFORE DEDUPLICATION (API + DB Vector): {len(combined_studies_data)}")

        # 6. Deduplicate studies (on the combined list)
        seen_dois = set()
        seen_titles = {} # Use dict to store title -> doi (or None if no DOI)
        unique_studies_data = []
        duplicates_removed_count = 0

        for study_data in combined_studies_data:
            doi = study_data.get('doi')
            title = study_data.get('title')
            doi_norm = doi.lower().strip() if doi else None
            title_lower = title.lower().strip() if title else None
            is_duplicate = False

            # Check DOI first
            if doi_norm:
                if doi_norm in seen_dois:
                    is_duplicate = True
                else:
                    seen_dois.add(doi_norm)
                    # Also add title associated with this DOI to prevent title-based dupes later
                    if title_lower:
                        seen_titles[title_lower] = doi_norm
            # If no DOI, check title
            elif title_lower:
                if title_lower in seen_titles:
                    is_duplicate = True
                else:
                    seen_titles[title_lower] = None # Mark title as seen, without DOI

            if not is_duplicate:
                unique_studies_data.append(study_data)
            else:
                duplicates_removed_count += 1

        logger.info(f"TOTAL UNIQUE STUDIES AFTER DEDUPLICATION: {len(unique_studies_data)} (Removed {duplicates_removed_count} duplicates)")


        # 7. Limit total studies to process
        max_evidence = app.config['MAX_EVIDENCE_TO_STORE']
        studies_to_process_data = unique_studies_data[:max_evidence]
        logger.info(f"Using {len(studies_to_process_data)} studies for analysis (max limit: {max_evidence})")

        if not studies_to_process_data:
            logger.warning("No usable evidence found after API retrieval, DB search, and filtering.")
            # Return standard 'inconclusive' response
            return {
                "claim": claim,
                "verdict": "Inconclusive",
                "reasoning": "No relevant academic studies with abstracts could be retrieved and processed.",
                "detailed_reasoning": "No relevant academic studies with abstracts could be retrieved and processed.",
                "simplified_reasoning": "No relevant academic studies could be found to analyze this claim.",
                "accuracy_score": 0.0,
                "evidence": [],
                "keywords_used": keywords, # Use keywords from preprocessing
                "category": category, # Use category from preprocessing
                "processing_time_seconds": round(time.time() - start_time, 2)
            }


        # 8. Store Evidence & Embed NEW Studies Immediately
        # (This section remains largely the same, operating on studies_to_process_data)
        stored_studies = []
        new_studies_to_create_data = [] # Store data dicts for new studies
        existing_study_objects = [] # Store existing Study objects found
        success_count = 0 # Initialize counters here
        error_count = 0   # Initialize counters here

        try:
            # Find existing studies among the ones selected for processing
            study_dois_to_check = [s.get('doi').lower() for s in studies_to_process_data if s.get('doi')]
            existing_dois_in_db_map = {} # Define it here

            if study_dois_to_check:
                 existing_studies_in_db = self.db.query(Study).filter(Study.doi.in_(study_dois_to_check)).all()
                 # --- Enhanced DOI Normalization Logic ---
                 # Create a more comprehensive lookup map that includes multiple variations of DOIs
                 raw_dois_in_db = {study.doi for study in existing_studies_in_db if study.doi}
                 # Create normalized map of DOIs -> Study objects
                 existing_dois_in_db_map = {study.doi.lower().strip(): study 
                                          for study in existing_studies_in_db if study.doi}
                 # Add more variations to catch different DOI formats
                 for study in existing_studies_in_db:
                     if not study.doi:
                         continue
                     # Original DOI
                     raw_doi = study.doi
                     # Normalized (lowercase, stripped)
                     norm_doi = raw_doi.lower().strip()
                     # Without https://doi.org/ prefix
                     if norm_doi.startswith('https://doi.org/'):
                         plain_doi = norm_doi[16:]
                         existing_dois_in_db_map[plain_doi] = study
                     # With https://doi.org/ prefix if not present
                     elif not norm_doi.startswith('https://doi.org/'):
                         prefixed_doi = 'https://doi.org/' + norm_doi
                         existing_dois_in_db_map[prefixed_doi] = study
                     # Without trailing slash if present
                     if norm_doi.endswith('/'):
                         existing_dois_in_db_map[norm_doi[:-1]] = study
                     # With trailing slash if not present
                     elif not norm_doi.endswith('/'):
                         existing_dois_in_db_map[norm_doi + '/'] = study
                     # DOI with 'doi:' prefix
                     if not norm_doi.startswith('doi:'):
                         doi_prefixed = 'doi:' + norm_doi
                         existing_dois_in_db_map[doi_prefixed] = study
                     # DOI without 'doi:' prefix if present
                     elif norm_doi.startswith('doi:'):
                         doi_unprefixed = norm_doi[4:]
                         existing_dois_in_db_map[doi_unprefixed] = study
                     # Additional common DOI variations
                     # Handle http:// variant
                     if norm_doi.startswith('https://doi.org/'):
                         http_variant = 'http://doi.org/' + norm_doi[16:]
                         existing_dois_in_db_map[http_variant] = study
                     # Handle doi.org without protocol
                     if norm_doi.startswith('https://doi.org/') or norm_doi.startswith('http://doi.org/'):
                         no_protocol = 'doi.org/' + (norm_doi[16:] if norm_doi.startswith('https://') else norm_doi[14:])
                         existing_dois_in_db_map[no_protocol] = study
                     # Check for various capitalization of DOI in prefixes
                     if norm_doi.startswith('doi:'):
                         existing_dois_in_db_map['DOI:' + norm_doi[4:]] = study
                 # --- End Enhanced DOI Normalization ---

                 logger.info(f"Found {len(existing_studies_in_db)} of the selected studies already in database.")
                 logger.info(f"Created lookup map with {len(existing_dois_in_db_map)} DOI variations for duplicate checking.")
                 # Ensure uniqueness of study objects, even if multiple DOIs map to the same one
                 existing_study_objects = list(set(existing_dois_in_db_map.values())) 
                 logger.info(f"Tracking {len(existing_study_objects)} unique existing study objects.")

            # Separate new study data from existing ones
            logger.info("Separating new studies from existing ones...")
            new_duplicates_found = 0  # Counter for newly detected duplicates
            for study_data in studies_to_process_data:
                doi = study_data.get('doi')
                doi_norm = doi.lower().strip() if doi else None
                
                # Enhanced duplicate check with more variations
                is_duplicate = False
                if doi_norm:
                    # Check the direct normalized DOI
                    if doi_norm in existing_dois_in_db_map:
                        is_duplicate = True
                    else:
                        # Check with/without https://doi.org/ prefix
                        plain_doi = doi_norm[16:] if doi_norm.startswith('https://doi.org/') else doi_norm
                        prefixed_doi = 'https://doi.org/' + plain_doi
                        
                        if plain_doi in existing_dois_in_db_map or prefixed_doi in existing_dois_in_db_map:
                            is_duplicate = True
                        else:
                            # Check with/without trailing slash
                            with_slash = doi_norm + '/' if not doi_norm.endswith('/') else doi_norm
                            without_slash = doi_norm[:-1] if doi_norm.endswith('/') else doi_norm
                            
                            if with_slash in existing_dois_in_db_map or without_slash in existing_dois_in_db_map:
                                is_duplicate = True
                            else:
                                # Check doi: prefix variations
                                doi_prefixed = 'doi:' + plain_doi if not doi_norm.startswith('doi:') else doi_norm
                                doi_unprefixed = doi_norm[4:] if doi_norm.startswith('doi:') else plain_doi
                                
                                if doi_prefixed in existing_dois_in_db_map or doi_unprefixed in existing_dois_in_db_map:
                                    is_duplicate = True
                
                if is_duplicate:
                    new_duplicates_found += 1
                    continue
                else:
                    # This is potentially a new study, add its data dict
                    new_studies_to_create_data.append(study_data)
            
            if new_duplicates_found > 0:
                logger.info(f"Enhanced DOI normalization detected {new_duplicates_found} additional duplicates during processing.")
            logger.info(f"Identified {len(new_studies_to_create_data)} potential new studies to process.")

            # 9. CONCURRENT EMBEDDING FOR NEW STUDIES
            new_study_objects_with_embeddings = []
            if new_studies_to_create_data:
                # Collect texts and track original data index for mapping back
                texts_to_embed = []
                original_indices_map = {} # {index_in_texts_to_embed: index_in_new_studies_to_create_data}
                studies_needing_embedding_count = 0

                for idx, study_data in enumerate(new_studies_to_create_data):
                    if study_data.get('abstract'):
                        texts_to_embed.append(study_data['abstract'])
                        original_indices_map[len(texts_to_embed) - 1] = idx
                        studies_needing_embedding_count += 1

                logger.info(f"Starting concurrent embedding for {studies_needing_embedding_count} new studies with abstracts...")
                embeddings = [None] * len(new_studies_to_create_data) # Initialize list for embeddings

                if texts_to_embed:
                    embedding_start_time = time.time()
                    try:
                        # Use the batch embedding function directly
                        # Assuming get_embedding_for_text handles its own internal batching and returns a list
                        # matching the order of texts_to_embed
                        batch_embeddings = self.vector_store.get_embedding_for_text(
                            texts_to_embed,
                            task_type="retrieval_document"
                        )

                        # Map results back using original_indices_map
                        if batch_embeddings and len(batch_embeddings) == len(texts_to_embed):
                            for embed_idx, embedding in enumerate(batch_embeddings):
                                original_data_idx = original_indices_map.get(embed_idx)
                                if original_data_idx is not None:
                                    if embedding:
                                        embeddings[original_data_idx] = embedding
                                        grand_total_studies_embedded += 1
                                    else:
                                        logger.warning(f"Embedding failed for new study data at index {original_data_idx} (DOI: {new_studies_to_create_data[original_data_idx].get('doi')}).")
                        else:
                            logger.error(f"Concurrent embedding call failed or returned mismatched results ({len(batch_embeddings) if batch_embeddings else 0} results for {len(texts_to_embed)} texts).")

                    except Exception as embed_err:
                        logger.error(f"Error during concurrent embedding API call: {embed_err}")
                        # Embeddings list remains mostly None

                    embedding_duration = time.time() - embedding_start_time
                    # Estimate tokens based on embedded count - crude but better than nothing
                    # A more accurate way would be to modify get_embedding_for_text to return tokens
                    estimated_tokens = sum(num_tokens_from_string(txt) for txt, emb in zip(texts_to_embed, batch_embeddings) if emb)
                    grand_total_tokens_embedded = estimated_tokens
                    logger.info(f"Concurrent embedding finished in {embedding_duration:.2f}s. Successfully embedded: {grand_total_studies_embedded}/{studies_needing_embedding_count}. Estimated Tokens: {estimated_tokens}")

                # --- FIX: Deduplicate new study data before creating objects ---
                unique_new_study_data_map = {}
                deduplicated_new_studies_data = []
                original_embeddings_map = {idx: emb for idx, emb in enumerate(embeddings)} # Map original index to embedding

                logger.info(f"Deduplicating {len(new_studies_to_create_data)} potential new studies before object creation...")
                new_data_duplicates_removed = 0
                for idx, study_data in enumerate(new_studies_to_create_data):
                    doi = study_data.get('doi')
                    doi_norm = doi.lower().strip() if doi else None
                    # --- Safely handle potential None title --- 
                    title = study_data.get('title') # Get title, could be None
                    title_lower = title.lower().strip() if title else None # Only call lower() if title is not None
                    # --- End safe handling ---
                    is_duplicate = False
 
                    if doi_norm:
                        if doi_norm in unique_new_study_data_map:
                            is_duplicate = True
                        else:
                            unique_new_study_data_map[doi_norm] = idx # Store original index
                    elif title_lower: # Fallback to title if no DOI and title exists
                        if title_lower in unique_new_study_data_map:
                            is_duplicate = True
                        else:
                            unique_new_study_data_map[title_lower] = idx # Store original index
 
                    if not is_duplicate:
                        # Keep the study data AND its corresponding embedding (using original index)
                        study_data['_original_embedding'] = original_embeddings_map.get(idx)
                        deduplicated_new_studies_data.append(study_data)
                    else:
                        new_data_duplicates_removed += 1
                logger.info(f"Removed {new_data_duplicates_removed} duplicates from new study list. Processing {len(deduplicated_new_studies_data)} unique new studies.")
                # --- END FIX ---

                # Create Study objects for all new studies, adding embeddings where available
                logger.info("Creating new Study objects...")
                # Use the deduplicated list now
                for study_data in deduplicated_new_studies_data:
                    try:
                        source_api_cleaned = study_data.get('source_api', '').replace('_db_vector', '')
                        doi_raw = study_data.get('doi')
                        doi_cleaned = doi_raw.strip() if doi_raw else None

                        study_obj = Study(
                            doi=doi_cleaned,
                            title=study_data.get('title'),
                            authors=study_data.get('authors'),
                            pub_date=study_data.get('pub_date'),
                            abstract=study_data.get('abstract'),
                            source_api=source_api_cleaned,
                            citation_count=study_data.get('citation_count', 0),
                            embedding=study_data['_original_embedding'] # Assign the fetched embedding (or None)
                        )
                        new_study_objects_with_embeddings.append(study_obj)
                        success_count += 1 # Count successful object creation attempt
                    except Exception as e:
                         logger.error(f"Error creating Study object for data index {idx} (DOI: {study_data.get('doi')}): {e}")
                         error_count += 1

            # 10. ADD AND COMMIT NEW STUDIES
            if new_study_objects_with_embeddings:
                logger.info(f"Adding {len(new_study_objects_with_embeddings)} new studies to the database session...")
                try:
                    self.db.add_all(new_study_objects_with_embeddings)
                    logger.info("Committing new studies...")
                    commit_start_time = time.time()
                    self.db.commit()
                    commit_duration = time.time() - commit_start_time
                    logger.info(f"Successfully committed {len(new_study_objects_with_embeddings)} new studies in {commit_duration:.2f}s.")
                    # Refresh objects to ensure they have IDs etc. (Optional, depends on subsequent access needs)
                    # for obj in new_study_objects_with_embeddings:
                    #    self.db.refresh(obj)
                except SQLAlchemyError as e:
                    self.db.rollback()
                    error_msg = str(e)
                    logger.error(f"Database error committing new studies: {error_msg}. Rolling back transaction.")
                    
                    # --- FALLBACK: Try smaller batches if we hit a UniqueViolation ---
                    if "UniqueViolation" in error_msg and "doi" in error_msg:
                        logger.warning("Detected UniqueViolation error with DOI. Trying to commit in smaller batches...")
                        # Parse which DOI caused the issue to help with debugging
                        try:
                            error_doi_match = re.search(r'Key \(doi\)=\((.*?)\)', error_msg)
                            error_doi = error_doi_match.group(1) if error_doi_match else "unknown"
                            logger.warning(f"Conflicting DOI from error: {error_doi}")
                        except Exception:
                            error_doi = "unknown"
                            
                        # Try committing in smaller batches, skip any DOIs already in the database
                        batch_size = 50
                        committed_count = 0
                        skip_count = 0
                        error_batch_count = 0
                        successfully_committed_studies = [] # List to track successful commits
                        
                        # Get all DOIs currently in the database to skip them
                        try:
                            existing_dois_query = self.db.query(Study.doi).filter(Study.doi != None).all()
                            skip_dois_set = {doi[0].lower().strip() for doi in existing_dois_query if doi[0]}
                            
                            # Add variations of DOIs to the skip set
                            expanded_skip_dois = set()
                            for doi in skip_dois_set:
                                expanded_skip_dois.add(doi)
                                # Without https://doi.org/ prefix
                                if doi.startswith('https://doi.org/'):
                                    expanded_skip_dois.add(doi[16:])
                                # With https://doi.org/ prefix
                                else:
                                    expanded_skip_dois.add('https://doi.org/' + doi)
                            skip_dois_set = expanded_skip_dois
                            
                            logger.info(f"Found {len(skip_dois_set)} DOIs to skip during batch commits")
                        except Exception as query_err:
                            logger.error(f"Error querying existing DOIs: {query_err}")
                            skip_dois_set = set()
                        
                        # Process studies in batches
                        initial_object_count = len(new_study_objects_with_embeddings)
                        for i in range(0, initial_object_count, batch_size):
                            batch = new_study_objects_with_embeddings[i:i+batch_size]
                            studies_to_add = []
                            
                            # Filter out studies with DOIs that already exist
                            for study in batch:
                                if not study.doi:
                                    # Studies without DOIs are usually safe to add
                                    studies_to_add.append(study)
                                else:
                                    doi_lower = study.doi.lower().strip()
                                    doi_plain = doi_lower[16:] if doi_lower.startswith('https://doi.org/') else doi_lower
                                    doi_prefixed = f"https://doi.org/{doi_plain}" if not doi_lower.startswith('https://doi.org/') else doi_lower
                                    
                                    # Skip if any variant is in the skip set
                                    if doi_lower in skip_dois_set or doi_plain in skip_dois_set or doi_prefixed in skip_dois_set:
                                        skip_count += 1
                                    else:
                                        studies_to_add.append(study)
                                        # Add to skip set to avoid duplicates in future batches
                                        skip_dois_set.add(doi_lower)
                                        skip_dois_set.add(doi_plain)
                                        skip_dois_set.add(doi_prefixed)
                            
                            if not studies_to_add:
                                logger.info(f"Batch {i//batch_size + 1}: No studies to add after filtering")
                                continue
                                
                            try:
                                # Add and commit this batch
                                self.db.add_all(studies_to_add)
                                self.db.commit()
                                committed_count += len(studies_to_add)
                                successfully_committed_studies.extend(studies_to_add) # Track successful ones
                                logger.info(f"Batch {i//batch_size + 1}: Successfully committed {len(studies_to_add)} studies (total: {committed_count})")
                            except SQLAlchemyError as batch_err:
                                self.db.rollback()
                                batch_err_msg = str(batch_err)
                                logger.warning(f"Error in batch {i//batch_size + 1}: {batch_err_msg[:100]}...")
                                error_batch_count += 1
                                
                                # Last resort: Try one by one
                                if error_batch_count <= 2:  # Only try individual commits for the first few batches that fail
                                    logger.info(f"Attempting individual commits for batch {i//batch_size + 1}...")
                                    for study in studies_to_add:
                                        try:
                                            # Check again if DOI exists before adding individually
                                            should_add = True
                                            if study.doi:
                                                doi_l = study.doi.lower().strip()
                                                doi_p = doi_l[16:] if doi_l.startswith('https://doi.org/') else doi_l
                                                if doi_l in skip_dois_set or doi_p in skip_dois_set:
                                                    should_add = False
                                                    skip_count += 1 # Increment skip count here
                                            
                                            if should_add:
                                                self.db.add(study)
                                                self.db.commit()
                                                committed_count += 1
                                                successfully_committed_studies.append(study) # Track successful one
                                                
                                                # Add to skip set if it had a DOI
                                                if study.doi:
                                                    doi_l = study.doi.lower().strip()
                                                    doi_p = doi_l[16:] if doi_l.startswith('https://doi.org/') else doi_l
                                                    skip_dois_set.add(doi_l)
                                                    skip_dois_set.add(doi_p)
                                        except SQLAlchemyError:
                                            self.db.rollback()
                                            # This study failed even individually, count as skip/error
                                            # skip_count += 1 # Already counted if duplicate DOI was found
                                            pass # Move to next study
                                else:
                                    # Batches failed, and we are not trying individually anymore
                                    # Count remaining in studies_to_add as skips/errors for this batch
                                    skip_count += len(studies_to_add) 
                        
                        # Update error count and the list of new objects
                        logger.info(f"Batch commit results: {committed_count} studies committed, {skip_count} skipped")
                        success_count = committed_count # Update overall success count
                        error_count = initial_object_count - committed_count # Calculate errors correctly
                        new_study_objects_with_embeddings = successfully_committed_studies # Use the tracked list
                        logger.info(f"Updated new_study_objects_with_embeddings with {len(new_study_objects_with_embeddings)} successfully committed studies.")

                    else:
                        # Not a UniqueViolation or not DOI-related
                        logger.error(f"Database error not related to DOI duplicates: {error_msg}")
                        error_count += len(new_study_objects_with_embeddings)
                        new_study_objects_with_embeddings = []
                except Exception as e:
                    self.db.rollback()
                    logger.error(f"Unexpected error committing new studies: {e}. Rolling back transaction.")
                    error_count += len(new_study_objects_with_embeddings) # Count these as errors
                    new_study_objects_with_embeddings = [] # Clear list

            # 11. GATHER FINAL LIST FOR RANKING
            # Combine existing objects and successfully committed new objects
            stored_studies = existing_study_objects + new_study_objects_with_embeddings
            logger.info(f"Total studies available for ranking: {len(stored_studies)} ({len(existing_study_objects)} existing, {len(new_study_objects_with_embeddings)} new)")

            logger.info(f"DB/PRE-RANKING PHASE SUMMARY: Processed {success_count} studies data, {error_count} errors creating objects/committing.")
            logger.info(f"EMBEDDING SUMMARY: Embedded {grand_total_studies_embedded} new studies, {grand_total_tokens_embedded} tokens (estimated).")

        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error during evidence storage/embedding phase: {e}")
            return {"error": "Database error storing/embedding evidence.", "status": "failed"}
        except Exception as e:
            self.db.rollback()
            logger.error(f"Unexpected error during evidence storage/embedding phase: {e}")
            return {"error": f"Unexpected error: {str(e)}", "status": "failed"}


        # 12. Rank Studies (using the final 'stored_studies' list which contains DB objects)
        # (This logic remains the same, but operates on the potentially larger/more relevant pool)
        logger.info(f"Ranking {len(stored_studies)} studies for RAG analysis...")

        # Get claim embedding (ensure it's done only once)
        claim_embedding_np = None # Initialize
        try:
            claim_embedding_list = self.vector_store.get_embedding_for_text([claim], task_type="retrieval_query")
            if not claim_embedding_list or claim_embedding_list[0] is None:
                logger.error("Failed to generate embedding for the claim (returned None). Cannot rank by relevance.")
                # Fallback: Rank by citation only? Or return error? Return error for now.
                return {"error": "Failed to generate claim embedding for ranking.", "status": "failed"}

            claim_embedding = claim_embedding_list[0]
            expected_dimension = app.config['EMBEDDING_DIMENSION']
            actual_claim_dimension = len(claim_embedding)
            logger.info(f"Claim embedding generated with dimension: {actual_claim_dimension}")

            if actual_claim_dimension != expected_dimension:
                 logger.error(f"CRITICAL: Claim embedding dimension ({actual_claim_dimension}) does not match expected ({expected_dimension})! Cannot rank by relevance.")
                 return {"error": f"Claim embedding dimension mismatch ({actual_claim_dimension} vs {expected_dimension}). Cannot rank.", "status": "failed"}
            else:
                 claim_embedding_np = np.array(claim_embedding).astype('float32')

        except Exception as e:
            logger.error(f"Error generating claim embedding: {e}")
            return {"error": "Failed to generate claim embedding.", "status": "failed"}


        # Rank studies
        ranked_studies_with_scores = []
        studies_with_valid_embeddings_count = 0
        for study in stored_studies:
            relevance_score = 0.0
            # Check if study has a valid embedding matching the expected dimension
            study_embedding_np = None
            study_has_valid_embedding = False
            if study.embedding is not None and len(study.embedding) == expected_dimension:
                study_embedding_np = np.array(study.embedding).astype('float32')
                study_has_valid_embedding = True
                studies_with_valid_embeddings_count += 1 # Count studies that *could* have relevance calculated

            # Calculate relevance only if both claim and study have valid embeddings
            if claim_embedding_np is not None and study_has_valid_embedding:
                try:
                    dot_product = np.dot(claim_embedding_np, study_embedding_np)
                    norm_claim = np.linalg.norm(claim_embedding_np)
                    norm_study = np.linalg.norm(study_embedding_np)

                    if norm_claim > 0 and norm_study > 0:
                        similarity = dot_product / (norm_claim * norm_study)
                        relevance_score = max(0.0, min(1.0, similarity)) # Clamp [0, 1]
                    else:
                         logger.warning(f"Zero norm encountered for claim or study {study.doi or study.id}. Relevance set to 0.")

                except Exception as e:
                    logger.warning(f"Error calculating relevance for study {study.doi or study.id}: {e}")
                    # relevance_score remains 0.0
            elif claim_embedding_np is None:
                 # This case should be handled by the error check above, but log if it occurs
                 logger.warning("Claim embedding is None during ranking phase.")
            elif study.embedding is None:
                 # logger.debug(f"Study {study.doi or study.id} has no embedding, relevance is 0.")
                 pass # Expected if embedding failed
            else: # Embedding exists but wrong dimension
                 logger.warning(f"Study {study.doi or study.id} embedding dim {len(study.embedding)} != expected {expected_dimension}. Relevance is 0.")


            # Credibility score (log scale citation count)
            citation_count = study.citation_count if study.citation_count is not None else 0
            try:
                 # Add offset +1 before log10 to handle 0 citations gracefully (log10(1)=0)
                 # Add another small epsilon to handle potential float issues near 0? No, log10(1) is fine.
                 credibility_score = math.log10(citation_count + 1)
            except ValueError: # Handle negative counts
                 logger.warning(f"Invalid citation count ({study.citation_count}) for study {study.doi or study.id}. Setting credibility to 0.")
                 credibility_score = 0.0

            # Combine scores with weights
            # Give more weight to relevance *only if* it could be calculated
            relevance_weight = 0.7 if study_has_valid_embedding and relevance_score > 0 else 0.1
            credibility_weight = 1.0 - relevance_weight

            combined_score = (relevance_weight * relevance_score) + (credibility_weight * credibility_score)
            ranked_studies_with_scores.append((study, combined_score, relevance_score, credibility_score)) # Store individual scores too


        # Sort and select top K
        ranked_studies_with_scores.sort(key=lambda item: item[1], reverse=True)
        top_k = app.config['RAG_TOP_K']
        top_ranked_studies_with_scores = ranked_studies_with_scores[:top_k]
        top_ranked_studies = [study for study, score, rel, cred in top_ranked_studies_with_scores]
        relevant_chunks = [study.abstract for study in top_ranked_studies if study.abstract]

        logger.info(f"Selected top {len(top_ranked_studies)} ranked studies for LLM analysis from {len(stored_studies)} candidates.")
        # Log score distribution of top K
        if top_ranked_studies_with_scores:
             top_score = top_ranked_studies_with_scores[0][1]
             median_index = len(top_ranked_studies_with_scores) // 2
             median_score = top_ranked_studies_with_scores[median_index][1]
             bottom_score = top_ranked_studies_with_scores[-1][1]
             avg_relevance = sum(s[2] for s in top_ranked_studies_with_scores) / len(top_ranked_studies_with_scores)
             avg_credibility = sum(s[3] for s in top_ranked_studies_with_scores) / len(top_ranked_studies_with_scores)
             logger.info(f"  Top K Score Range: {bottom_score:.3f} (Bottom) - {median_score:.3f} (Median) - {top_score:.3f} (Top)")
             logger.info(f"  Top K Avg Relevance (0-1): {avg_relevance:.3f}, Avg Credibility (Log10(Cit+1)): {avg_credibility:.3f}")


        if not relevant_chunks:
             logger.warning("No abstracts available from top ranked studies for RAG input.")
             # Return standard 'inconclusive' response
             return {
                "claim": claim,
                "verdict": "Inconclusive",
                "reasoning": "Relevant studies were found, but none had abstracts suitable for analysis.",
                "detailed_reasoning": "Relevant studies were found, but none had abstracts suitable for analysis.",
                "simplified_reasoning": "Could not find study details needed to analyze the claim.",
                "accuracy_score": 0.0,
                "evidence": [], # Provide empty evidence list
                "keywords_used": keywords,
                "category": category,
                "processing_time_seconds": round(time.time() - start_time, 2)
             }

        # 10. Analyze with LLM
        analysis_result = self.gemini.analyze_with_rag(claim, relevant_chunks)

        # 11. Format and Return Output
        evidence_details = []
        for idx, (study, score, rel, cred) in enumerate(top_ranked_studies_with_scores):
            if study.abstract:
                evidence_details.append({
                    "id": idx + 1, # Rank in the final list
                    "title": study.title,
                    "abstract": study.abstract,
                    "authors": study.authors,
                    "doi": study.doi,
                    "pub_date": study.pub_date,
                    "source_api": study.source_api, # Original source
                    "citation_count": study.citation_count or 0,
                    # --- Convert NumPy floats to Python floats --- 
                    "combined_score": round(float(score), 4), # Include ranking score
                    "relevance_score": round(float(rel), 4),
                    "credibility_score": round(float(cred), 4)
                    # --- End Conversion --- 
                })


        # Calculate token counts for better reporting
        total_chunk_tokens = sum(num_tokens_from_string(chunk) for chunk in relevant_chunks)
        avg_token_per_chunk = total_chunk_tokens / len(relevant_chunks) if relevant_chunks else 0
        total_process_time = time.time() - start_time

        # Count studies with embeddings among *all* processed studies
        studies_with_embeddings = sum(1 for study in stored_studies if study.embedding is not None and len(study.embedding) == expected_dimension)
        studies_without_embeddings = len(stored_studies) - studies_with_embeddings

        # Print a clear, comprehensive summary
        logger.info("="*80)
        logger.info("📊 CLAIM VERIFICATION SUMMARY 📊")
        logger.info("-"*80)
        logger.info("🔍 STUDY COLLECTION STATS:")
        logger.info(f"  • API Studies Found: {len(api_studies_data)}")
        logger.info(f"    - OpenAlex: {len(openalex_studies)}, CrossRef: {len(crossref_studies)}, SemSch: {len(semantic_scholar_studies)}, PubMed: {len(pubmed_studies)}")
        logger.info(f"  • DB Vector Search Found: {len(vector_studies_data)} (Target K={vector_search_db_top_k})")
        logger.info(f"  • Total Combined Before Dedup: {len(combined_studies_data)}")
        logger.info(f"  • Studies after Deduplication: {len(unique_studies_data)} (Removed {duplicates_removed_count})")
        logger.info(f"  • Studies limited to processing cap: {len(studies_to_process_data)} (MAX_EVIDENCE_TO_STORE: {max_evidence})")

        logger.info("💾 DATABASE & EMBEDDING STATS:")
        logger.info(f"  • DB Records Processed (Add/Update attempts): {success_count}")
        logger.info(f"  • Processing errors: {error_count}")
        logger.info(f"  • New studies embedded concurrently: {grand_total_studies_embedded}")
        logger.info(f"  • Total embedding tokens (estimated): {grand_total_tokens_embedded}")
        if grand_total_studies_embedded > 0:
            logger.info(f"  • Average tokens per embedded study: {grand_total_tokens_embedded/grand_total_studies_embedded:.1f}")

        logger.info("🔬 RANKING & RAG ANALYSIS STATS:") # Renamed slightly
        logger.info(f"  • Candidate studies for ranking: {len(stored_studies)}")
        studies_with_valid_embeddings_count = sum(1 for s in stored_studies if s.embedding is not None and len(s.embedding) == expected_dimension)
        percentage_with_embeddings = (studies_with_valid_embeddings_count / len(stored_studies) * 100) if stored_studies else 0.0
        logger.info(f"  • Candidates with valid embeddings for ranking: {studies_with_valid_embeddings_count}/{len(stored_studies)} ({percentage_with_embeddings:.1f}%)")
        logger.info(f"  • Top-K studies selected for RAG: {len(top_ranked_studies)} (configured as RAG_TOP_K: {top_k})")
        logger.info(f"  • Total tokens in RAG input chunks: {total_chunk_tokens}")
        logger.info(f"  • Average tokens per RAG chunk: {avg_token_per_chunk:.1f}")

        logger.info("📋 RESULTS:")
        logger.info(f"  • Verdict: {analysis_result.get('verdict', 'Error')}")
        logger.info(f"  • Accuracy Score: {analysis_result.get('accuracy_score', 0.0)}")
        logger.info(f"  • Processing Time: {total_process_time:.2f} seconds")
        logger.info("="*80)

        final_response = {
            "claim": claim,
            "verdict": analysis_result.get("verdict", "Error"),
            "reasoning": analysis_result.get("reasoning", "Analysis failed."), # Keep for compatibility if needed
            "detailed_reasoning": analysis_result.get("detailed_reasoning", analysis_result.get("reasoning", "Analysis failed.")),
            "simplified_reasoning": analysis_result.get("simplified_reasoning", analysis_result.get("reasoning", "Analysis failed.")),
            "accuracy_score": analysis_result.get("accuracy_score", 0.0),
            "evidence": evidence_details,
            "keywords_used": keywords, # Original keywords
            "synonyms_identified": synonyms, # Add synonyms here
            "search_query_used": search_query, # The query used for APIs
            "category": category,
            "processing_time_seconds": round(total_process_time, 2)
        }

        return final_response

# --- End Modified RAG Verification Service ---

# Initialize services
openalex_service = OpenAlexService(config=app.config)
crossref_service = CrossRefService(config=app.config)
semantic_scholar_service = SemanticScholarService(config=app.config)
gemini_service = GeminiService(gemini_model)
pubmed_service = PubMedService(config=app.config, clean_abstract_func=clean_abstract) # Initialize PubMed service

# Routes
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    db_status = "disconnected"
    try:
        connection = engine.connect()
        connection.close()
        db_status = "connected"
    except Exception as e:
        logger.error(f"Database connection failed: {e}")

    # --- REMOVE Redis Check ---
    # redis_status = "disconnected" ...
    # --- End REMOVE Redis Check ---

    gemini_status = "ok" if gemini_model else "unavailable"
    # Change embedding status check - API based now
    embedding_status = "ok" if gemini_api_key else "api_key_missing"
    vector_db_status = db_status # Still tied to main DB

    return jsonify({
        "status": "ok",
        "service": "Factify RAG API",
        "version": "2.2.0", # Update version
        "dependencies": {
            "database": db_status,
            "gemini_model": gemini_status,
            "embedding_api": embedding_status, # Changed from embedding_model
            "vector_storage": vector_db_status,
            # REMOVE "redis_queue": redis_status,
            "openalex_api": "ok",
            "crossref_api": "ok",
            "semantic_scholar_api": "ok", # Assuming ok
            "pubmed_api": "ok" # Assuming ok
        }
    })

# --- Updated Claim Verification Endpoint (Logic inside RAG service is changed) ---
@app.route('/api/verify_claim', methods=['POST'])
def verify_claim_rag():
    """Verifies a claim using RAG with immediate API embedding."""
    try:
        # Parse request JSON data
        data = request.get_json()
        
        if not data or 'claim' not in data:
            return jsonify({
                "error": "Missing 'claim' in request body",
                "status": "failed"
            }), 400
            
        claim = data['claim']
        
        if not claim or not isinstance(claim, str) or len(claim.strip()) < 5:
            return jsonify({
                "error": "Claim must be a string with at least 5 characters",
                "status": "failed"
            }), 400
            
        # Get database session
        db = next(get_db())
        
        try:
            # Initialize RAG service
            rag_service = RAGVerificationService(
                gemini_service,
                openalex_service,
                crossref_service,
                semantic_scholar_service,
                pubmed_service, # Pass PubMed service instance
                db
            )
            
            # Process the claim
            result = rag_service.process_claim_request(claim)
            
            # Check if the result has an error key
            if result and "error" in result:
                # Still return error directly, not nested
                return jsonify(result), 500
                
            # Return the result, WRAPPED in a 'result' key for frontend compatibility
            return jsonify({"result": result})
            
        except Exception as e:
            logger.error(f"Error in RAG verification: {e}")
            return jsonify({
                "error": f"Internal server error during claim verification: {str(e)}",
                "status": "failed"
            }), 500
        finally:
            # --- Add expunge_all() here ---
            try:
                logger.debug("Expunging objects from DB session before closing.")
                db.expunge_all() # Detach all objects from the session
                logger.debug("Expunged objects successfully.")
            except Exception as expunge_e:
                logger.error(f"Error expunging session objects: {expunge_e}")
            # --- End Add ---
            logger.debug("Closing DB session.")
            db.close()
            logger.debug("DB session closed.")
            
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({
            "error": f"Failed to process request: {str(e)}",
            "status": "failed"
        }), 400

# Run the Flask app
if __name__ == "__main__":
    port = int(os.getenv('PORT', 8080))
    app.run(host="0.0.0.0", port=port, debug=app.config.get("DEBUG", False))