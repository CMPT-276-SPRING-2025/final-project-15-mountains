# Factify: AI-Powered Scientific Fact Checker

## Project Overview

Factify is an advanced real-time fact-checking platform designed to verify scientific claims by leveraging cutting-edge artificial intelligence and scholarly databases. Our application provides researchers, students, journalists, and curious minds with evidence-based verification of scientific statements through a modern, responsive interface.

## Key Features

- **Instant Claim Verification**: Enter any scientific claim and get real-time analysis of its validity backed by academic research
- **Research-Backed Results**: Access evidence from peer-reviewed studies across multiple academic databases
- **AI-Powered Analysis**: Utilizes advanced AI models to analyze claims and extract relevant information
- **Citation Support**: View and explore supporting research papers with full citation details and relevance scoring
- **User-Friendly Interface**: Clean, intuitive design with responsive and interactive UI elements

## Technology Stack

### Frontend
- React 19 with Vite for optimized builds and HMR
- React Router for client-side navigation
- Custom CSS with responsive design principles
- Interactive Magnetic UI elements for enhanced user experience
- Jest and Vitest for component testing

### Backend
- Flask (Python) RESTful API with CORS support
- SQLAlchemy ORM with comprehensive database models
- Supabase for authentication, storage, and PostgreSQL database
- FAISS vector database for high-performance similarity search
- Sentence Transformers for semantic text embeddings
- Concurrent processing with thread pooling for faster responses

### AI/ML Components
- **RAG Architecture**: Implements a sophisticated Retrieval Augmented Generation pipeline for accurate fact verification
- **Semantic Embeddings**: Utilizes the all-MiniLM-L6-v2 model (384-dimensional embeddings) for document and query representation
- **Vector Search**: FAISS IndexFlatL2 for efficient L2 distance-based nearest neighbor search
- **LLM Integration**: Google's Generative AI (Gemini) with prompt engineering for:
  - Claim decomposition into atomic verifiable statements
  - Evidence synthesis across multiple research papers
  - Confidence scoring with numerical justification
  - Citation and reference formatting
- **Adaptive Knowledge Retrieval**: Dynamically adjusts retrieval strategy based on claim complexity
- **Memory Optimization**: Implements lazy loading for ML models and memory-efficient batching

### External APIs
- OpenAlex API for comprehensive academic paper retrieval with citation metrics
- CrossRef API for cross-referencing academic sources
- Semantic Scholar API for additional paper information and citation contexts

## Getting Started

### Prerequisites
- Node.js (v18+)
- Python (v3.10+)
- pip (Python package manager)
- npm (Node package manager)
- Supabase account (for database and auth)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-org/factify.git](https://github.com/CMPT-276-SPRING-2025/final-project-15-mountains
   cd factify
   ```

2. Set up the backend:
   ```
   cd src/backend
   pip install -r requirements.txt
   cp .env.example .env
   # Update the .env file with your API keys and Supabase credentials
   ```

3. Set up the frontend:
   ```
   cd ../
   npm install
   ```

4. Run the application:
   ```
   # In one terminal (from the src directory)
   python backend/app.py
   
   # In another terminal (from the src directory)
   npm run dev
   ```

5. Open your browser and navigate to `http://localhost:5173`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Factify Team
