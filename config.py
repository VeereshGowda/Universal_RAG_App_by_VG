"""
Configuration file for Universal RAG Application
This file contains all the configurable parameters for the application.
"""

# Ollama Models Configuration
EMBEDDING_MODEL = "nomic-embed-text"  # Better for structured/tabular data than mxbai-embed-large
LLM_MODEL = "llama3.2:latest"

# ChromaDB Configuration
VECTOR_DB_PATH = "chroma_db"
COLLECTION_NAME = "documents"
# Removed TABBERT_COLLECTION_NAME - now using unified collection

# Document Processing Configuration
DATA_DIRECTORY = "data"
SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.xlsx', '.jpg', '.jpeg', '.png']

# Text Chunking Configuration
CHUNK_SIZE = 1500  # Increased base chunk size to capture more context
CHUNK_OVERLAP = 300  # Increased overlap to reduce information loss

# Excel Processing Configuration (using unified embedding)
EXCEL_CHUNK_SIZE = 3000  # Larger chunks for tabular data
EXCEL_CHUNK_OVERLAP = 500  # Overlap for Excel data processing

# RAG Configuration
DEFAULT_CONTEXT_DOCS = 10  # Increased from 8 to handle complex list queries better
MAX_CONTEXT_LENGTH = 8000  # Increased context length limit for better list retrieval
CHAT_CONTEXT_DOCS = 8  # Increased from 5 for better conversation context
CHAT_MAX_CONTEXT_LENGTH = 6000  # Increased chat context limit

# LLM Generation Parameters
DEFAULT_TEMPERATURE = 0.7
SUMMARY_TEMPERATURE = 0.3
TOP_P = 0.9
TOP_K = 40

# Streamlit Configuration
PAGE_TITLE = "Universal RAG Application"
PAGE_ICON = "🤖"
LAYOUT = "wide"

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# File Size Limits (in bytes)
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB

# UI Configuration
SIDEBAR_WIDTH = 300
CHAT_CONTAINER_HEIGHT = 400

# Error Messages
ERROR_MESSAGES = {
    'ollama_not_connected': "Ollama is not running. Please start Ollama service first.",
    'model_not_found': "Required model not found. Please pull the model using 'ollama pull {model}'",
    'tabbert_init_failed': "TabBERT embedder initialization failed: {error}",
    'file_too_large': "File is too large. Maximum size allowed is {max_size}",
    'unsupported_format': "Unsupported file format. Supported formats: {formats}",
    'no_documents': "No documents found. Please add documents to the data directory first.",
    'processing_failed': "Failed to process document: {error}",
    'embedding_failed': "Failed to generate embeddings: {error}",
    'search_failed': "Search failed: {error}",
    'generation_failed': "Response generation failed: {error}"
}

# Success Messages
SUCCESS_MESSAGES = {
    'documents_processed': "Successfully processed {count} documents",
    'chunks_added': "Added {count} text chunks to vector database",
    'tabbert_initialized': "TabBERT embedder initialized successfully",
    'excel_processed': "Excel file processed with TabBERT embeddings",
    'database_reset': "Database reset successfully",
    'response_generated': "Response generated successfully"
}

# OCR Configuration (for image processing)
OCR_CONFIG = {
    'lang': 'eng',  # Language for OCR
    'psm': 6,       # Page segmentation mode
    'oem': 3        # OCR Engine mode
}

# Chain of Thought (CoT) Configuration
COT_ENABLED_BY_DEFAULT = True
COT_SHOW_REASONING_BY_DEFAULT = True
COT_TEMPERATURE = 0.6  # Slightly lower for more structured reasoning
COT_MAX_STEPS = 6  # Maximum reasoning steps to prevent overly long responses

# Tabular Query Detection Configuration
TABULAR_KEYWORDS = [
    'population', 'data', 'table', 'sheet', 'excel', 'city', 'cities', 'state', 'states',
    'count', 'total', 'sum', 'average', 'maximum', 'minimum', 'highest', 'lowest',
    'compare', 'comparison', 'list', 'show', 'display', 'statistics', 'stats',
    'demographics', 'census', 'survey', 'database', 'records', 'entries',
    'literacy', 'education', 'employment', 'income', 'age', 'gender', 'male', 'female'
]

# Search Configuration - Unified embedding approach
SEARCH_RESULTS_BALANCE = True  # Balance results from different file types
MIN_EXCEL_RESULTS = 2  # Minimum results from Excel files when available
MIN_OTHER_RESULTS = 2  # Minimum results from other file types
