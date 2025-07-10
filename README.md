# Universal RAG Application

A comprehensive RAG (Retrieval-Augmented Generation) application that processes and queries multiple file types including PDFs, DOCX, Excel files, and images using a unified embedding system. Features advanced sequential chunk retrieval for complete list extraction, intelligent context assembly, and file upload capabilities.

## 🎯 Key Features

### 🔧 Core Architecture
- **Unified Embedding System**: Single `nomic-embed-text` model for all file types ensuring consistent ranking and eliminating distance skew
- **Advanced List Query Detection**: Automatically detects list-based queries and optimizes retrieval for complete multi-part answers
- **Sequential Chunk Retrieval**: Retrieves adjacent chunks from the same document for comprehensive context
- **Priority Context Assembly**: Intelligent context prioritization and combination for better response quality

### 📁 Document Processing
- **Universal Support**: PDF, DOCX, Excel (.xlsx), and image files (JPG, PNG)
- **File Upload Interface**: Upload and process your own documents directly through the web interface
- **Smart Text Extraction**: Specialized processing for each file type with OCR support for images
- **Flexible Configuration**: Adjustable chunk sizes and overlap settings for optimal performance

### 🤖 AI-Powered Features
- **Llama3.2 Integration**: Advanced language model with Chain-of-Thought reasoning capabilities
- **Enhanced Prompt Engineering**: Specialized prompts for different query types (lists, facts, analysis)
- **Context-Aware Responses**: Maintains conversation history for follow-up questions
- **Source Citations**: Answers include references to source documents with relevance scores

### 🎨 User Interface
- **Modern Streamlit UI**: Beautiful, responsive web interface with real-time processing
- **Interactive Controls**: Adjustable context parameters and reasoning modes
- **Progress Tracking**: Real-time feedback during document processing
- **Status Dashboard**: System health monitoring and database statistics

## 🏗️ Project Structure

```
Universal-RAG/
├── main.py                      # Main Streamlit application with file upload
├── config.py                    # Unified configuration settings
├── requirements.txt             # Python dependencies
├── pyproject.toml              # Project configuration
├── setup.py                    # Package setup
├── README.md                   # This file
├── USER_GUIDE.md               # Detailed user guide
├── src/
│   ├── document_processor.py   # Universal document processing
│   ├── vector_store_unified.py # Unified ChromaDB vector store
│   ├── rag_system_unified.py   # Unified RAG pipeline with enhanced retrieval
│   └── utils.py                # Utility functions and helpers
├── data/                       # Sample documents (optional)
│   ├── *.pdf                   # PDF documents
│   ├── *.docx                  # Word documents
│   ├── *.xlsx                  # Excel spreadsheets
│   └── *.jpg/*.png             # Image files
├── chroma_db/                  # ChromaDB storage (auto-created)
│   └── documents/              # Unified collection for all file types
└── test_*.py                   # Test and diagnostic scripts
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ 
- [Ollama](https://ollama.ai/) installed and running
- (Optional) Tesseract OCR for image text extraction

### 1. Install Dependencies
```bash
# Clone the repository
git clone <repository-url>
cd Universal-RAG

# Install Python packages
pip install -r requirements.txt
```

### 2. Setup Ollama Models
```bash
# Start Ollama service
ollama serve

# In another terminal, pull required models
ollama pull nomic-embed-text
ollama pull llama3.2:latest
```

### 3. Run the Application
```bash
streamlit run main.py
```

The application will be available at `http://localhost:8501`

## 📖 Usage Guide

### Processing Documents

#### Option 1: Data Folder
1. Place your documents in the `data/` directory
2. Click "🚀 Process Data Folder" in the sidebar
3. Wait for processing to complete

#### Option 2: File Upload
1. Go to the "⬆️ Upload Files" tab in the sidebar
2. Select and upload your files (up to 50MB each)
3. Files are automatically processed and embedded

### Asking Questions

#### Basic Questions
- Simply type your question in the chat input
- The system will find relevant context and provide answers
- Sources are automatically cited with relevance scores

#### List Queries (Enhanced)
For queries requesting lists or multiple items:
- "What are the five applications of building with Generative AI?"
- "List the major cities in Maharashtra"
- "What were Krishnadevaraya's main achievements?"

The system automatically:
- Detects list-based queries
- Retrieves more context documents
- Assembles sequential chunks for complete answers
- Formats responses as numbered lists

#### Chain-of-Thought Reasoning
Enable CoT for complex analytical questions:
- "Compare population trends between different states"
- "Analyze the relationship between historical events"
- "What factors influenced demographic changes?"

### Configuration Options

#### Context Parameters
- **Context Documents**: Number of relevant chunks to retrieve (default: 10)
- **Max Context Length**: Maximum context length for the AI (default: 8000)
- **Temperature**: Controls response creativity (0.0 = focused, 1.0 = creative)

#### Query Types
- **Standard**: Fast responses for factual questions
- **Chain-of-Thought**: Detailed reasoning for complex queries
- **List Queries**: Automatically enhanced retrieval for comprehensive lists

## 🔧 Configuration

### Key Settings (config.py)

```python
# Embedding Model (Unified)
EMBEDDING_MODEL = "nomic-embed-text"

# LLM Model
LLM_MODEL = "llama3.2:latest"

# Chunking Configuration
CHUNK_SIZE = 1500                # Base chunk size
CHUNK_OVERLAP = 300              # Overlap between chunks

# RAG Configuration
DEFAULT_CONTEXT_DOCS = 10        # Default context documents
MAX_CONTEXT_LENGTH = 8000        # Maximum context length
```

### File Type Support

| File Type | Extensions | Features |
|-----------|------------|----------|
| PDF | `.pdf` | Text extraction with metadata |
| Word | `.docx` | Full document processing |
| Excel | `.xlsx` | Table-aware processing |
| Images | `.jpg`, `.png` | OCR text extraction |

### Advanced Features

#### Sequential Chunk Retrieval
- Automatically retrieves adjacent chunks from the same document
- Ensures complete context for multi-part answers
- Prioritizes chunks with list indicators

#### Context Assembly
- Intelligent prioritization of relevant chunks
- Combines related content from the same file
- Prevents truncation of important information

#### File Type Awareness
- Optimizes retrieval based on query type
- Prefers Excel data for numerical/statistical queries
- Prioritizes text documents for historical/narrative queries

## 🛠️ Development

### Project Architecture

#### Core Components
1. **Document Processor** (`src/document_processor.py`)
   - Handles universal file processing
   - Extracts text with format-specific optimizations
   - Manages chunking strategies

2. **Unified Vector Store** (`src/vector_store_unified.py`)
   - Single ChromaDB collection for all file types
   - Consistent embedding using nomic-embed-text
   - Efficient similarity search with metadata filtering

3. **RAG System** (`src/rag_system_unified.py`)
   - Advanced query processing and retrieval
   - Context assembly and prioritization
   - Chain-of-Thought reasoning integration

4. **Streamlit UI** (`main.py`)
   - File upload and processing interface
   - Real-time chat with the RAG system
   - Status monitoring and configuration

### Key Algorithms

#### List Query Detection
```python
def _detect_list_query(self, query: str) -> bool:
    """Detect if query is asking for a list or multiple items"""
    list_indicators = [
        'list', 'five', 'top', 'applications', 'examples',
        'ways', 'methods', 'steps', 'factors', 'reasons'
    ]
    return any(indicator in query.lower() for indicator in list_indicators)
```

#### Sequential Chunk Retrieval
```python
def _enhance_with_sequential_chunks(self, search_results, query):
    """Retrieve adjacent chunks for complete context"""
    # Identifies chunks from the same document
    # Retrieves sequential chunks to fill context gaps
    # Prioritizes chunks with list-relevant content
```

### Testing

#### Diagnostic Scripts
- `test_five_applications.py`: Tests list query completeness
- `test_chunk_examination.py`: Analyzes chunk content and retrieval
- `test_context_debug.py`: Debugs context assembly
- `test_enhanced_five_applications.py`: Validates enhanced retrieval

#### Running Tests
```bash
# Test specific functionality
python test_five_applications.py

# Check chunk content
python test_chunk_examination.py

# Debug context assembly
python test_context_debug.py
```

## 🔍 Troubleshooting

### Common Issues

#### Ollama Connection Issues
```bash
# Check if Ollama is running
ollama list

# Start Ollama service
ollama serve

# Pull required models
ollama pull nomic-embed-text
ollama pull llama3.2:latest
```

#### Image Processing Issues
- Install Tesseract OCR for image text extraction
- Windows: Download from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
- macOS: `brew install tesseract`
- Linux: `sudo apt-get install tesseract-ocr`

#### Database Issues
- Use "🗑️ Reset Database" in the sidebar to clear all data
- Check ChromaDB folder permissions
- Ensure sufficient disk space

#### Performance Issues
- Reduce `DEFAULT_CONTEXT_DOCS` for faster responses
- Lower `MAX_CONTEXT_LENGTH` to reduce memory usage
- Use smaller chunk sizes for faster processing

### Error Messages

| Error | Solution |
|-------|----------|
| "Ollama not connected" | Start Ollama service with `ollama serve` |
| "Model not found" | Pull required models with `ollama pull` |
| "Tesseract not found" | Install Tesseract OCR or skip image processing |
| "No results found" | Check document processing and embedding |

## 📚 Advanced Usage

### Custom Document Processing

#### Adding New File Types
1. Extend `DocumentProcessor` class
2. Add new file type handling
3. Update `SUPPORTED_EXTENSIONS` in config.py

#### Custom Chunking Strategies
```python
# In config.py
CHUNK_SIZE = 2000           # Larger chunks for more context
CHUNK_OVERLAP = 400         # More overlap for better continuity
```

### API Integration

#### Direct RAG System Usage
```python
from src.rag_system_unified import UnifiedRAGSystem
from src.vector_store_unified import UnifiedVectorStore

# Initialize system
vector_store = UnifiedVectorStore()
rag_system = UnifiedRAGSystem(vector_store)

# Generate response
response = rag_system.generate_response(
    query="What are the applications of AI?",
    n_context_docs=10,
    use_cot=True
)
```

### Performance Tuning

#### Memory Optimization
- Adjust `MAX_CONTEXT_LENGTH` based on available RAM
- Use batch processing for large document collections
- Enable garbage collection for long-running sessions

#### Speed Optimization
- Increase `DEFAULT_CONTEXT_DOCS` only when needed
- Use standard mode instead of CoT for simple queries
- Cache frequently accessed embeddings

## 🔄 Migration from Dual-Collection System

If you're migrating from a previous dual-collection system:

1. **Backup your data**: Copy `chroma_db/` directory
2. **Run migration script**: `python migrate_to_unified.py`
3. **Verify migration**: Check that all documents are re-embedded
4. **Update configuration**: Ensure `config.py` uses unified settings

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) for local LLM hosting
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Streamlit](https://streamlit.io/) for the web interface
- [nomic-embed-text](https://huggingface.co/nomic-ai/nomic-embed-text-v1) for embeddings
