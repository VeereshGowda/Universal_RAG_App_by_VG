# Universal RAG Application - User Guide

## Quick Start

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Install Ollama**: Download from [ollama.ai](https://ollama.ai/)
3. **Pull models**: `ollama pull nomic-embed-text` and `ollama pull llama3.2:latest`
4. **Start Ollama**: `ollama serve`
5. **Run app**: `streamlit run main.py`

## Overview

This application processes PDFs, Word docs, Excel files, and images to answer questions with source citations. It features file upload, advanced list retrieval, and Chain-of-Thought reasoning.

## Installation

### Prerequisites
- Python 3.8+
- Ollama
- Tesseract OCR (optional, for images)

### Setup Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Install Ollama (visit ollama.ai for your platform)
ollama pull nomic-embed-text
ollama pull llama3.2:latest

# Install Tesseract OCR (optional)
# Windows: Download from UB Mannheim Tesseract
# macOS: brew install tesseract
# Linux: sudo apt-get install tesseract-ocr
```

### Start Application
```bash
ollama serve  # Start Ollama service
streamlit run main.py  # Launch app at http://localhost:8501
```

## Using the Application

### Document Processing
- **Data folder**: Place files in `data/` folder, click "🚀 Process Data Folder"
- **File upload**: Use "⬆️ Upload Files" tab (up to 50MB per file)

### Supported File Types
| Type | Extensions | Max Size | Notes |
|------|------------|----------|-------|
| PDF | .pdf | 50MB | Text extraction |
| Word | .docx | 50MB | Full document |
| Excel | .xlsx | 50MB | Table-aware |
| Images | .jpg, .png | 10MB | Requires Tesseract |

### Query Types

#### Standard Queries
- Simple facts: "What is the population of Mumbai?"
- Direct questions: "When did Krishnadevaraya rule?"

#### List Queries (Auto-optimized)
System detects keywords like "list", "five", "top", "applications" and automatically:
- Increases context documents (10→15)
- Doubles context length (8K→16K chars)
- Retrieves sequential chunks

Examples:
- "What are the five applications of building with Generative AI?"
- "List the major cities in Tamil Nadu"

#### Chain-of-Thought Queries
Enable CoT for complex analysis:
- "Compare population trends between states"
- "Analyze factors that led to the Anglo-Mysore Wars"

## Configuration

### UI Settings
- **Context Documents**: 1-20 (default: 10)
- **Max Context Length**: 2K-12K chars (default: 8K)
- **Temperature**: 0.0-1.0 (default: 0.7)
- **Reasoning Mode**: Standard/Chain-of-Thought/Auto-detect

### System Settings (config.py)
```python
# Core models
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:latest"

# Processing
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300

# Retrieval
DEFAULT_CONTEXT_DOCS = 10
MAX_CONTEXT_LENGTH = 8000
```

## Troubleshooting

### Common Issues

| Issue | Symptoms | Solution |
|-------|----------|----------|
| Ollama not connected | "❌ Ollama is not connected" | `ollama serve` |
| Missing models | "Model not found" | `ollama pull nomic-embed-text` |
| OCR not working | Images don't process | Install Tesseract OCR |
| Slow performance | Long response times | Reduce context settings |
| Empty responses | "No relevant information" | Rephrase query or reprocess docs |

### Quick Fixes
```bash
# Check Ollama status
ollama list

# Restart Ollama
ollama serve

# Verify Tesseract
tesseract --version

# Reset database (use sidebar button in app)
```

## Advanced Usage

### Direct API Access
```python
from src.rag_system_unified import UnifiedRAGSystem

response = rag_system.generate_response(
    query="Your question",
    n_context_docs=15,
    max_context_length=10000,
    temperature=0.3,
    use_cot=True
)
```

### Performance Tips
- Use specific keywords in queries
- Enable CoT for complex analysis
- Adjust context settings based on query type
- Process documents in batches for large datasets

### Best Practices
- Use descriptive filenames
- Start with specific questions
- Verify source citations
- Cross-check important information

## Getting Help

1. Check system status indicators in sidebar
2. Review error messages in logs
3. Try processing files individually
4. Reset database if needed
5. Verify all dependencies are installed

For detailed information, see the main README.md file.
