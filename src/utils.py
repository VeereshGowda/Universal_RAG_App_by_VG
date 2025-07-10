"""
Utility functions for the Universal RAG application.
This module contains helper functions used across the application.
"""

import os
import logging
from typing import List, Dict, Any
import streamlit as st

def setup_logging() -> None:
    """
    Configure logging for the application.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def check_ollama_connection() -> bool:
    """
    Check if Ollama is running and accessible.
    
    Returns:
        bool: True if Ollama is accessible, False otherwise
    """
    try:
        import ollama
        # Try to list available models to test connection
        ollama.list()
        return True
    except Exception as e:
        logging.error(f"Ollama connection failed: {e}")
        return False

def check_required_models() -> Dict[str, bool]:
    """
    Check if required Ollama models are available.
    
    Returns:
        Dict[str, bool]: Status of each required model
    """
    required_models = ['mxbai-embed-large', 'llama3.2:latest']
    model_status = {}
    
    try:
        import ollama
        available_models = [model.model for model in ollama.list()['models']]
        
        for model in required_models:
            # Check if model exists (with or without :latest suffix)
            model_exists = (model in available_models or 
                          f"{model}:latest" in available_models or
                          f"{model.replace(':latest', '')}" in available_models)
            model_status[model] = model_exists
    except Exception as e:
        logging.error(f"Error checking models: {e}")
        for model in required_models:
            model_status[model] = False
    
    return model_status

def get_supported_file_extensions() -> List[str]:
    """
    Get list of supported file extensions.
    
    Returns:
        List[str]: List of supported file extensions
    """
    return ['.pdf', '.docx', '.xlsx', '.jpg', '.jpeg', '.png']

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes (int): File size in bytes
    
    Returns:
        str: Formatted file size (e.g., "1.5 MB")
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks for better context preservation.
    
    Args:
        text (str): Text to be chunked
        chunk_size (int): Maximum size of each chunk
        overlap (int): Number of characters to overlap between chunks
    
    Returns:
        List[str]: List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # If we're not at the end, try to find a natural break point
        if end < len(text):
            # Look for sentence ending or paragraph break
            break_point = text.rfind('.', start, end)
            if break_point == -1:
                break_point = text.rfind(' ', start, end)
            if break_point != -1 and break_point > start:
                end = break_point + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap if end < len(text) else len(text)
    
    return chunks

def display_system_status():
    """
    Display system status in the Streamlit sidebar.
    """
    with st.sidebar:
        st.subheader("🔧 System Status")
        
        # Check Ollama connection
        if check_ollama_connection():
            st.success("✅ Ollama Connected")
        else:
            st.error("❌ Ollama Not Connected")
            st.warning("Please ensure Ollama is running locally")
        
        # Check required models
        model_status = check_required_models()
        st.write("**Required Models:**")
        
        for model, available in model_status.items():
            if available:
                st.success(f"✅ {model}")
            else:
                st.error(f"❌ {model}")
                st.code(f"ollama pull {model}")

def create_info_box(title: str, content: str, type: str = "info"):
    """
    Create an information box in Streamlit.
    
    Args:
        title (str): Title of the info box
        content (str): Content to display
        type (str): Type of box ('info', 'success', 'warning', 'error')
    """
    if type == "info":
        st.info(f"**{title}**\n\n{content}")
    elif type == "success":
        st.success(f"**{title}**\n\n{content}")
    elif type == "warning":
        st.warning(f"**{title}**\n\n{content}")
    elif type == "error":
        st.error(f"**{title}**\n\n{content}")
