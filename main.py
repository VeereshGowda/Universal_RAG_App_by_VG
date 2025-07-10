"""
Universal RAG Application with Streamlit UI
===========================================

A comprehensive RAG application that processes multiple file types
(PDF, DOCX, Excel, Images) and provides an AI-powered chat interface
using ChromaDB vector storage and Ollama models.

Author: Universal RAG System
"""

import streamlit as st
import logging
import os
from typing import List, Dict, Any
import time
import tempfile
import shutil

# Import configuration
import config

# Import our custom modules
from src.document_processor import DocumentProcessor
from src.vector_store_unified import UnifiedVectorStore
from src.rag_system_unified import UnifiedRAGSystem
from src.utils import (
    setup_logging, display_system_status, check_ollama_connection,
    get_supported_file_extensions, format_file_size, create_info_box
)

# Configure the Streamlit page
st.set_page_config(
    page_title="Universal RAG Application",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = False
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = {}
    if 'uploaded_files_processed' not in st.session_state:
        st.session_state.uploaded_files_processed = False

def setup_rag_system():
    """Initialize the RAG system components."""
    try:
        # Initialize unified vector store
        if st.session_state.vector_store is None:
            with st.spinner("Initializing unified vector store..."):
                st.session_state.vector_store = UnifiedVectorStore()
        
        # Initialize unified RAG system
        if st.session_state.rag_system is None:
            with st.spinner("Initializing unified RAG system..."):
                st.session_state.rag_system = UnifiedRAGSystem(st.session_state.vector_store)
        
        return True
    except Exception as e:
        st.error(f"Failed to initialize unified RAG system: {str(e)}")
        logger.error(f"Unified RAG system initialization error: {e}")
        return False

def display_header():
    """Display the application header."""
    st.title("🤖 Universal RAG Application")
    st.markdown("""
    **Intelligent Document Analysis & Question Answering System**
    
    This application can process multiple file types (PDF, DOCX, Excel, Images) and answer questions 
    about their content using advanced AI models. You can process documents from the data folder or 
    upload your own files directly through the interface.
    """)

def display_sidebar():
    """Display the sidebar with system controls and status."""
    with st.sidebar:
        st.header("🔧 System Controls")
        
        # System status
        display_system_status()
        
        st.divider()
        
        # Document processing section
        st.subheader("📁 Document Processing")
        
        # Add tabs for different document sources
        doc_tab1, doc_tab2 = st.tabs(["📂 Data Folder", "⬆️ Upload Files"])
        
        with doc_tab1:
            # Existing data folder processing
            data_path = "data"
            if os.path.exists(data_path):
                files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
                supported_files = [f for f in files if any(f.lower().endswith(ext) for ext in get_supported_file_extensions())]
                
                st.write(f"**Found {len(supported_files)} supported files:**")
                for file in supported_files:
                    file_path = os.path.join(data_path, file)
                    file_size = format_file_size(os.path.getsize(file_path))
                    st.write(f"📄 {file} ({file_size})")
                
                # Process documents button
                if st.button("🚀 Process Data Folder", type="primary", use_container_width=True):
                    process_documents()
            else:
                st.warning("Data directory not found. Please ensure your documents are in the 'data' folder.")
        
        with doc_tab2:
            # File upload interface
            display_file_upload_interface()
        
        # Reset database button (outside tabs)
        st.divider()
        if st.button("🗑️ Reset Database", use_container_width=True):
            reset_database()
        
        st.divider()
        
        # Vector store statistics
        if st.session_state.vector_store:
            st.subheader("📊 Database Stats")
            display_vector_store_stats()
        
        st.divider()
        
        # Chain of Thought Help Section
        st.subheader("🧠 Chain of Thought Guide")
        
        with st.expander("ℹ️ How to Use CoT Reasoning"):
            st.markdown("""
            **Chain of Thought (CoT) Reasoning** helps the AI break down complex questions into steps:
            
            **✅ Best for:**
            - Complex analytical questions
            - Multi-part queries
            - Comparisons between data
            - Questions requiring logical reasoning
            
            **📝 Example CoT Questions:**
            - "Compare population trends between Maharashtra and Karnataka cities"
            - "Analyze the relationship between Krishnadevaraya and the Anglo-Mysore Wars"
            - "What factors influenced the demographic changes in Indian cities?"
            
            **⚡ Quick Questions:**
            For simple facts, you can disable CoT for faster responses:
            - "What is the population of Mumbai?"
            - "When did Krishnadevaraya rule?"
            """)
        
        with st.expander("🎯 Question Examples by File Type"):
            st.markdown("""
            **📊 Excel Data (Cities):**
            - "Which state has the most cities with population over 1 million?"
            - "Compare literacy rates between northern and southern states"
            
            **📚 Historical Docs (Krishnadevaraya):**
            - "What were Krishnadevaraya's major achievements?"
            - "How did his rule impact the Vijayanagara Empire?"
            
            **🏛️ War Documents (Anglo-Mysore):**
            - "What led to the Anglo-Mysore Wars?"
            - "How did these wars affect regional politics?"
            
            **🏝️ Travel Images (Andaman):**
            - "What can you tell me about Andaman Islands tourism?"
            - "Describe the scenic attractions shown in the images"
            """)

def process_documents():
    """Process all documents in the data directory."""
    if not check_ollama_connection():
        st.error("❌ Ollama is not connected. Please start Ollama first.")
        return
    
    if not setup_rag_system():
        return
    
    data_path = "data"
    
    try:
        # Initialize document processor
        processor = DocumentProcessor()
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("📂 Scanning data directory...")
        
        # Process all documents in the data directory
        results = processor.process_directory(data_path)
        
        if not results:
            st.warning("No supported documents found in the data directory.")
            progress_bar.empty()
            status_text.empty()
            return
        
        progress_bar.progress(0.3)
        status_text.text("📄 Processing document content...")
        
        # Display processing results
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'error']
        
        if successful:
            st.success(f"✅ Successfully processed {len(successful)} documents")
            for doc in successful[:3]:  # Show first 3 documents
                st.write(f"📄 {doc['file_name']} - {len(doc['content'])} characters")
            if len(successful) > 3:
                st.write(f"... and {len(successful) - 3} more documents")
        
        if failed:
            st.error(f"❌ Failed to process {len(failed)} documents")
            for doc in failed:
                st.write(f"📄 {doc['file_name']} - {doc.get('error', 'Unknown error')}")
        
        # Add documents to vector store with progress tracking
        if successful:
            progress_bar.progress(0.5)
            status_text.text("🔗 Creating embeddings and storing in vector database...")
            
            # Create a progress callback
            def update_progress(message):
                status_text.text(f"🔗 {message}")
            
            try:
                chunks_added = st.session_state.vector_store.add_documents(
                    successful, 
                    chunk_size=config.CHUNK_SIZE,
                    overlap=config.CHUNK_OVERLAP,
                    progress_callback=update_progress
                )
                
                progress_bar.progress(1.0)
                status_text.text("✅ Processing complete!")
                
                st.success(f"✅ Added {chunks_added} text chunks to the vector database")
                st.session_state.documents_processed = True
                
                # Store processing status
                st.session_state.processing_status = {
                    'total_files': len(results),
                    'successful_files': len(successful),
                    'failed_files': len(failed),
                    'chunks_added': chunks_added,
                    'processed_at': time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Clear progress indicators immediately
                progress_bar.empty()
                status_text.empty()
                
                # Force a rerun to update the UI
                st.rerun()
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                raise e
    
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        logger.error(f"Document processing error: {e}")
        # Clean up progress indicators
        try:
            progress_bar.empty()
            status_text.empty()
        except:
            pass

def reset_database():
    """Reset the vector database."""
    if st.session_state.vector_store:
        with st.spinner("Resetting database..."):
            try:
                success = st.session_state.vector_store.reset_collection()
                if success:
                    st.success("✅ Database reset successfully")
                    st.session_state.documents_processed = False
                    st.session_state.uploaded_files_processed = False
                    st.session_state.chat_history = []
                    st.session_state.processing_status = {}
                    # Force UI refresh
                    st.rerun()
                else:
                    st.error("❌ Failed to reset database")
            except Exception as e:
                st.error(f"❌ Failed to reset database: {str(e)}")
    else:
        st.warning("No vector store available to reset")

def display_vector_store_stats():
    """Display vector store statistics."""
    if st.session_state.vector_store:
        stats = st.session_state.vector_store.get_collection_stats()
        
        if 'error' in stats:
            st.error(f"Error: {stats['error']}")
        else:
            st.metric("Total Chunks", stats['total_chunks'])
            st.metric("Total Documents", stats['total_documents'])
            st.metric("Embedding Model", stats['embedding_model'])
            
            if stats['file_types']:
                st.write("**File Types:**")
                for file_type, count in stats['file_types'].items():
                    st.write(f"• {file_type}: {count} chunks")
            
            # Show processing status if available
            if st.session_state.processing_status:
                st.write("**Processing History:**")
                status = st.session_state.processing_status
                
                if 'processed_at' in status:
                    st.write(f"• Data folder: {status.get('successful_files', 0)} files, {status.get('chunks_added', 0)} chunks")
                    st.write(f"  Processed: {status['processed_at']}")
                
                if 'last_upload_at' in status:
                    st.write(f"• Uploaded: {status.get('uploaded_successful', 0)} files, {status.get('uploaded_chunks', 0)} chunks")
                    st.write(f"  Last upload: {status['last_upload_at']}")

def display_chat_interface():
    """Display the main chat interface with Chain of Thought options."""
    st.header("💬 Ask Questions About Your Documents")
    
    if not st.session_state.documents_processed:
        create_info_box(
            "Getting Started",
            "To start asking questions, please process your documents first using the sidebar controls.",
            "info"
        )
        return
    
    # Chain of Thought Controls
    st.subheader("🧠 Reasoning Options")
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        use_cot = st.toggle(
            "🔗 Enable Chain of Thought Reasoning",
            value=True,
            help="When enabled, the AI will show step-by-step reasoning before providing the final answer. This improves accuracy for complex questions."
        )
    
    with col2:
        show_reasoning = st.toggle(
            "👁️ Show Reasoning",
            value=True,
            help="Display the reasoning steps when Chain of Thought is enabled."
        )
    
    with col3:
        context_docs = st.selectbox(
            "📚 Context Documents",
            options=[5, 8, 10, 15, 20],
            index=2,  # Default to 10 documents for better list retrieval
            help="Number of relevant documents to use for answering. Higher values help with complex list queries."
        )
    
    st.divider()
    
    # Chat input at the top for easy access
    user_question = st.chat_input("Ask a question about your documents...")
    
    if user_question:
        # Add user message to history (newest first)
        st.session_state.chat_history.insert(0, {
            'role': 'user',
            'content': user_question
        })
        
        # Generate response
        with st.spinner("🧠 Processing your question..."):
            try:
                # Pass only the previous conversation history (excluding current question)
                previous_history = st.session_state.chat_history[1:] if len(st.session_state.chat_history) > 1 else []
                
                response = st.session_state.rag_system.generate_response(
                    user_question,
                    n_context_docs=context_docs,
                    max_context_length=config.MAX_CONTEXT_LENGTH,
                    use_cot=use_cot,
                    conversation_history=previous_history
                )
                
                if response['status'] == 'success':
                    # Add response to history (after user message, so index 1)
                    st.session_state.chat_history.insert(1, {
                        'role': 'assistant',
                        'content': response['answer'],
                        'reasoning': response.get('reasoning'),
                        'sources': response.get('sources', []),
                        'cot_used': response.get('cot_used', False),
                        'retrieved_docs': response.get('retrieved_docs', 0)
                    })
                    
                elif response['status'] == 'no_results':
                    st.session_state.chat_history.insert(1, {
                        'role': 'assistant',
                        'content': response['answer'],
                        'cot_used': use_cot
                    })
                
                else:
                    st.session_state.chat_history.insert(1, {
                        'role': 'assistant',
                        'content': f"Error: {response['answer']}",
                        'cot_used': use_cot
                    })
            
            except Exception as e:
                st.session_state.chat_history.insert(1, {
                    'role': 'assistant',
                    'content': f"I encountered an error: {str(e)}",
                    'cot_used': use_cot
                })
        
        # Force refresh to show new content at top
        st.rerun()
    
    # Display conversation history (newest at top, oldest at bottom)
    if st.session_state.chat_history:
        st.subheader("💭 Conversation History (Newest First)")
        
        # Create a container for the chat history with scrolling
        chat_container = st.container()
        
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                if message['role'] == 'user':
                    with st.chat_message("user"):
                        st.write(f"**Q{len(st.session_state.chat_history)//2 - i//2}:** {message['content']}")
                else:
                    with st.chat_message("assistant"):
                        # Display reasoning if available and show_reasoning is enabled
                        if message.get('reasoning') and show_reasoning:
                            with st.expander("🧠 Reasoning Steps", expanded=False):
                                st.markdown(message['reasoning'])
                        
                        # Display main answer
                        st.write(message['content'])
                        
                        # Display metadata
                        metadata_col1, metadata_col2 = st.columns(2)
                        with metadata_col1:
                            if message.get('cot_used') is not None:
                                st.caption(f"🔗 Chain of Thought: {'Enabled' if message.get('cot_used') else 'Disabled'}")
                        with metadata_col2:
                            if message.get('retrieved_docs'):
                                st.caption(f"📚 Documents Used: {message.get('retrieved_docs', 0)}")
                        
                        # Display sources if available
                        if 'sources' in message and message['sources']:
                            with st.expander("📚 Sources"):
                                for source in message['sources']:
                                    # Handle both old and new source format
                                    if 'relevance_score' in source:
                                        relevance_score = source['relevance_score']
                                        relevance_emoji = "🟢" if relevance_score > 0.8 else "🟡" if relevance_score > 0.6 else "🔴"
                                        st.write(f"{relevance_emoji} **{source['file_name']}** ({source['file_type']}) - Relevance: {relevance_score:.3f}")
                                    else:
                                        # Fallback for sources without relevance score
                                        st.write(f"📄 **{source['file_name']}** ({source['file_type']})")
                                        if 'distance' in source:
                                            st.caption(f"Distance: {source['distance']:.3f}")
                                        if 'embedding_model' in source:
                                            st.caption(f"Embedding: {source['embedding_model']}")
                
                # Add separator between conversations
                if i < len(st.session_state.chat_history) - 1:
                    st.divider()

def display_document_summary():
    """Display document summary tab."""
    st.header("📋 Document Summary")
    
    if not st.session_state.documents_processed:
        create_info_box(
            "No Documents Processed",
            "Please process your documents first to see a summary.",
            "info"
        )
        return
    
    if st.button("📊 Generate Document Summary"):
        with st.spinner("Generating summary..."):
            try:
                summary_result = st.session_state.rag_system.summarize_documents()
                
                if summary_result['status'] == 'success':
                    st.markdown("### 📖 Document Collection Summary")
                    st.write(summary_result['summary'])
                    
                    # Display statistics
                    st.markdown("### 📊 Statistics")
                    stats = summary_result['stats']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Chunks", stats['total_chunks'])
                    with col2:
                        st.metric("Unique Files", stats['unique_files'])
                    with col3:
                        st.metric("File Types", len(stats['file_types']))
                    
                    # Display file types
                    if stats['file_types']:
                        st.markdown("### 📁 File Types")
                        for file_type, count in stats['file_types'].items():
                            st.write(f"• **{file_type}**: {count} chunks")
                
                else:
                    st.error(f"Error generating summary: {summary_result.get('error', 'Unknown error')}")
            
            except Exception as e:
                st.error(f"Error generating summary: {str(e)}")

def display_file_upload_interface():
    """Display the file upload interface in the sidebar."""
    st.write("**Upload your own documents:**")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['pdf', 'docx', 'xlsx', 'jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help="Upload PDF, DOCX, Excel, or image files"
    )
    
    if uploaded_files:
        st.write(f"**{len(uploaded_files)} files selected:**")
        total_size = 0
        
        for file in uploaded_files:
            file_size = len(file.getvalue())
            total_size += file_size
            st.write(f"📄 {file.name} ({format_file_size(file_size)})")
        
        st.write(f"**Total size:** {format_file_size(total_size)}")
        
        # Process uploaded files button
        if st.button("🚀 Process Uploaded Files", type="primary", use_container_width=True):
            process_uploaded_files(uploaded_files)
    else:
        st.info("Select files to upload and process")

def process_uploaded_files(uploaded_files):
    """Process uploaded files and add them to the vector database."""
    if not check_ollama_connection():
        st.error("❌ Ollama is not connected. Please start Ollama first.")
        return
    
    if not setup_rag_system():
        return
    
    try:
        # Create temporary directory for uploaded files
        temp_dir = tempfile.mkdtemp()
        
        # Initialize document processor
        processor = DocumentProcessor()
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("📤 Processing uploaded files...")
        
        # Save uploaded files to temporary directory
        temp_files = []
        for uploaded_file in uploaded_files:
            # Check file size
            file_size = len(uploaded_file.getvalue())
            max_size = config.MAX_IMAGE_SIZE if uploaded_file.name.lower().endswith(('.jpg', '.jpeg', '.png')) else config.MAX_FILE_SIZE
            
            if file_size > max_size:
                st.error(f"❌ File {uploaded_file.name} is too large. Maximum size: {format_file_size(max_size)}")
                continue
            
            # Save file to temporary directory
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_file_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            temp_files.append(temp_file_path)
        
        if not temp_files:
            st.error("❌ No valid files to process")
            return
        
        progress_bar.progress(0.2)
        status_text.text("📄 Processing document content...")
        
        # Process documents
        results = []
        for temp_file_path in temp_files:
            try:
                result = processor.process_file(temp_file_path)
                results.append(result)
            except Exception as e:
                st.error(f"❌ Error processing {os.path.basename(temp_file_path)}: {str(e)}")
                logger.error(f"Error processing uploaded file {temp_file_path}: {e}")
        
        # Clean up temporary files
        shutil.rmtree(temp_dir)
        
        if not results:
            st.error("❌ No documents were successfully processed")
            return
        
        progress_bar.progress(0.5)
        
        # Display processing results
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'error']
        
        if successful:
            st.success(f"✅ Successfully processed {len(successful)} uploaded documents")
            for doc in successful:
                st.write(f"📄 {doc['file_name']} - {len(doc['content'])} characters")
        
        if failed:
            st.error(f"❌ Failed to process {len(failed)} documents")
            for doc in failed:
                st.write(f"📄 {doc['file_name']} - {doc.get('error', 'Unknown error')}")
        
        # Add documents to vector store
        if successful:
            progress_bar.progress(0.7)
            status_text.text("🔗 Creating embeddings and storing in vector database...")
            
            # Create a progress callback
            def update_progress(message):
                status_text.text(f"🔗 {message}")
            
            try:
                chunks_added = st.session_state.vector_store.add_documents(
                    successful,
                    chunk_size=config.CHUNK_SIZE,
                    overlap=config.CHUNK_OVERLAP,
                    progress_callback=update_progress
                )
                
                progress_bar.progress(1.0)
                status_text.text("✅ Upload processing complete!")
                
                st.success(f"✅ Added {chunks_added} text chunks from uploaded files to the vector database")
                st.session_state.documents_processed = True
                
                # Update processing status
                current_status = st.session_state.processing_status
                current_status.update({
                    'uploaded_files': len(uploaded_files),
                    'uploaded_successful': len(successful),
                    'uploaded_failed': len(failed),
                    'uploaded_chunks': chunks_added,
                    'last_upload_at': time.strftime("%Y-%m-%d %H:%M:%S")
                })
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Force UI refresh
                st.rerun()
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                raise e
    
    except Exception as e:
        st.error(f"❌ Error processing uploaded files: {str(e)}")
        logger.error(f"Upload processing error: {e}")
        # Clean up progress indicators
        try:
            progress_bar.empty()
            status_text.empty()
        except:
            pass

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Display header
    display_header()
    
    # Display sidebar
    display_sidebar()
    
    # Main content area with tabs
    tab1, tab2 = st.tabs(["💬 Chat", "📋 Summary"])
    
    with tab1:
        display_chat_interface()
    
    with tab2:
        display_document_summary()
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <small>Universal RAG Application • Built with Streamlit, ChromaDB & Ollama</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
