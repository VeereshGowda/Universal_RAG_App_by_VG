"""
Unified vector store management for Universal RAG application.
This module handles ChromaDB vector database operations using a single
embedding model (nomic-embed-text) for all file types to ensure
consistent vector space and eliminate ranking skew.
"""

import os
import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import ollama
from src.utils import chunk_text
import config

class UnifiedVectorStore:
    """
    A unified vector store that uses nomic-embed-text for all file types.
    
    This class handles:
    - Single ChromaDB collection for all documents
    - Unified embedding generation using nomic-embed-text
    - Storing document chunks with file type metadata
    - Consistent similarity searches across all file types
    """
    
    def __init__(self, persist_directory: str = "chroma_db", collection_name: str = "documents"):
        """
        Initialize the unified vector store.
        
        Args:
            persist_directory (str): Directory to store ChromaDB data
            collection_name (str): Name of the collection to use
        """
        self.logger = logging.getLogger(__name__)
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Use unified embedding model for all file types
        self.embedding_model = config.EMBEDDING_MODEL  # nomic-embed-text
        
        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(allow_reset=True)
        )
        
        # Get or create unified collection
        self.collection = self._get_or_create_collection()
        
        self.logger.info(f"Unified vector store initialized with collection: {collection_name}")
        self.logger.info(f"Using embedding model: {self.embedding_model}")
    
    def _get_or_create_collection(self):
        """
        Get existing collection or create a new one.
        
        Returns:
            Collection: ChromaDB collection object
        """
        try:
            # Try to get existing collection
            collection = self.client.get_collection(name=self.collection_name)
            self.logger.info(f"Loaded existing collection: {self.collection_name}")
        except Exception:
            # Create new collection if it doesn't exist
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Unified universal document embeddings using nomic-embed-text"}
            )
            self.logger.info(f"Created new collection: {self.collection_name}")
        
        return collection
    
    def generate_embedding(self, text: str, timeout: int = 30) -> List[float]:
        """
        Generate embedding for given text using nomic-embed-text.
        
        Args:
            text (str): Text to generate embedding for
            timeout (int): Timeout in seconds
        
        Returns:
            List[float]: Embedding vector
        """
        try:
            # Truncate text if too long
            if len(text) > 8000:  # Limit text length
                text = text[:8000] + "..."
            
            # Use nomic-embed-text for all content types
            response = ollama.embeddings(
                model=self.embedding_model,
                prompt=text
            )
            return response['embedding']
            
        except Exception as e:
            self.logger.error(f"Error generating embedding with {self.embedding_model}: {e}")
            raise Exception(f"Failed to generate embedding: {e}")
    
    def add_documents(self, documents: List[Dict[str, Any]], chunk_size: int = 1000, overlap: int = 200, progress_callback=None) -> int:
        """
        Add documents to the unified vector store.
        
        Args:
            documents (List[Dict[str, Any]]): List of processed documents
            chunk_size (int): Base size of text chunks
            overlap (int): Base overlap between chunks
            progress_callback: Callback function for progress updates
        
        Returns:
            int: Number of chunks added
        """
        chunks_added = 0
        total_docs = len([doc for doc in documents if doc['status'] == 'success' and doc['content'].strip()])
        
        # Check for existing documents to avoid duplicates
        existing_files = set()
        try:
            existing_data = self.collection.get()
            if existing_data and existing_data['metadatas']:
                existing_files = {meta['file_name'] for meta in existing_data['metadatas']}
        except Exception as e:
            self.logger.warning(f"Could not check for existing files: {e}")
        
        for doc_idx, doc in enumerate(documents):
            if doc['status'] != 'success' or not doc['content'].strip():
                self.logger.warning(f"Skipping document: {doc['file_name']} (no content)")
                continue
            
            # Skip if document already exists
            if doc['file_name'] in existing_files:
                self.logger.info(f"Skipping {doc['file_name']} - already in database")
                if progress_callback:
                    progress_callback(f"Skipping {doc['file_name']} (already processed)")
                continue
            
            try:
                # Determine adaptive chunk size based on file type
                file_type = doc['file_type']
                if file_type == '.xlsx':
                    # Use larger chunks for Excel files
                    adaptive_chunk_size = config.EXCEL_CHUNK_SIZE
                    adaptive_overlap = config.EXCEL_CHUNK_OVERLAP
                else:
                    # Use standard chunks for other files
                    adaptive_chunk_size = chunk_size
                    adaptive_overlap = overlap
                
                # Create text chunks
                text_chunks = chunk_text(
                    doc['content'], 
                    chunk_size=adaptive_chunk_size, 
                    overlap=adaptive_overlap
                )
                
                if not text_chunks:
                    self.logger.warning(f"No chunks created for {doc['file_name']}")
                    continue
                
                # Generate embeddings and add to collection
                embeddings = []
                chunk_texts = []
                metadatas = []
                ids = []
                
                for chunk_idx, chunk in enumerate(text_chunks):
                    if not chunk.strip():
                        continue
                    
                    # Generate embedding using unified model
                    embedding = self.generate_embedding(chunk)
                    embeddings.append(embedding)
                    chunk_texts.append(chunk)
                    
                    # Enhanced metadata with file type information
                    metadata = {
                        'file_name': doc['file_name'],
                        'file_type': file_type,
                        'file_path': doc['file_path'],
                        'chunk_index': chunk_idx,
                        'chunk_size': len(chunk),
                        'embedding_model': self.embedding_model,
                        'is_excel': file_type == '.xlsx',
                        'is_image': file_type in ['.jpg', '.jpeg', '.png'],
                        'is_document': file_type in ['.pdf', '.docx']
                    }
                    metadatas.append(metadata)
                    
                    # Create unique ID
                    chunk_id = f"{doc['file_name']}_chunk_{chunk_idx}"
                    ids.append(chunk_id)
                
                # Add to collection in batch
                if embeddings:
                    self.collection.add(
                        embeddings=embeddings,
                        documents=chunk_texts,
                        metadatas=metadatas,
                        ids=ids
                    )
                    
                    chunks_added += len(embeddings)
                    self.logger.info(f"Added {len(embeddings)} chunks from {doc['file_name']}")
                    
                    if progress_callback:
                        progress_callback(f"Processed {doc['file_name']}: {len(embeddings)} chunks")
                
            except Exception as e:
                self.logger.error(f"Error adding document {doc['file_name']}: {e}")
                continue
        
        self.logger.info(f"Total chunks added to unified vector store: {chunks_added}")
        return chunks_added
    
    def search(self, query: str, n_results: int = 5, prefer_file_type: str = None) -> Dict[str, Any]:
        """
        Search for similar documents using query in unified vector space.
        
        Args:
            query (str): Search query
            n_results (int): Number of results to return
            prefer_file_type (str): Optional file type to prefer in results
        
        Returns:
            Dict[str, Any]: Search results with documents and metadata
        """
        try:
            # Generate embedding for query
            query_embedding = self.generate_embedding(query)
            
            # Search in unified collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # Process results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    result = {
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'distance': results['distances'][0][i] if results['distances'] else 0,
                        'id': results['ids'][0][i] if results['ids'] else '',
                        'file_type': results['metadatas'][0][i].get('file_type', 'unknown') if results['metadatas'] else 'unknown'
                    }
                    formatted_results.append(result)
            
            # Apply file type preference if specified
            if prefer_file_type and formatted_results:
                # Separate results by file type
                preferred_results = [r for r in formatted_results if r['file_type'] == prefer_file_type]
                other_results = [r for r in formatted_results if r['file_type'] != prefer_file_type]
                
                # Reorder to prefer specified file type while maintaining distance ordering
                if preferred_results:
                    # Take some preferred results and some others
                    preferred_count = min(len(preferred_results), max(1, n_results // 2))
                    other_count = n_results - preferred_count
                    
                    final_results = preferred_results[:preferred_count] + other_results[:other_count]
                else:
                    final_results = formatted_results
            else:
                final_results = formatted_results
            
            # Format response
            response = {
                'query': query,
                'n_results': len(final_results),
                'documents': final_results,
                'distances': [result['distance'] for result in final_results],
                'embedding_model': self.embedding_model
            }
            
            # Log search statistics
            file_type_counts = {}
            for result in final_results:
                file_type = result['file_type']
                file_type_counts[file_type] = file_type_counts.get(file_type, 0) + 1
            
            self.logger.info(f"Search completed: {len(final_results)} results, file types: {file_type_counts}")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error searching vector store: {e}")
            return {
                'query': query,
                'n_results': 0,
                'documents': [],
                'distances': [],
                'error': str(e)
            }
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the unified collection.
        
        Returns:
            Dict[str, Any]: Collection statistics
        """
        try:
            # Get all data from collection
            all_data = self.collection.get()
            
            if not all_data or not all_data['metadatas']:
                return {
                    'total_documents': 0,
                    'total_chunks': 0,
                    'file_types': {},
                    'embedding_model': self.embedding_model
                }
            
            # Count by file type
            file_type_counts = {}
            file_names = set()
            
            for metadata in all_data['metadatas']:
                file_type = metadata.get('file_type', 'unknown')
                file_name = metadata.get('file_name', 'unknown')
                
                file_type_counts[file_type] = file_type_counts.get(file_type, 0) + 1
                file_names.add(file_name)
            
            return {
                'total_documents': len(file_names),
                'total_chunks': len(all_data['metadatas']),
                'file_types': file_type_counts,
                'embedding_model': self.embedding_model,
                'collection_name': self.collection_name
            }
            
        except Exception as e:
            self.logger.error(f"Error getting collection stats: {e}")
            return {
                'total_documents': 0,
                'total_chunks': 0,
                'file_types': {},
                'embedding_model': self.embedding_model,
                'error': str(e)
            }
    
    def reset_collection(self):
        """Reset the collection by deleting and recreating it."""
        try:
            # Delete existing collection
            self.client.delete_collection(name=self.collection_name)
            self.logger.info(f"Deleted collection: {self.collection_name}")
            
            # Create new collection
            self.collection = self._get_or_create_collection()
            self.logger.info(f"Reset collection: {self.collection_name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error resetting collection: {e}")
            return False
    
    def delete_documents_by_filename(self, filename: str) -> int:
        """
        Delete all chunks for a specific file.
        
        Args:
            filename (str): Name of the file to delete
        
        Returns:
            int: Number of chunks deleted
        """
        try:
            # Get all data to find chunks for this file
            all_data = self.collection.get()
            
            if not all_data or not all_data['metadatas']:
                return 0
            
            # Find IDs of chunks for this file
            ids_to_delete = []
            for i, metadata in enumerate(all_data['metadatas']):
                if metadata.get('file_name') == filename:
                    ids_to_delete.append(all_data['ids'][i])
            
            # Delete chunks
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                self.logger.info(f"Deleted {len(ids_to_delete)} chunks for file: {filename}")
            
            return len(ids_to_delete)
            
        except Exception as e:
            self.logger.error(f"Error deleting documents for {filename}: {e}")
            return 0
