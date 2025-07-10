"""
Unified RAG (Retrieval-Augmented Generation) system for Universal RAG application.
This module uses a unified vector store with nomic-embed-text for all file types,
ensuring consistent ranking and eliminating distance skew between different
embedding models.
"""

import logging
from typing import List, Dict, Any, Optional
import ollama
from src.vector_store_unified import UnifiedVectorStore
import config

class UnifiedRAGSystem:
    """
    A unified RAG system that uses consistent embeddings for all file types.
    
    This class handles:
    - Query processing with unified embedding
    - Document retrieval from unified vector store
    - Conversation context management
    - Context preparation with file type awareness
    - AI response generation using Ollama
    """
    
    def __init__(self, vector_store: UnifiedVectorStore, llm_model: str = "llama3.2:latest"):
        """
        Initialize the unified RAG system.
        
        Args:
            vector_store (UnifiedVectorStore): Unified vector store instance
            llm_model (str): Ollama model name for text generation
        """
        self.logger = logging.getLogger(__name__)
        self.vector_store = vector_store
        self.llm_model = llm_model
        
        # Tabular query keywords for file type preference
        self.tabular_keywords = config.TABULAR_KEYWORDS
        
        self.logger.info(f"Unified RAG system initialized with model: {llm_model}")
    
    def generate_response(
        self, 
        query: str, 
        n_context_docs: int = 5,
        max_context_length: int = 4000,
        temperature: float = 0.7,
        use_cot: bool = True,
        conversation_history: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a response to a user query using unified RAG.
        
        Args:
            query (str): User's question or query
            n_context_docs (int): Number of relevant documents to retrieve
            max_context_length (int): Maximum length of context to include
            temperature (float): Temperature for text generation
            use_cot (bool): Whether to use Chain of Thought reasoning
            conversation_history (List[Dict[str, Any]]): Previous conversation turns
        
        Returns:
            Dict[str, Any]: Response containing answer, sources, and metadata
        """
        try:
            # Step 1: Process conversation history and enhance query
            enhanced_query = self._enhance_query_with_context(query, conversation_history)
            self.logger.info(f"Processing query: {enhanced_query[:100]}...")
            
            # Step 2: Determine file type preference for search
            prefer_file_type = self._determine_file_type_preference(enhanced_query)
            
            # Step 2.5: Adjust retrieval parameters for list queries
            is_list_query = self._detect_list_query(enhanced_query)
            if is_list_query:
                # Significantly increase retrieval for list queries
                n_context_docs = min(n_context_docs + 5, 15)  # Add 5 more docs, max 15
                max_context_length = min(max_context_length * 2, 12000)  # Double context length
                self.logger.info(f"List query detected - increased context docs to {n_context_docs}, context length to {max_context_length}")
            
            # Step 3: Retrieve relevant documents using unified search
            search_results = self.vector_store.search(
                enhanced_query, 
                n_results=n_context_docs, 
                prefer_file_type=prefer_file_type
            )
            
            # Step 3.5: For list queries, try to retrieve sequential chunks
            if is_list_query:
                search_results = self._enhance_with_sequential_chunks(search_results, enhanced_query)
            
            if search_results['n_results'] == 0:
                return {
                    'query': query,
                    'answer': "I couldn't find any relevant information in the documents to answer your question.",
                    'sources': [],
                    'context_used': "",
                    'status': 'no_results'
                }
            
            # Step 4: Prepare context from retrieved documents
            context = self._prepare_context(search_results['documents'], max_context_length)
            
            # Step 5: Prepare conversation context
            conversation_context = self._prepare_conversation_context(conversation_history)
            
            # Step 6: Generate response using LLM with optional CoT
            self.logger.info(f"Generating response using LLM with CoT: {use_cot}")
            
            if use_cot:
                prompt = self._create_cot_prompt(query, context, conversation_context)
                temp = config.COT_TEMPERATURE
            else:
                prompt = self._create_prompt(query, context, conversation_context)
                temp = temperature
            
            response = ollama.generate(
                model=self.llm_model,
                prompt=prompt,
                options={
                    'temperature': temp,
                    'top_p': config.TOP_P,
                    'top_k': config.TOP_K
                }
            )
            
            # Step 7: Process CoT response if used
            if use_cot:
                answer, reasoning = self._process_cot_response(response['response'])
            else:
                answer = response['response'].strip()
                reasoning = None
            
            # Step 8: Format and return result
            result = {
                'query': query,
                'answer': answer,
                'reasoning': reasoning,
                'sources': self._extract_sources(search_results['documents']),
                'context_used': context,
                'retrieved_docs': len(search_results['documents']),
                'model_used': self.llm_model,
                'embedding_model': search_results.get('embedding_model', 'unknown'),
                'cot_used': use_cot,
                'file_type_preference': prefer_file_type,
                'status': 'success'
            }
            
            self.logger.info("Response generated successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return {
                'query': query,
                'answer': f"I encountered an error while processing your question: {str(e)}",
                'sources': [],
                'context_used': "",
                'status': 'error',
                'error': str(e)
            }
    
    def _enhance_query_with_context(self, query: str, conversation_history: List[Dict[str, Any]] = None) -> str:
        """
        Enhance the search query with context from conversation history.
        
        Args:
            query (str): Original user query
            conversation_history (List[Dict[str, Any]]): Previous conversation turns
        
        Returns:
            str: Enhanced query for better document retrieval
        """
        if not conversation_history:
            return query
        
        # Get last few turns for context
        recent_turns = conversation_history[-3:] if len(conversation_history) > 3 else conversation_history
        
        # Extract relevant context
        context_parts = []
        for turn in recent_turns:
            if turn.get('user_message'):
                context_parts.append(turn['user_message'])
        
        if context_parts:
            # Combine with current query
            context_text = " ".join(context_parts[-2:])  # Last 2 turns
            enhanced_query = f"{context_text} {query}"
            return enhanced_query
        
        return query
    
    def _determine_file_type_preference(self, query: str) -> Optional[str]:
        """
        Determine if query should prefer specific file types.
        
        Args:
            query (str): User query
        
        Returns:
            Optional[str]: Preferred file type or None
        """
        query_lower = query.lower()
        
        # Check for tabular/Excel keywords
        if any(keyword in query_lower for keyword in self.tabular_keywords):
            return '.xlsx'
        
        # Check for image-related keywords
        image_keywords = ['image', 'picture', 'photo', 'visual', 'diagram', 'chart', 'graph']
        if any(keyword in query_lower for keyword in image_keywords):
            return '.jpg'  # Represents image files
        
        # Check for document-related keywords
        doc_keywords = ['document', 'text', 'article', 'paper', 'report', 'essay']
        if any(keyword in query_lower for keyword in doc_keywords):
            return '.pdf'  # Represents text documents
        
        return None  # No specific preference
    
    def _prepare_context(self, documents: List[Dict[str, Any]], max_length: int) -> str:
        """
        Prepare context from retrieved documents with intelligent chunking and priority ordering.
        
        Args:
            documents (List[Dict[str, Any]]): Retrieved documents
            max_length (int): Maximum context length
        
        Returns:
            str: Formatted context string
        """
        # Separate documents into priority and regular groups
        priority_docs = []
        regular_docs = []
        
        for doc in documents:
            content = doc.get('content', '').lower()
            # Check if this chunk contains list content or key applications
            is_priority = any(term in content for term in [
                'below are', 'five types', 'five common', 'five applications',
                'chat with your own data', 'create personalised', 'create personalized',
                'hyper-personalisation', 'hyper-personalization', 'prospect profiles',
                'pull information from', 'ai agents', 'customer service avatars',
                'build customer service'
            ])
            
            if is_priority:
                priority_docs.append(doc)
            else:
                regular_docs.append(doc)
        
        # Group documents by file and sort by relevance
        def group_by_file(docs):
            file_groups = {}
            for doc in docs:
                metadata = doc.get('metadata', {})
                file_name = metadata.get('file_name', 'unknown')
                chunk_index = metadata.get('chunk_index', 0)
                
                if file_name not in file_groups:
                    file_groups[file_name] = []
                file_groups[file_name].append({
                    'content': doc.get('content', ''),
                    'metadata': metadata,
                    'distance': doc.get('distance', 1.0),
                    'chunk_index': chunk_index
                })
            
            # Sort chunks within each file by chunk index
            for file_name in file_groups:
                file_groups[file_name].sort(key=lambda x: x['chunk_index'])
            
            return file_groups
        
        # Process priority documents first
        priority_file_groups = group_by_file(priority_docs)
        regular_file_groups = group_by_file(regular_docs)
        
        context_parts = []
        current_length = 0
        
        # Process priority files first
        for file_name, docs in priority_file_groups.items():
            if current_length >= max_length:
                break
                
            # Get file type from first document
            file_type = docs[0]['metadata'].get('file_type', 'unknown')
            
            # Add file type context
            if file_type == '.xlsx':
                prefix = f"[PRIORITY - Excel Data from {file_name}]"
            elif file_type in ['.jpg', '.jpeg', '.png']:
                prefix = f"[PRIORITY - Image Text from {file_name}]"
            elif file_type == '.pdf':
                prefix = f"[PRIORITY - PDF Document from {file_name}]"
            elif file_type == '.docx':
                prefix = f"[PRIORITY - Word Document from {file_name}]"
            else:
                prefix = f"[PRIORITY - Document from {file_name}]"
            
            # Combine chunks from the same file
            file_content_parts = []
            
            for doc in docs:
                chunk_content = doc['content'].strip()
                if chunk_content:
                    chunk_idx = doc['metadata'].get('chunk_index', 0)
                    file_content_parts.append(f"[Chunk {chunk_idx}] {chunk_content}")
            
            if file_content_parts:
                combined_content = "\n\n".join(file_content_parts)
                formatted_content = f"{prefix}\n{combined_content}\n"
                
                # For priority content, be more generous with space
                if current_length + len(formatted_content) <= max_length:
                    context_parts.append(formatted_content)
                    current_length += len(formatted_content)
                else:
                    # Try to fit at least some priority content
                    remaining_space = max_length - current_length
                    if remaining_space > 300:  # Minimum for meaningful priority content
                        truncated_content = formatted_content[:remaining_space-100]
                        # Try to cut at chunk boundary
                        last_chunk = truncated_content.rfind('[Chunk')
                        if last_chunk > len(prefix) + 50:
                            truncated_content = truncated_content[:last_chunk]
                        truncated_content += "\n[Priority content truncated - more available]"
                        context_parts.append(truncated_content)
                        current_length = max_length
                    break
        
        # Process regular files if we have space
        if current_length < max_length:
            # Sort regular files by relevance
            sorted_regular_files = sorted(regular_file_groups.items(), 
                                        key=lambda x: sum(doc['distance'] for doc in x[1]) / len(x[1]))
            
            for file_name, docs in sorted_regular_files:
                if current_length >= max_length:
                    break
                    
                # Skip if we already processed this file in priority
                if file_name in priority_file_groups:
                    continue
                
                # Get file type from first document
                file_type = docs[0]['metadata'].get('file_type', 'unknown')
                
                # Add file type context
                if file_type == '.xlsx':
                    prefix = f"[Excel Data from {file_name}]"
                elif file_type in ['.jpg', '.jpeg', '.png']:
                    prefix = f"[Image Text from {file_name}]"
                elif file_type == '.pdf':
                    prefix = f"[PDF Document from {file_name}]"
                elif file_type == '.docx':
                    prefix = f"[Word Document from {file_name}]"
                else:
                    prefix = f"[Document from {file_name}]"
                
                # Combine chunks from the same file
                file_content_parts = []
                
                for doc in docs:
                    chunk_content = doc['content'].strip()
                    if chunk_content:
                        chunk_idx = doc['metadata'].get('chunk_index', 0)
                        file_content_parts.append(f"[Chunk {chunk_idx}] {chunk_content}")
                
                if file_content_parts:
                    combined_content = "\n\n".join(file_content_parts)
                    formatted_content = f"{prefix}\n{combined_content}\n"
                    
                    if current_length + len(formatted_content) <= max_length:
                        context_parts.append(formatted_content)
                        current_length += len(formatted_content)
                    else:
                        # Try to fit at least some content
                        remaining_space = max_length - current_length
                        if remaining_space > 200:
                            truncated_content = formatted_content[:remaining_space-50]
                            # Try to cut at chunk boundary
                            last_chunk = truncated_content.rfind('[Chunk')
                            if last_chunk > len(prefix) + 50:
                                truncated_content = truncated_content[:last_chunk]
                            truncated_content += "\n[Content truncated - more available]"
                            context_parts.append(truncated_content)
                            current_length = max_length
                        break
        
        return "\n".join(context_parts)
    
    def _prepare_conversation_context(self, conversation_history: List[Dict[str, Any]] = None) -> str:
        """
        Prepare conversation context from history.
        
        Args:
            conversation_history (List[Dict[str, Any]]): Previous conversation turns
        
        Returns:
            str: Formatted conversation context
        """
        if not conversation_history:
            return ""
        
        context_parts = []
        # Get last few turns for context
        recent_turns = conversation_history[-3:] if len(conversation_history) > 3 else conversation_history
        
        for turn in recent_turns:
            if turn.get('user_message'):
                context_parts.append(f"Previous question: {turn['user_message']}")
            if turn.get('assistant_response'):
                context_parts.append(f"Previous answer: {turn['assistant_response']}")
        
        if context_parts:
            return "\\n".join(context_parts)
        
        return ""
    
    def _create_prompt(self, query: str, context: str, conversation_context: str) -> str:
        """
        Create prompt for LLM generation.
        
        Args:
            query (str): User query
            context (str): Document context
            conversation_context (str): Conversation history
        
        Returns:
            str: Formatted prompt
        """
        # Check if this is a list query
        is_list_query = self._detect_list_query(query)
        
        prompt_parts = [
            "You are an AI assistant that answers questions based on provided document context.",
            "Use the following context to answer the question accurately and comprehensively.",
            "If the context doesn't contain enough information, say so clearly.",
            "Pay attention to the file type indicators in the context (Excel, PDF, Image, etc.) to provide appropriate responses.",
        ]
        
        # Add specific instructions for list queries
        if is_list_query:
            prompt_parts.extend([
                "",
                "IMPORTANT: This question appears to be asking for a list or multiple items.",
                "Please provide a complete, numbered list with all items mentioned in the context.",
                "Make sure to include ALL relevant items, not just the first few you find.",
                "If the context contains partial information across multiple chunks, combine them to give a complete answer."
            ])
        
        prompt_parts.append("")
        
        if conversation_context:
            prompt_parts.extend([
                "Previous conversation context:",
                conversation_context,
                ""
            ])
        
        prompt_parts.extend([
            "Document context:",
            context,
            "",
            f"Question: {query}",
            "",
            "Answer:"
        ])
        
        return "\\n".join(prompt_parts)
    
    def _create_cot_prompt(self, query: str, context: str, conversation_context: str) -> str:
        """
        Create Chain of Thought prompt for LLM generation.
        
        Args:
            query (str): User query
            context (str): Document context  
            conversation_context (str): Conversation history
        
        Returns:
            str: Formatted CoT prompt
        """
        # Check if this is a list query
        is_list_query = self._detect_list_query(query)
        
        prompt_parts = [
            "You are an AI assistant that uses step-by-step reasoning to answer questions.",
            "Think through the problem carefully and show your reasoning process.",
            "Use the following format:",
            "",
            "REASONING:",
            "1. [First step of analysis]",
            "2. [Second step of analysis]",
            "3. [Continue as needed]",
            "",
            "ANSWER:",
            "[Your final answer based on the reasoning above]",
            "",
            "Pay attention to file type indicators in the context to provide appropriate responses.",
        ]
        
        # Add specific instructions for list queries
        if is_list_query:
            prompt_parts.extend([
                "",
                "IMPORTANT: This question appears to be asking for a list or multiple items.",
                "In your REASONING:",
                "1. Systematically scan ALL chunks and documents for relevant items",
                "2. Look for numbered lists, bullet points, or sequential information",
                "3. Check if information spans multiple chunks from the same document",
                "4. Identify ANY pattern that suggests multiple related items",
                "5. Combine information from different chunks to form complete lists",
                "",
                "In your ANSWER:",
                "- Provide a complete, numbered list with ALL items you found",
                "- If you find references to 'five applications' but only see one, explicitly state what you found",
                "- Search thoroughly across all provided context for the complete information",
                "- If the context appears incomplete, mention that some items may be in parts not provided"
            ])
        
        prompt_parts.append("")
        
        if conversation_context:
            prompt_parts.extend([
                "Previous conversation context:",
                conversation_context,
                ""
            ])
        
        prompt_parts.extend([
            "Document context:",
            context,
            "",
            f"Question: {query}",
            ""
        ])
        
        return "\\n".join(prompt_parts)
    
    def _process_cot_response(self, response: str) -> tuple[str, str]:
        """
        Process Chain of Thought response to extract reasoning and answer.
        
        Args:
            response (str): Raw LLM response
        
        Returns:
            tuple[str, str]: (answer, reasoning)
        """
        try:
            # Look for REASONING: and ANSWER: sections
            parts = response.split("ANSWER:")
            if len(parts) >= 2:
                reasoning_part = parts[0].replace("REASONING:", "").strip()
                answer_part = parts[1].strip()
                return answer_part, reasoning_part
            else:
                # Fallback if format is not followed
                return response.strip(), "Reasoning not clearly separated"
        except Exception as e:
            self.logger.warning(f"Error processing CoT response: {e}")
            return response.strip(), "Error processing reasoning"
    
    def _extract_sources(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract source information from documents.
        
        Args:
            documents (List[Dict[str, Any]]): Retrieved documents
        
        Returns:
            List[Dict[str, Any]]: Source information
        """
        sources = []
        seen_files = set()
        
        for doc in documents:
            metadata = doc.get('metadata', {})
            file_name = metadata.get('file_name', 'unknown')
            file_type = metadata.get('file_type', 'unknown')
            distance = doc.get('distance', 1.0)  # Default to 1.0 if no distance
            
            # Avoid duplicate sources
            if file_name not in seen_files:
                # Convert distance to relevance score (lower distance = higher relevance)
                # For cosine distance, good matches are typically < 0.5, moderate < 1.0, poor > 1.0
                # But distances can vary widely, so we use a more adaptive approach
                if distance <= 0.3:
                    relevance_score = 1.0  # Excellent match
                elif distance <= 0.6:
                    relevance_score = 0.8  # Good match
                elif distance <= 1.0:
                    relevance_score = 0.6  # Moderate match
                else:
                    # For distances > 1.0, use exponential decay
                    relevance_score = max(0.1, 0.6 * (1.0 / max(1.0, distance)))
                
                source = {
                    'file_name': file_name,
                    'file_type': file_type,
                    'file_path': metadata.get('file_path', ''),
                    'embedding_model': metadata.get('embedding_model', 'unknown'),
                    'distance': distance,
                    'relevance_score': relevance_score
                }
                sources.append(source)
                seen_files.add(file_name)
        
        return sources
    
    def _detect_list_query(self, query: str) -> bool:
        """
        Detect if the query is asking for a list or multiple items.
        
        Args:
            query (str): User query
        
        Returns:
            bool: True if query appears to be asking for a list
        """
        query_lower = query.lower()
        
        # List indicators
        list_keywords = [
            'five', 'five', '5', 'all', 'list', 'types', 'kinds', 'ways',
            'methods', 'steps', 'applications', 'uses', 'examples',
            'approaches', 'strategies', 'techniques', 'benefits',
            'advantages', 'features', 'components', 'elements',
            'factors', 'aspects', 'categories', 'classifications',
            'options', 'alternatives', 'varieties', 'forms',
            'common', 'main', 'primary', 'key', 'important',
            'major', 'significant', 'top', 'best', 'popular'
        ]
        
        # Number indicators (including written numbers)
        number_patterns = [
            '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.',
            'first', 'second', 'third', 'fourth', 'fifth', 'sixth',
            'multiple', 'several', 'various', 'different', 'numerous',
            'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'
        ]
        
        # Question patterns that suggest lists
        question_patterns = [
            'what are', 'which are', 'how many', 'name the', 'identify the',
            'enumerate', 'describe the', 'explain the', 'outline the',
            'give me', 'show me', 'tell me about', 'provide', 'present'
        ]
        
        # Check for list indicators
        for keyword in list_keywords + number_patterns + question_patterns:
            if keyword in query_lower:
                return True
        
        return False
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics.
        
        Returns:
            Dict[str, Any]: System statistics
        """
        try:
            # Get vector store stats
            vector_stats = self.vector_store.get_collection_stats()
            
            # Add system-level stats
            system_stats = {
                'rag_model': self.llm_model,
                'embedding_model': vector_stats.get('embedding_model', 'unknown'),
                'vector_store_stats': vector_stats,
                'unified_embedding': True,
                'cot_enabled': config.COT_ENABLED_BY_DEFAULT
            }
            
            return system_stats
            
        except Exception as e:
            self.logger.error(f"Error getting system stats: {e}")
            return {
                'rag_model': self.llm_model,
                'embedding_model': 'unknown',
                'error': str(e)
            }
    
    def _enhance_with_sequential_chunks(self, search_results: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Enhance search results by retrieving sequential chunks for list queries.
        
        Args:
            search_results (Dict[str, Any]): Original search results
            query (str): The original query
        
        Returns:
            Dict[str, Any]: Enhanced search results with sequential chunks
        """
        try:
            # Check if we have results to enhance
            if search_results['n_results'] == 0:
                return search_results
            
            # Identify chunks that might be part of a sequence
            sequential_chunks = []
            chunk_groups = {}  # Group chunks by file
            
            # Group existing chunks by file and sort by chunk_index
            for i, doc in enumerate(search_results['documents']):
                metadata = doc.get('metadata', {})
                file_name = metadata.get('file_name', 'unknown')
                chunk_index = metadata.get('chunk_index', -1)
                
                if file_name not in chunk_groups:
                    chunk_groups[file_name] = []
                
                chunk_groups[file_name].append({
                    'doc': doc,
                    'distance': search_results['distances'][i],
                    'chunk_index': chunk_index,
                    'original_index': i
                })
            
            # For each file, check if we need to retrieve adjacent chunks
            enhanced_docs = []
            enhanced_distances = []
            
            for file_name, chunks in chunk_groups.items():
                # Sort chunks by index
                chunks.sort(key=lambda x: x['chunk_index'])
                
                # Look for indicators that this might be a partial list
                needs_expansion = False
                for chunk in chunks:
                    content = chunk['doc']['content'].lower()
                    # Check for list indicators
                    if any(indicator in content for indicator in [
                        'below are', 'five types', 'applications', 'the following',
                        '1.', '2.', '3.', '4.', '5.', 'first', 'second', 'third'
                    ]):
                        needs_expansion = True
                        break
                
                if needs_expansion and len(chunks) > 0:
                    # Try to retrieve adjacent chunks
                    self.logger.info(f"Expanding chunks for file {file_name} to capture complete list")
                    
                    # Get the range of chunk indices we already have
                    min_chunk = min(chunk['chunk_index'] for chunk in chunks if chunk['chunk_index'] >= 0)
                    max_chunk = max(chunk['chunk_index'] for chunk in chunks if chunk['chunk_index'] >= 0)
                    
                    # Try to get a few chunks before and after
                    target_range = range(max(0, min_chunk - 1), max_chunk + 4)  # Expand by 1 before, 3 after
                    
                    # Search for chunks in this range
                    additional_chunks = self._retrieve_chunks_by_range(file_name, target_range)
                    
                    # Merge original chunks with additional ones
                    all_chunks = chunks.copy()
                    
                    for add_chunk in additional_chunks:
                        # Check if we already have this chunk
                        if not any(existing['chunk_index'] == add_chunk['chunk_index'] for existing in all_chunks):
                            all_chunks.append(add_chunk)
                    
                    # Sort by chunk index and add to results
                    all_chunks.sort(key=lambda x: x['chunk_index'])
                    for chunk in all_chunks:
                        enhanced_docs.append(chunk['doc'])
                        enhanced_distances.append(chunk['distance'])
                else:
                    # No expansion needed, just add original chunks
                    for chunk in chunks:
                        enhanced_docs.append(chunk['doc'])
                        enhanced_distances.append(chunk['distance'])
            
            # Create enhanced results
            enhanced_results = {
                'documents': enhanced_docs,
                'distances': enhanced_distances,
                'n_results': len(enhanced_docs),
                'embedding_model': search_results.get('embedding_model', 'unknown')
            }
            
            self.logger.info(f"Enhanced search results: {search_results['n_results']} -> {enhanced_results['n_results']} chunks")
            return enhanced_results
            
        except Exception as e:
            self.logger.error(f"Error enhancing search results with sequential chunks: {e}")
            return search_results  # Return original results if enhancement fails
    
    def _retrieve_chunks_by_range(self, file_name: str, chunk_range: range) -> List[Dict[str, Any]]:
        """
        Retrieve chunks from a specific file within a given range.
        
        Args:
            file_name (str): Name of the file to search in
            chunk_range (range): Range of chunk indices to retrieve
        
        Returns:
            List[Dict[str, Any]]: List of chunks in the specified range
        """
        try:
            # Search for chunks from the specific file
            # We'll use a broad search and then filter
            broad_search = self.vector_store.search(
                f"content from {file_name}",
                n_results=50  # Get more results to have better chance of finding target chunks
            )
            
            matching_chunks = []
            for i, doc in enumerate(broad_search['documents']):
                metadata = doc.get('metadata', {})
                if file_name in metadata.get('file_name', ''):
                    chunk_index = metadata.get('chunk_index', -1)
                    if chunk_index in chunk_range:
                        matching_chunks.append({
                            'doc': doc,
                            'distance': broad_search['distances'][i],
                            'chunk_index': chunk_index
                        })
            
            return matching_chunks
            
        except Exception as e:
            self.logger.error(f"Error retrieving chunks by range: {e}")
            return []
