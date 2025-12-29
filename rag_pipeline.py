import os
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any, Optional

from bedrock import get_embedding, generate_text


class RAGPipeline:
    """
    Complete RAG pipeline for retrieval and generation
    """
    
    def __init__(self, 
                 index_path: str = "faiss_index.bin",
                 metadata_path: str = "faiss_index_metadata.pkl"):
        """
        Initialize RAG pipeline
        
        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata pickle file
        """
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.chunks = []
        self.metadatas = []
        self.is_loaded = False
    
    def load_vector_store(self) -> bool:
        """
        Load FAISS index and metadata from disk
        
        Returns:
            bool: True if loading successful, False otherwise
        """
        # Check if files exist
        if not os.path.exists(self.index_path):
            print(f"Index file not found: {self.index_path}")
            return False
            
        if not os.path.exists(self.metadata_path):
            print(f"Metadata file not found: {self.metadata_path}")
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(self.index_path)
            print(f"Loaded FAISS index with {self.index.ntotal} vectors")
            
            # Load metadata
            with open(self.metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                self.chunks = metadata['chunks']
                self.metadatas = metadata['metadatas']
            
            print(f"Loaded {len(self.chunks)} chunks and metadata")
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return False
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant chunks for a query using FAISS
        
        Args:
            query: Search query string
            top_k: Number of chunks to retrieve
            
        Returns:
            List of relevant chunks with metadata and scores
        """
        if not self.is_loaded:
            raise ValueError("Vector store not loaded. Call load_vector_store() first.")
        
        # Create embedding for the query using AWS Bedrock
        query_embedding = get_embedding(query)
        query_embedding = np.array([query_embedding]).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search for similar chunks
        similarity_scores, indices = self.index.search(query_embedding, top_k)
        
        relevant_chunks = []
        for score, idx in zip(similarity_scores[0], indices[0]):
            # Filter out invalid FAISS results
            # Valid indices are < len(chunks) and scores are reasonable
            if idx < len(self.chunks) and score < 1e10:  # Filter very large distances
                chunk_info = {
                    'text': self.chunks[idx],
                    'metadata': self.metadatas[idx],
                    'similarity_score': float(score),
                    'chunk_id': idx
                }
                relevant_chunks.append(chunk_info)
        
        print(f"Retrieved {len(relevant_chunks)} relevant chunks")
        return relevant_chunks
    
    def format_context(self, chunks: List[Dict]) -> str:
        """
        Concatenate retrieved chunks into a context string
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Formatted context string
        """
        if not chunks:
            return ""
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            # Add chunk with source information
            source_info = chunk['metadata'].get('source_document', 'Unknown')
            context_part = f"[Chunk {i} from {source_info}]:\n{chunk['text']}"
            context_parts.append(context_part)
        
        context = "\n\n".join(context_parts)
        return context
    
    def create_prompt(self, query: str, context: str) -> str:
        """
        Create a structured prompt combining context and question
        
        Args:
            query: User's original query
            context: Retrieved relevant text chunks
            
        Returns:
            Formatted prompt for text generation
        """
        if context:
            prompt = f"""Context:
{context}

Question:
{query}

Answer:"""
        else:
            prompt = f"""Question:
{query}

Answer:"""
        
        return prompt
    
    def query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Execute complete RAG pipeline for a query
        
        Args:
            query: User query string
            top_k: Number of chunks to retrieve
            
        Returns:
            Dictionary containing response and metadata
        """
        # Ensure vector store is loaded
        if not self.is_loaded:
            if not self.load_vector_store():
                raise ValueError("Failed to load vector store")
        
        # Retrieve relevant chunks
        relevant_chunks = self.retrieve_relevant_chunks(query, top_k=top_k)
        
        # Format context
        context = self.format_context(relevant_chunks)
        
        # Create prompt
        prompt = self.create_prompt(query, context)
        
        # Generate response using AWS Bedrock
        response = generate_text(prompt)
        
        # Prepare result
        result = {
            'query': query,
            'response': response,
            'num_chunks_retrieved': len(relevant_chunks),
            'similarity_scores': [chunk['similarity_score'] for chunk in relevant_chunks],
            'context': context,
            'chunks': relevant_chunks
        }
        
        return result


# Example usage and testing
if __name__ == "__main__":
    # Initialize RAG pipeline
    rag_pipeline = RAGPipeline(
        index_path="faiss_index.bin",
        metadata_path="faiss_index_metadata.pkl"
    )
    
    # Load vector store
    if rag_pipeline.load_vector_store():
        # Test query
        query = "What is machine learning?"
        result = rag_pipeline.query(query, top_k=3)
        
        print(f"Query: {result['query']}")
        print(f"Response: {result['response']}")
        print(f"Chunks retrieved: {result['num_chunks_retrieved']}")
        print(f"Similarity scores: {result['similarity_scores']}")
        print(f"Context:\n{result['context']}")
    else:
        print("Failed to load vector store. Make sure index files exist.")
        print("Run 'python vectorstore.py' to create the index first.")
