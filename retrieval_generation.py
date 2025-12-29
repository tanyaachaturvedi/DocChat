import os
import pickle
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional

# Import custom functions from bedrock.py
from bedrock import get_embedding, generate_text


class RAGRetriever:
    """Handles retrieval of relevant chunks from FAISS vector store"""
    
    def __init__(self, 
                 index_path: str = "faiss_index.bin",
                 metadata_path: str = "faiss_index_metadata.pkl"):
        """
        Initialize RAG retriever
        
        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata pickle file
        """
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.chunks = []
        self.metadatas = []
        
    def load_vector_store(self) -> bool:
        """
        Load FAISS index and metadata from disk
        
        Returns:
            bool: True if loading successful, False otherwise
        """
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
            return True
            
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return False
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 5, 
                                min_score_threshold: float = 0.1) -> List[Dict]:
        """
        Retrieve top-k relevant chunks for a query using FAISS
        
        Args:
            query: Search query string
            top_k: Number of chunks to retrieve
            min_score_threshold: Minimum similarity score to include
            
        Returns:
            List of relevant chunks with metadata and scores
        """
        if self.index is None:
            raise ValueError("Vector store not loaded. Call load_vector_store() first.")
        
        # Create embedding for the query
        query_embedding = get_embedding(query)
        query_embedding = np.array([query_embedding]).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search for similar chunks
        similarity_scores, indices = self.index.search(query_embedding, top_k)
        
        relevant_chunks = []
        for score, idx in zip(similarity_scores[0], indices[0]):
            # Filter out invalid similarity scores
            if idx < len(self.chunks) and score >= min_score_threshold:
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


class RAGGenerator:
    """Handles text generation using Amazon Titan with retrieved context"""
    
    def __init__(self, model_id: str = "amazon.titan-text-lite-v1"):
        """
        Initialize RAG generator
        
        Args:
            model_id: Amazon Titan model ID
        """
        self.model_id = model_id
    
    def create_prompt(self, query: str, context: str) -> str:
        """
        Create a prompt combining user query and retrieved context
        
        Args:
            query: User's original query
            context: Retrieved relevant text chunks
            
        Returns:
            Formatted prompt for text generation
        """
        if context:
            prompt = f"""Based on the following context information, please answer the user's question. If the context doesn't contain relevant information to answer the question, please state that you don't have enough information.

Context:
{context}

Question: {query}

Answer:"""
        else:
            prompt = f"""Question: {query}

Answer:"""
        
        return prompt
    
    def generate_response(self, query: str, context: str, 
                         max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """
        Generate response using Amazon Titan text model
        
        Args:
            query: User's query
            context: Retrieved context information
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            
        Returns:
            Generated response text
        """
        # Create prompt
        prompt = self.create_prompt(query, context)
        
        # Generate text using bedrock function
        response = generate_text(
            prompt=prompt,
            model_id=self.model_id,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return response


class RAGPipeline:
    """Complete RAG pipeline: retrieval + generation"""
    
    def __init__(self, 
                 index_path: str = "faiss_index.bin",
                 metadata_path: str = "faiss_index_metadata.pkl",
                 top_k: int = 5,
                 min_score_threshold: float = 0.1):
        """
        Initialize complete RAG pipeline
        
        Args:
            index_path: Path to FAISS index
            metadata_path: Path to metadata
            top_k: Number of chunks to retrieve
            min_score_threshold: Minimum similarity score
        """
        self.retriever = RAGRetriever(index_path, metadata_path)
        self.generator = RAGGenerator()
        self.top_k = top_k
        self.min_score_threshold = min_score_threshold
    
    def load_vector_store(self) -> bool:
        """Load the vector store"""
        return self.retriever.load_vector_store()
    
    def query(self, query: str, return_context: bool = False) -> Dict:
        """
        Execute complete RAG pipeline for a query
        
        Args:
            query: User query string
            return_context: Whether to return retrieved context
            
        Returns:
            Dictionary containing response and optionally context
        """
        # Retrieve relevant chunks
        relevant_chunks = self.retriever.retrieve_relevant_chunks(
            query, 
            top_k=self.top_k,
            min_score_threshold=self.min_score_threshold
        )
        
        # Format context
        context = self.retriever.format_context(relevant_chunks)
        
        # Generate response
        response = self.generator.generate_response(query, context)
        
        # Prepare result
        result = {
            'query': query,
            'response': response,
            'num_chunks_retrieved': len(relevant_chunks),
            'similarity_scores': [chunk['similarity_score'] for chunk in relevant_chunks]
        }
        
        if return_context:
            result['context'] = context
            result['chunks'] = relevant_chunks
        
        return result


# Example usage and testing
if __name__ == "__main__":
    # Initialize RAG pipeline
    rag_pipeline = RAGPipeline(
        index_path="faiss_index.bin",
        metadata_path="faiss_index_metadata.pkl",
        top_k=5,
        min_score_threshold=0.1
    )
    
    # Load vector store
    if rag_pipeline.load_vector_store():
        # Test query
        query = "What is machine learning?"
        result = rag_pipeline.query(query, return_context=True)
        
        print(f"Query: {result['query']}")
        print(f"Response: {result['response']}")
        print(f"Chunks retrieved: {result['num_chunks_retrieved']}")
        print(f"Similarity scores: {result['similarity_scores']}")
        print(f"Context:\n{result['context']}")
    else:
        print("Failed to load vector store. Make sure index files exist.")
