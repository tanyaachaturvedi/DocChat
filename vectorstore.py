import os
import glob
from typing import List, Dict, Any
import pickle
import re

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

from bedrock import get_embedding


class VectorStoreManager:
    """Manages FAISS vector store operations for RAG functionality"""
    
    def __init__(self, 
                 documents_dir: str = "documents",
                 chunk_size: int = 500,
                 chunk_overlap: int = 50):
        """
        Initialize VectorStoreManager
        
        Args:
            documents_dir: Directory containing .txt and .pdf files
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between consecutive chunks
        """
        self.documents_dir = documents_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Text splitter configuration
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize FAISS index
        self.index = None
        self.chunks = []
        self.metadatas = []
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF file using PyPDF2
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text from all pages
        """
        try:
            reader = PdfReader(pdf_path)
            all_text = []
            
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    
                    # Clean the extracted text
                    page_text = self.clean_pdf_text(page_text)
                    
                    # Only add non-empty pages
                    if page_text.strip():
                        all_text.append(f"[Page {page_num + 1}]\n{page_text}")
                        
                except Exception as e:
                    print(f"Error extracting text from page {page_num + 1} in {pdf_path}: {e}")
                    continue
            
            # Join all pages with double newline
            full_text = "\n\n".join(all_text)
            return full_text
            
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
            return ""
    
    def clean_pdf_text(self, text: str) -> str:
        """
        Clean extracted PDF text by removing excessive whitespace and artifacts
        
        Args:
            text: Raw text extracted from PDF
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove standalone numbers/letters that might be artifacts
        text = re.sub(r'\b[A-Z]\b', '', text)
        text = re.sub(r'\b\d+\b', '', text)
        
        # Remove common PDF artifacts
        text = re.sub(r'•|·', '', text)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
        
        # Clean up line breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def load_documents(self) -> List[str]:
        """
        Load all .txt and .pdf files from the documents directory
        
        Returns:
            List of text content from all files
        """
        # Get both .txt and .pdf files
        txt_files = glob.glob(os.path.join(self.documents_dir, "*.txt"))
        pdf_files = glob.glob(os.path.join(self.documents_dir, "*.pdf"))
        
        all_files = txt_files + pdf_files
        
        if not all_files:
            raise FileNotFoundError(f"No .txt or .pdf files found in {self.documents_dir}")
        
        documents = []
        file_info = []
        
        # Process .txt files
        for file_path in txt_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    if content.strip():  # Only add non-empty files
                        documents.append(content)
                        file_info.append({
                            'path': file_path,
                            'filename': os.path.basename(file_path),
                            'file_type': 'txt',
                            'page_number': None
                        })
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        # Process .pdf files
        for file_path in pdf_files:
            try:
                content = self.extract_text_from_pdf(file_path)
                if content.strip():  # Only add non-empty PDFs
                    documents.append(content)
                    file_info.append({
                        'path': file_path,
                        'filename': os.path.basename(file_path),
                        'file_type': 'pdf',
                        'page_number': None  # Will be updated during chunking
                    })
            except Exception as e:
                print(f"Error processing PDF {file_path}: {e}")
        
        # Store file info for metadata generation
        self.file_info = file_info
        
        print(f"Loaded {len(documents)} documents ({len(txt_files)} .txt, {len(pdf_files)} .pdf) from {self.documents_dir}")
        return documents
    
    def split_text(self, documents: List[str]) -> List[str]:
        """
        Split documents into smaller chunks with enhanced metadata
        
        Args:
            documents: List of document texts
            
        Returns:
            List of text chunks
        """
        all_chunks = []
        all_metadatas = []
        
        for i, doc in enumerate(documents):
            # Get file info for this document
            file_info = self.file_info[i] if i < len(self.file_info) else None
            
            chunks = self.text_splitter.split_text(doc)
            all_chunks.extend(chunks)
            
            # Create enhanced metadata for each chunk
            for chunk_idx, chunk in enumerate(chunks):
                metadata = {
                    "chunk_id": len(all_chunks) - 1,
                    "source_file": file_info['filename'] if file_info else f"document_{i}",
                    "file_type": file_info['file_type'] if file_info else "unknown",
                    "page_number": self._extract_page_number(chunk, file_info) if file_info and file_info['file_type'] == 'pdf' else None,
                    "chunk_length": len(chunk),
                    "source_document": f"document_{i}"
                }
                all_metadatas.append(metadata)
        
        self.chunks = all_chunks
        self.metadatas = all_metadatas
        
        print(f"Split documents into {len(all_chunks)} chunks")
        return all_chunks
    
    def _extract_page_number(self, chunk: str, file_info: Dict) -> int:
        """
        Extract page number from chunk text for PDF files
        
        Args:
            chunk: Text chunk
            file_info: File information dictionary
            
        Returns:
            Page number if found, None otherwise
        """
        # Look for [Page X] pattern in the chunk
        page_match = re.search(r'\[Page (\d+)\]', chunk)
        if page_match:
            return int(page_match.group(1))
        return None
    
    def create_embeddings(self) -> np.ndarray:
        """
        Create embeddings for all text chunks using AWS Bedrock
        
        Returns:
            numpy array of embeddings
        """
        if not self.chunks:
            raise ValueError("No chunks to embed. Run load_documents() and split_text() first.")
        
        print("Creating embeddings using AWS Bedrock...")
        embeddings = []
        
        for i, chunk in enumerate(self.chunks):
            try:
                # Use embedding function from bedrock.py
                embedding = get_embedding(chunk)
                embeddings.append(embedding)
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(self.chunks)} chunks")
                    
            except Exception as e:
                print(f"Error creating embedding for chunk {i}: {e}")
                # Use zero vector as fallback
                embeddings.append(np.zeros(1536))  # Assuming 1536 dimensions
        
        embeddings_array = np.array(embeddings).astype('float32')
        print(f"Created embeddings with shape: {embeddings_array.shape}")
        return embeddings_array
    
    def create_faiss_index(self, embeddings: np.ndarray):
        """
        Create and initialize FAISS index with embeddings
        
        Args:
            embeddings: numpy array of embeddings
        """
        dimension = embeddings.shape[1]
        
        # Initialize FAISS index (using L2 distance for similarity search)
        self.index = faiss.IndexFlatL2(dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        print(f"Created FAISS index with {self.index.ntotal} vectors of dimension {dimension}")
    
    def save_index(self):
        """Save FAISS index and metadata to disk"""
        if self.index is None:
            raise ValueError("No index to save. Create index first.")
        
        # Save FAISS index to faiss_index.bin
        faiss.write_index(self.index, "faiss_index.bin")
        
        # Save chunks and metadata to faiss_index_metadata.pkl
        with open("faiss_index_metadata.pkl", 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'metadatas': self.metadatas,
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap
            }, f)
        
        print("Index saved to faiss_index.bin")
        print("Metadata saved to faiss_index_metadata.pkl")
    
    def build_and_save_index(self):
        """
        Complete pipeline: Load documents, create embeddings, build index, and save
        """
        print("Building FAISS vector store...")
        
        # Load and process documents (both .txt and .pdf)
        documents = self.load_documents()
        chunks = self.split_text(documents)
        
        # Create embeddings
        embeddings = self.create_embeddings()
        
        # Create FAISS index
        self.create_faiss_index(embeddings)
        
        # Save index
        self.save_index()
        
        print("Vector store creation completed successfully!")
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks in the vector store
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of dictionaries containing chunk text, metadata, and similarity scores
        """
        if self.index is None:
            raise ValueError("No index loaded. Load or create index first.")
        
        # Create embedding for query
        query_embedding = get_embedding(query)
        query_embedding = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search for similar chunks
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):  # Valid index
                result = {
                    'text': self.chunks[idx],
                    'metadata': self.metadatas[idx],
                    'similarity_score': float(score)
                }
                results.append(result)
        
        return results


# Test functionality
if __name__ == "__main__":
    # Initialize vector store manager
    vector_store = VectorStoreManager(
        documents_dir="documents",
        chunk_size=500,
        chunk_overlap=50
    )
    
    # Build and save index
    try:
        vector_store.build_and_save_index()
        
        # Test search functionality
        query = "What is machine learning?"
        results = vector_store.search(query, k=3)
        
        print(f"\nSearch results for query: '{query}'")
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Text: {result['text'][:200]}...")
            print(f"File: {result['metadata']['source_file']} ({result['metadata']['file_type']})")
            if result['metadata']['page_number']:
                print(f"Page: {result['metadata']['page_number']}")
            print(f"Similarity Score: {result['similarity_score']:.4f}")
            
    except Exception as e:
        print(f"Error: {e}")
