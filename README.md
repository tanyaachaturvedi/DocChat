# 📄 DocChat — Document-Grounded RAG System using AWS Bedrock

DocChat is an end-to-end **Retrieval-Augmented Generation (RAG)** system that allows users to ask questions over their own documents (TXT and PDF) and receive **document-grounded answers** using **AWS Bedrock (Amazon Titan models)** and **FAISS**.

This project is intentionally designed to **avoid hallucination** by constraining the language model to answer based only on retrieved document context.

---
## 🎥 Project Demo

Click the thumbnail below to watch a working demo of DocChat:

[![DocChat Demo](https://www.youtube.com/watch?v=NA0gOrDO9zU)


## 🚀 Key Features

- 📂 **Document Ingestion**
  - Supports `.txt` and `.pdf` files
  - PDF text extraction using PyPDF2
  - Page-level metadata handling

- 🧠 **Semantic Retrieval**
  - Amazon Titan embeddings (`amazon.titan-embed-text-v1`)
  - FAISS vector store for fast similarity search

- 🤖 **Context-Aware Generation**
  - Amazon Titan text generation (`amazon.titan-text-express-v1`)
  - Answers generated using retrieved document context

- 🛡️ **Hallucination Control**
  - If relevant context is not found, the system refuses to speculate
  - Designed as a **pure document-grounded RAG system**

- 🌐 **Interactive UI**
  - Built with Streamlit
  - Adjustable retrieval depth (top-k chunks)
  - Inspect retrieved context and similarity scores

---

## 🧠 System Architecture

```
User Query → Query Embedding (Titan) → FAISS Vector Search → Relevant Document Chunks → Context Injection → AWS Bedrock (Titan Text) → Grounded Answer
``` 
---

## 🛠️ Tech Stack

- **Cloud AI**: AWS Bedrock  
  - `amazon.titan-embed-text-v1`
  - `amazon.titan-text-express-v1`
- **Vector Database**: FAISS  
- **Backend**: Python  
- **Frontend**: Streamlit  
- **PDF Processing**: PyPDF2  

---

## Project Structure

```
├── app.py                 # Streamlit web application
├── bedrock.py             # AWS Bedrock integration layer
├── vectorstore.py         # Document processing and vector indexing
├── rag_pipeline.py        # RAG retrieval and generation pipeline
├── requirements.txt       # Python dependencies
├── documents/             # Document storage directory
│   ├── sample.txt         # Example text document
│   └── *.pdf              # PDF documents
└── .gitignore            # Git ignore rules
```

## Quick Start

### Prerequisites
- Python 3.13+
- AWS account with Bedrock access
- AWS CLI configured with appropriate permissions

### Installation

1. **Clone and setup**
```bash
git clone <repository-url>
cd rag-document-qa
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure AWS credentials**
```bash
aws configure
# Ensure Bedrock access is enabled in your AWS account
```

4. **Prepare documents**
```bash
# Add your documents to the documents/ directory
cp your_document.pdf documents/
cp your_text.txt documents/
```

5. **Build vector index**
```bash
python vectorstore.py
```

6. **Launch application**
```bash
streamlit run app.py
```

Visit `http://localhost:8501` to use the interface.

## Usage

The web interface allows you to:
- Enter natural language questions about your documents
- Adjust the number of context chunks retrieved (1-10)
- View AI-generated answers with source attribution
- Examine retrieved chunks and similarity scores

## Design Decisions

### Why FAISS for Vector Search?
FAISS was chosen for its superior performance characteristics:
- **Speed**: Sub-millisecond similarity search through millions of vectors
- **Local Deployment**: No external dependencies or vendor lock-in
- **Scalability**: Efficiently handles high-dimensional embeddings
- **Reliability**: Battle-tested by Facebook/Meta in production

### How Hallucination is Prevented
The RAG architecture inherently reduces hallucinations through:
- **Grounded Generation**: Responses are based on retrieved document chunks
- **Context Limitation**: AI only sees relevant passages, not entire documents
- **Source Attribution**: Users can verify answers against original sources
- **Similarity Filtering**: Only high-similarity chunks are included in context

### Why AWS Bedrock?
AWS Bedrock provides several advantages:
- **Enterprise Security**: Built-in compliance and data governance
- **Managed Infrastructure**: No ML model management required
- **Cost Efficiency**: Pay-per-use pricing model
- **Model Diversity**: Access to multiple state-of-the-art models
- **Integration**: Seamless with existing AWS infrastructure

### Core Classes

**VectorStoreManager** (`vectorstore.py`)
- `load_documents()`: Load PDF/TXT files from documents directory
- `build_and_save_index()`: Complete ingestion and indexing pipeline
- `search(query, k)`: Retrieve similar chunks for a query

**RAGPipeline** (`rag_pipeline.py`)
- `load_vector_store()`: Load pre-built FAISS index
- `query(question, top_k)`: Execute complete RAG pipeline
- `retrieve_relevant_chunks()`: FAISS-based similarity search

**Bedrock Integration** (`bedrock.py`)
- `get_embedding(text)`: Generate text embeddings using Titan
- `generate_text(prompt)`: Generate responses using Titan



**Document Processing** (`vectorstore.py`)
```python
chunk_size=500        # Character limit per chunk
chunk_overlap=50      # Overlap between chunks
```

**Retrieval** (`rag_pipeline.py`)
```python
top_k=5              # Number of chunks to retrieve
min_score_threshold=0.1  # Similarity score filter
```

**Search Results** (`app.py`)
```python
k=5                  # Results per query in UI
```

## Performance Characteristics

- **Indexing Speed**: ~100-500 documents/minute (depends on content length)
- **Query Response**: <2 seconds end-to-end
- **Memory Usage**: ~1GB for 10K document chunks
- **Scalability**: Tested up to 100K document chunks

## Limitations

- AWS Bedrock costs apply per API call
- PDF processing quality depends on document formatting
- Context window limits may affect very long documents
- Similarity search is exact (not approximate) - may impact very large datasets



