
import streamlit as st
import sys
import os

# Add current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_pipeline import RAGPipeline


def initialize_rag_pipeline():
    """Initialize and cache the RAG pipeline"""
    @st.cache_resource
    def create_pipeline():
        """Create and return RAG pipeline instance"""
        try:
            pipeline = RAGPipeline(
                index_path="faiss_index.bin",
                metadata_path="faiss_index_metadata.pkl"
            )
            
            # Try to load vector store
            if pipeline.load_vector_store():
                return pipeline
            else:
                return None
        except Exception as e:
            st.error(f"Error initializing RAG pipeline: {e}")
            return None
    
    return create_pipeline()


def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="RAG Q&A System",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Title and description
    st.title("🤖 RAG Question & Answer System")
    st.markdown("Ask questions about your documents using FAISS vector search and AWS Bedrock AI")
    
    # Initialize RAG pipeline
    pipeline = initialize_rag_pipeline()
    
    if pipeline is None:
        st.error("❌ **Failed to load vector store**")
        st.info("""
        Please ensure:
        - `faiss_index.bin` exists in the current directory
        - `faiss_index_metadata.pkl` exists in the current directory
        - You have run the vector store creation process first
        
        Run this command to create the vector store:
        ```bash
        python vectorstore.py
        ```
        """)
        return
    
    # Success message when pipeline is loaded
    st.success("✅ Vector store loaded successfully!")
    
    # Create input form
    with st.form("question_form"):
        st.subheader("Ask a Question")
        
        # Text input for user question
        question = st.text_input(
            "Enter your question:",
            placeholder="What would you like to know about your documents?",
            key="question_input"
        )
        
        # Number input for top_k chunks
        top_k = st.number_input(
            "Number of chunks to retrieve:",
            min_value=1,
            max_value=10,
            value=5,
            help="How many relevant chunks to find and include in the context"
        )
        
        # Form submission button
        submitted = st.form_submit_button("Ask Question", type="primary")
        
        # Handle form submission
        if submitted:
            if not question or question.strip() == "":
                st.warning("⚠️ Please enter a question before submitting.")
            else:
                # Process the question
                with st.spinner("Searching documents and generating answer..."):
                    try:
                        # Get answer from RAG pipeline
                        result = pipeline.query(question.strip(), top_k=int(top_k))
                        
                        # Display results
                        st.subheader("📝 Answer")
                        st.write(result['response'])
                        
                        # Display metadata
                        with st.expander("📊 Search Details"):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Chunks Retrieved", result['num_chunks_retrieved'])
                            
                            with col2:
                                if result['similarity_scores']:
                                    avg_score = sum(result['similarity_scores']) / len(result['similarity_scores'])
                                    st.metric("Avg Similarity", f"{avg_score:.3f}")
                                else:
                                    st.metric("Avg Similarity", "N/A")
                            
                            with col3:
                                if result['similarity_scores']:
                                    max_score = max(result['similarity_scores'])
                                    st.metric("Best Match", f"{max_score:.3f}")
                                else:
                                    st.metric("Best Match", "N/A")
                        
                        # Show retrieved context if user wants to see it
                        if result.get('context'):
                            with st.expander("📚 Retrieved Context"):
                                st.write(result['context'])
                        
                        # Show individual chunks with scores
                        if result.get('chunks'):
                            with st.expander("🔍 Individual Chunks"):
                                for i, chunk in enumerate(result['chunks'], 1):
                                    st.markdown(f"**Chunk {i}** (Score: {chunk['similarity_score']:.3f})")
                                    st.write(chunk['text'])
                                    st.markdown("---")
                    
                    except Exception as e:
                        st.error(f"❌ Error processing question: {e}")
                        st.info("Please check that your vector store files are valid and try again.")
    
    # Sidebar with information
    st.sidebar.header("ℹ️ About")
    st.sidebar.info("""
    This RAG system uses:
    
    • **FAISS** for vector similarity search
    • **AWS Bedrock** for AI text generation
    • **Amazon Titan** models for embeddings and generation
    
    Ask questions about your documents and get AI-powered answers!
    """)
    
    # Sidebar with instructions
    st.sidebar.header("📋 Instructions")
    st.sidebar.markdown("""
    1. Enter your question in the text box
    2. Choose how many chunks to retrieve
    3. Click "Ask Question"
    4. View the AI-generated answer
    5. Expand sections to see search details
    
    The system will search through your documents and provide relevant answers based on the content.
    """)
    
    # Sidebar with setup info
    st.sidebar.header("🔧 Setup")
    st.sidebar.markdown("""
    If you haven't set up the vector store yet:
    
    ```bash
    # 1. Add your documents to the documents/ folder
    # 2. Run the vector store creation
    python vectorstore.py
    
    # 3. Start the app
    streamlit run app.py
    ```
    """)
    
    # Example questions for user convenience
    st.sidebar.header("💡 Example Questions")
    example_questions = [
        "What is the main topic of the documents?",
        "Can you summarize the key points?",
        "What are the important concepts mentioned?",
        "How does the system work?",
        "What are the main features?"
    ]
    
    for example in example_questions:
        if st.sidebar.button(f"Ask: {example}", key=example):
            st.session_state.question_input = example
            st.rerun()


if __name__ == "__main__":
    main()
