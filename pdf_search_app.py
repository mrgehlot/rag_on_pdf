"""
Streamlit app for PDF text extraction, embedding generation, and semantic search.
Uses Gemma 300M embeddings and ChromaDB for vector storage.
"""

import streamlit as st
import fitz  # PyMuPDF
import chromadb
from chromadb.config import Settings
from transformers import AutoTokenizer, AutoModel
import torch
import re
import os
from typing import List, Tuple
import nltk
from nltk.tokenize import sent_tokenize

# Download NLTK data if not already present
def ensure_nltk_data():
    """Ensure NLTK punkt_tab tokenizer is downloaded."""
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        try:
            nltk.download('punkt_tab', quiet=True)
        except Exception as e:
            # Try alternative download methods
            try:
                nltk.download('punkt', quiet=True)
            except:
                # If all else fails, download punkt_tab with verbose output
                print(f"Attempting to download punkt_tab... Error: {e}")
                nltk.download('punkt_tab', quiet=False)

# Ensure NLTK data is available
ensure_nltk_data()

# Page configuration
st.set_page_config(
    page_title="PDF Semantic Search",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'collection' not in st.session_state:
    st.session_state.collection = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'device' not in st.session_state:
    st.session_state.device = None
if 'chroma_client' not in st.session_state:
    st.session_state.chroma_client = None
if 'processed_pdf' not in st.session_state:
    st.session_state.processed_pdf = None

@st.cache_resource
def load_embedding_model():
    """Load the Gemma 300M embedding model and tokenizer."""
    model_name = "google/embeddinggemma-300m"
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model = model.to(device)
        model.eval()  # Set to evaluation mode
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Make sure you have internet connection and sufficient disk space for model download.")
        return None, None, None

@st.cache_resource
def initialize_chromadb():
    """Initialize ChromaDB client with persistent storage."""
    try:
        # Create a persistent client that stores data locally
        client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        return client
    except Exception as e:
        st.error(f"Error initializing ChromaDB: {str(e)}")
        return None

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from a PDF file."""
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            text += page_text + "\n"
        doc.close()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation
    text = text.strip()
    return text

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using NLTK."""
    # Ensure NLTK data is available
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        try:
            nltk.download('punkt_tab', quiet=True)
        except:
            try:
                nltk.download('punkt', quiet=True)
            except:
                pass
    
    # Use NLTK for better sentence tokenization
    try:
        sentences = sent_tokenize(text)
    except LookupError:
        # Fallback to simple sentence splitting if NLTK fails
        sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
    
    # Filter out very short sentences (likely artifacts)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    return sentences

def create_paragraphs(sentences: List[str], sentences_per_paragraph: int = 2) -> List[str]:
    """Create paragraphs with specified number of sentences."""
    paragraphs = []
    for i in range(0, len(sentences), sentences_per_paragraph):
        paragraph = ' '.join(sentences[i:i + sentences_per_paragraph])
        paragraph = clean_text(paragraph)
        if paragraph:  # Only add non-empty paragraphs
            paragraphs.append(paragraph)
    return paragraphs

def generate_embedding(text: str, tokenizer, model, device: str = "cpu") -> List[float]:
    """Generate embedding for a given text using Gemma 300M model."""
    try:
        # Tokenize the input
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512  # Adjust based on model's max length
        )
        
        # Move inputs to the same device as model
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate embedding
        with torch.no_grad():
            outputs = model(**inputs)
            # Use mean pooling of the last hidden state
            # Handle both single and batch cases
            if len(outputs.last_hidden_state.shape) == 3:
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
            else:
                embedding = outputs.last_hidden_state.mean(dim=0)
            
            # Convert to list (move to CPU first if on GPU)
            if embedding.is_cuda:
                embedding = embedding.cpu()
            if embedding.dim() == 0:
                embedding = embedding.unsqueeze(0)
            embedding = embedding.tolist()
        
        return embedding
    except Exception as e:
        st.error(f"Error generating embedding: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

def process_pdf(uploaded_file, tokenizer, model, client, device):
    """Process PDF: extract text, create paragraphs, generate embeddings, and store in ChromaDB."""
    # Extract text
    with st.spinner("Extracting text from PDF..."):
        text = extract_text_from_pdf(uploaded_file)
        if not text:
            return False, 0
    
    st.success(f"Extracted {len(text)} characters from PDF")
    
    # Split into sentences
    with st.spinner("Splitting text into sentences..."):
        sentences = split_into_sentences(text)
        st.info(f"Found {len(sentences)} sentences")
    
    # Create paragraphs
    with st.spinner("Creating paragraphs (2 sentences each)..."):
        paragraphs = create_paragraphs(sentences, sentences_per_paragraph=2)
        st.info(f"Created {len(paragraphs)} paragraphs")
    
    # Create or get collection
    collection_name = f"pdf_{uploaded_file.name.replace('.', '_').replace(' ', '_')}"
    try:
        # Try to get existing collection
        collection = client.get_collection(name=collection_name)
        # Clear existing data if collection exists
        collection.delete()
        collection = client.create_collection(name=collection_name)
    except:
        # Create new collection if it doesn't exist
        collection = client.create_collection(name=collection_name)
    
    # Generate embeddings and store
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    embeddings_generated = 0
    for idx, paragraph in enumerate(paragraphs):
        # Update progress
        progress = (idx + 1) / len(paragraphs)
        progress_bar.progress(progress)
        status_text.text(f"Generating embeddings: {idx + 1}/{len(paragraphs)}")
        
        # Generate embedding
        embedding = generate_embedding(paragraph, tokenizer, model, device)
        if embedding:
            # Store in ChromaDB
            collection.add(
                ids=[f"para_{idx}"],
                embeddings=[embedding],
                documents=[paragraph],
                metadatas=[{"paragraph_id": idx, "source": uploaded_file.name}]
            )
            embeddings_generated += 1
    
    progress_bar.empty()
    status_text.empty()
    
    st.session_state.collection = collection
    st.session_state.processed_pdf = uploaded_file.name
    
    return True, embeddings_generated

def search_query(query: str, collection, tokenizer, model, device: str, top_k: int = 5):
    """Search the vector database for relevant paragraphs."""
    if not query or not collection:
        return None
    
    # Generate query embedding
    query_embedding = generate_embedding(query, tokenizer, model, device)
    if not query_embedding:
        return None
    
    # Search in ChromaDB
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results
    except Exception as e:
        st.error(f"Error searching: {str(e)}")
        return None

st.title("📚 PDF Semantic Search with Gemma 300M")
st.markdown("Upload a PDF, extract text, generate embeddings, and search using natural language queries.")

# Sidebar for model initialization
with st.sidebar:
    st.header("⚙️ Setup")
    
    if st.button("Initialize Model & Database"):
        with st.spinner("Loading Gemma 300M model (this may take a few minutes on first run)..."):
            tokenizer, model, device = load_embedding_model()
            if tokenizer and model:
                st.session_state.tokenizer = tokenizer
                st.session_state.model = model
                st.session_state.device = device
                device_text = "GPU" if device == "cuda" else "CPU"
                st.success(f"✅ Model loaded successfully on {device_text}!")
            else:
                st.error("❌ Failed to load model")
        
        with st.spinner("Initializing ChromaDB..."):
            client = initialize_chromadb()
            if client:
                st.session_state.chroma_client = client
                st.success("✅ ChromaDB initialized!")
            else:
                st.error("❌ Failed to initialize ChromaDB")
    
    st.markdown("---")
    st.markdown("### 📋 Instructions")
    st.markdown("""
    1. Click **Initialize Model & Database** first
    2. Upload your PDF file
    3. Wait for processing to complete
    4. Enter your search query
    5. View relevant results
    """)
    
    if st.session_state.processed_pdf:
        st.info(f"📄 Current PDF: {st.session_state.processed_pdf}")
        if st.button("Clear Current PDF"):
            st.session_state.processed_pdf = None
            st.session_state.collection = None
            st.rerun()

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload & Process PDF")
    
    # Check if model is loaded
    if not st.session_state.tokenizer or not st.session_state.model:
        st.warning("⚠️ Please initialize the model first using the sidebar.")
    elif not st.session_state.chroma_client:
        st.warning("⚠️ Please initialize ChromaDB first using the sidebar.")
    else:
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF file to extract and process"
        )
        
        if uploaded_file is not None:
            if st.button("Process PDF", type="primary"):
                success, count = process_pdf(
                    uploaded_file,
                    st.session_state.tokenizer,
                    st.session_state.model,
                    st.session_state.chroma_client,
                    st.session_state.device
                )
                if success:
                    st.success(f"✅ Successfully processed PDF! Generated {count} embeddings.")
                else:
                    st.error("❌ Failed to process PDF.")

with col2:
    st.header("🔍 Semantic Search")
    
    if not st.session_state.collection:
        st.info("👆 Process a PDF first to enable search.")
    else:
        query = st.text_input(
            "Enter your search query:",
            placeholder="e.g., What is the main topic?",
            help="Enter a natural language query to search the PDF content"
        )
        
        top_k = st.slider("Number of results:", min_value=1, max_value=10, value=5)
        
        if query and st.button("Search", type="primary"):
            with st.spinner("Searching..."):
                results = search_query(
                    query,
                    st.session_state.collection,
                    st.session_state.tokenizer,
                    st.session_state.model,
                    st.session_state.device,
                    top_k
                )
            
            if results and results.get('documents') and len(results['documents']) > 0:
                documents = results['documents'][0] if isinstance(results['documents'][0], list) else results['documents']
                metadatas = results.get('metadatas', [])
                if metadatas and len(metadatas) > 0:
                    metadatas = metadatas[0] if isinstance(metadatas[0], list) else metadatas
                else:
                    metadatas = [{}] * len(documents)
                
                if len(documents) > 0:
                    st.success(f"Found {len(documents)} relevant results:")
                    st.markdown("---")
                    
                    for idx, doc in enumerate(documents, 1):
                        metadata = metadatas[idx - 1] if idx - 1 < len(metadatas) else {}
                        with st.expander(f"Result {idx} - Paragraph {metadata.get('paragraph_id', 'N/A')}", expanded=idx == 1):
                            st.write(doc)
                            if metadata and metadata.get('source'):
                                st.caption(f"Source: {metadata.get('source', 'Unknown')}")
                else:
                    st.warning("No results found. Try a different query.")
            else:
                st.warning("No results found. Try a different query.")

