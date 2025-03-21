import streamlit as st
import os
import numpy as np
import torch
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import re
import logging
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import nltk
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK data for text processing
try:
    nltk.download('punkt', quiet=True)
except:
    pass

# Constants
DEFAULT_TAMIL_TEXT = "இந்த மொழிபெயர்ப்பு செயல்முறை ஏற்கனவே தயாராக உள்ளது."
DEFAULT_ENGLISH_TEXT = "This translation process is already set up."

# Initialize session state variables
if 'translation_model' not in st.session_state:
    st.session_state.translation_model = None
if 'translation_tokenizer' not in st.session_state:
    st.session_state.translation_tokenizer = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'history' not in st.session_state:
    st.session_state.history = []

# Load environment variables
def load_env_variables():
    """Load environment variables or prompt for API keys."""
    # For deployed app, use environment variables
    groq_api_key = os.environ.get("GROQ_API_KEY", None)
    
    # For local development, prompt for API key if not found
    if not groq_api_key:
        if 'GROQ_API_KEY' in st.secrets:
            groq_api_key = st.secrets['GROQ_API_KEY']
        else:
            groq_api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")
            if groq_api_key:
                os.environ["GROQ_API_KEY"] = groq_api_key
            else:
                st.sidebar.warning("Please enter your Groq API Key to continue.")
                st.stop()
    
    return groq_api_key

# Language detection and translation
def detect_language(text: str) -> str:
    """Detect if the text is in Tamil or English."""
    # Tamil Unicode range: 0x0B80-0x0BFF
    tamil_pattern = re.compile(r'[\u0B80-\u0BFF]')
    if tamil_pattern.search(text):
        return "tamil"
    return "english"

def load_translation_model():
    """Load the translation model for Tamil-English translation."""
    if st.session_state.translation_model is None or st.session_state.translation_tokenizer is None:
        with st.spinner("Loading translation model... This might take a minute."):
            try:
                # Load MBart model for Tamil-English translation
                model_name = "facebook/mbart-large-50-many-to-many-mmt"
                st.session_state.translation_model = MBartForConditionalGeneration.from_pretrained(model_name)
                st.session_state.translation_tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
                return True
            except Exception as e:
                st.error(f"Error loading translation model: {e}")
                return False
    return True

def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """Translate text between Tamil and English."""
    # Map language codes
    lang_map = {
        "english": "en_XX",
        "tamil": "ta_IN"
    }
    
    try:
        # If model is not loaded, load it
        if not load_translation_model():
            if source_lang == "tamil" and target_lang == "english":
                return "Translation failed. Using a fallback method."
            return text
            
        # Set the source language
        st.session_state.translation_tokenizer.src_lang = lang_map[source_lang]
        
        # Encode the text
        encoded = st.session_state.translation_tokenizer(text, return_tensors="pt")
        
        # Generate translation
        generated_tokens = st.session_state.translation_model.generate(
            **encoded,
            forced_bos_token_id=st.session_state.translation_tokenizer.lang_code_to_id[lang_map[target_lang]]
        )
        
        # Decode the translation
        translation = st.session_state.translation_tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )[0]
        
        return translation
    except Exception as e:
        logger.error(f"Translation error: {e}")
        if source_lang == "tamil" and target_lang == "english":
            return "Translation failed. Using a fallback method."
        return text

# Vector database functions
def initialize_embeddings():
    """Initialize the embeddings model."""
    if st.session_state.embeddings is None:
        with st.spinner("Initializing embeddings model..."):
            try:
                model_name = "sentence-transformers/all-MiniLM-L6-v2"
                model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
                
                st.session_state.embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs=model_kwargs
                )
                logger.info(f"Initialized embeddings model: {model_name}")
            except Exception as e:
                st.error(f"Error initializing embeddings model: {e}")
                st.stop()

def load_vectorstore(index_path="police_legal_docs"):
    """Load the FAISS vector database."""
    if st.session_state.vectorstore is None:
        initialize_embeddings()
        
        with st.spinner("Loading vector database..."):
            try:
                st.session_state.vectorstore = FAISS.load_local(index_path, st.session_state.embeddings)
                logger.info(f"Loaded FAISS index from {index_path}")
            except Exception as e:
                st.error(f"Error loading vector database: {e}")
                st.warning("Please make sure you've created the vector database first.")
                st.stop()

# LLM and QA Chain
def initialize_llm():
    """Initialize the Groq LLM with Llama 70B."""
    if st.session_state.llm is None:
        groq_api_key = load_env_variables()
        
        with st.spinner("Initializing Llama 70B model..."):
            try:
                st.session_state.llm = ChatGroq(
                    model_name="llama-3.3-70b-versatile",
                    temperature=0.2,
                    groq_api_key=groq_api_key
                )
                logger.info("Initialized Llama 70B model")
            except Exception as e:
                st.error(f"Error initializing Llama 70B model: {e}")
                st.stop()

def create_qa_chain():
    """Create a retrieval QA chain with the vector database and LLM."""
    if st.session_state.qa_chain is None:
        # Make sure LLM and vectorstore are initialized
        initialize_llm()
        load_vectorstore()
        
        with st.spinner("Creating QA chain..."):
            try:
                # Create a custom prompt template for legal document retrieval
                prompt_template = """
                You are a legal assistant specializing in police procedures and legal documents.
                Use the following pieces of context to answer the question at the end.
                If you don't know the answer, just say that you don't know, don't try to make up an answer.
                
                Context:
                {context}
                
                Question: {question}
                
                Answer:
                """
                
                PROMPT = PromptTemplate(
                    template=prompt_template,
                    input_variables=["context", "question"]
                )
                
                # Create the QA chain
                st.session_state.qa_chain = RetrievalQA.from_chain_type(
                    llm=st.session_state.llm,
                    chain_type="stuff",
                    retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5}),
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": PROMPT}
                )
                logger.info("Created QA chain")
            except Exception as e:
                st.error(f"Error creating QA chain: {e}")
                st.stop()

# Web search function (simplified version without using Google Search API)
def web_search(query, num_results=3):
    """Simple web search simulation."""
    # This is a placeholder - in a real app, you'd implement a proper web search
    return f"Web search results for: {query}\n\nNote: Web search functionality is simplified in this version."

# Process query function (replacing CrewAI functionality)
def process_query(query: str) -> str:
    """Process a user query using vector search and LLM."""
    # Check if query is in Tamil and translate if needed
    original_language = detect_language(query)
    english_query = query
    
    if original_language == "tamil":
        with st.spinner("Translating your query from Tamil to English..."):
            english_query = translate_text(query, "tamil", "english")
    
    # Make sure QA chain is initialized
    if st.session_state.qa_chain is None:
        create_qa_chain()
    
    # Process the query
    with st.spinner("Searching legal documents..."):
        try:
            # Search through documents
            document_results = st.session_state.qa_chain({"query": english_query})
            document_answer = document_results["result"]
            
            # Get sources
            sources_text = "\n\nSources:\n"
            for i, doc in enumerate(document_results["source_documents"]):
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "Unknown")
                sources_text += f"{i+1}. {source}, Page {page}\n"
            
            # Format final response
            response = f"{document_answer}\n{sources_text}"
            
            # Translate response back to Tamil if original query was in Tamil
            if original_language == "tamil":
                with st.spinner("Translating response to Tamil..."):
                    response = translate_text(response, "english", "tamil")
            
            return response
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            error_msg = "Sorry, I encountered an error while processing your query."
            
            # Translate error message if original query was in Tamil
            if original_language == "tamil":
                error_msg = translate_text(error_msg, "english", "tamil")
            
            return error_msg

# Streamlit UI
def build_ui():
    """Build the Streamlit user interface."""
    st.title("Police Legal Document Assistant")
    st.subheader("Ask questions about police legal procedures in English or Tamil")
    
    # Sidebar for settings and actions
    with st.sidebar:
        st.header("Settings")
        
        # Display translation status
        if st.session_state.translation_model is not None:
            st.success("Translation model loaded ✓")
        else:
            st.info("Translation model will load on first use")
        
        # Display vector DB status
        if st.session_state.vectorstore is not None:
            st.success("Vector database loaded ✓")
        else:
            # Create a button to load the vector DB
            if st.button("Load Vector Database"):
                load_vectorstore()
                st.success("Vector database loaded ✓")
        
        # Display LLM status
        if st.session_state.llm is not None:
            st.success("Llama 70B model connected ✓")
        else:
            # Create a button to initialize the LLM
            if st.button("Connect to Llama 70B"):
                initialize_llm()
                st.success("Llama 70B model connected ✓")
        
        # Language examples
        st.header("Example Questions")
        st.markdown("**English:**")
        st.markdown("- What are the legal requirements for an arrest?")
        st.markdown("- How should evidence be collected at a crime scene?")
        st.markdown("- What rights do suspects have during interrogation?")
        
        st.markdown("**Tamil:**")
        st.markdown("- கைது செய்வதற்கான சட்ட தேவைகள் என்ன?")
        st.markdown("- குற்ற இடத்தில் ஆதாரங்களை எவ்வாறு சேகரிக்க வேண்டும்?")
        st.markdown("- விசாரணையின் போது சந்தேகத்திற்குரியவர்களுக்கு என்ன உரிமைகள் உள்ளன?")
        
        # Clear history button
        if st.button("Clear Conversation History"):
            st.session_state.history = []
            st.success("Conversation history cleared")
    
    # Chat interface
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for i, (role, message) in enumerate(st.session_state.history):
            if role == "user":
                st.chat_message("user").write(message)
            else:
                st.chat_message("assistant").write(message)
    
    # Query input
    user_query = st.chat_input("Ask a question in English or Tamil...")
    
    if user_query:
        # Add user query to history
        st.session_state.history.append(("user", user_query))
        
        # Display user query
        st.chat_message("user").write(user_query)
        
        # Process the query
        response = process_query(user_query)
        
        # Add response to history
        st.session_state.history.append(("assistant", response))
        
        # Display response
        st.chat_message("assistant").write(response)

# Main function
def main():
    st.set_page_config(
        page_title="Police Legal Document Assistant",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Build the UI
    build_ui()

if __name__ == "__main__":
    main()
