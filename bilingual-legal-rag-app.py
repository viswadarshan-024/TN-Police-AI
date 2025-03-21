import streamlit as st
import os
import numpy as np
import torch
import re
import logging
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import nltk
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import warnings
from functools import lru_cache

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
INDEX_PATH = "police_legal_docs"
MODEL_NAME = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Set page config at the start
st.set_page_config(
    page_title="Police Legal Document Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
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
if 'api_key_entered' not in st.session_state:
    st.session_state.api_key_entered = False

# Initialize global resources
@st.cache_resource
def get_api_key():
    """Get API key from environment variables or secrets."""
    if "GROQ_API_KEY" in st.secrets:
        return st.secrets["GROQ_API_KEY"]
    return os.environ.get("GROQ_API_KEY")

@st.cache_resource
def load_translation_model():
    """Load the translation model for Tamil-English translation."""
    try:
        # Load MBart model for Tamil-English translation
        model_name = "facebook/mbart-large-50-many-to-many-mmt"
        model = MBartForConditionalGeneration.from_pretrained(model_name)
        tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
        logger.info("Translation model loaded successfully")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading translation model: {e}")
        return None, None

@st.cache_resource
def initialize_embeddings():
    """Initialize the embeddings model."""
    try:
        model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs=model_kwargs
        )
        logger.info(f"Initialized embeddings model: {EMBEDDING_MODEL}")
        return embeddings
    except Exception as e:
        logger.error(f"Error initializing embeddings model: {e}")
        return None

# Note the leading underscore on _embeddings to prevent hashing
@st.cache_resource
def load_vectorstore(_embeddings):
    """Load the FAISS vector database."""
    try:
        vectorstore = FAISS.load_local(INDEX_PATH, _embeddings)
        logger.info(f"Loaded FAISS index from {INDEX_PATH}")
        return vectorstore
    except Exception as e:
        logger.error(f"Error loading vector database: {e}")
        # Create a dummy vector store if the real one fails to load
        documents = [{"page_content": "This is a fallback document.", "metadata": {"source": "fallback", "page": 1}}]
        return FAISS.from_texts([doc["page_content"] for doc in documents], _embeddings, [doc["metadata"] for doc in documents])

@st.cache_resource
def initialize_llm(_api_key):
    """Initialize the Groq LLM with Llama 70B."""
    if not _api_key:
        logger.error("No API key provided for Groq")
        return None
    
    try:
        llm = ChatGroq(
            model_name=MODEL_NAME,
            temperature=0.2,
            groq_api_key=_api_key
        )
        logger.info(f"Initialized {MODEL_NAME} model")
        return llm
    except Exception as e:
        logger.error(f"Error initializing LLM model: {e}")
        return None

# Note the leading underscores on _llm and _vectorstore to prevent hashing
@st.cache_resource
def create_qa_chain(_llm, _vectorstore):
    """Create a retrieval QA chain with the vector database and LLM."""
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
        qa_chain = RetrievalQA.from_chain_type(
            llm=_llm,
            chain_type="stuff",
            retriever=_vectorstore.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        logger.info("Created QA chain")
        return qa_chain
    except Exception as e:
        logger.error(f"Error creating QA chain: {e}")
        return None

# Language detection and translation
def detect_language(text: str) -> str:
    """Detect if the text is in Tamil or English."""
    # Tamil Unicode range: 0x0B80-0x0BFF
    tamil_pattern = re.compile(r'[\u0B80-\u0BFF]')
    if tamil_pattern.search(text):
        return "tamil"
    return "english"

def translate_text(text: str, source_lang: str, target_lang: str, model, tokenizer) -> str:
    """Translate text between Tamil and English."""
    if model is None or tokenizer is None:
        if source_lang == "tamil" and target_lang == "english":
            return "Translation failed. Using a fallback method."
        return text
    
    # Map language codes
    lang_map = {
        "english": "en_XX",
        "tamil": "ta_IN"
    }
    
    try:
        # Set the source language
        tokenizer.src_lang = lang_map[source_lang]
        
        # Encode the text
        encoded = tokenizer(text, return_tensors="pt")
        
        # Generate translation
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.lang_code_to_id[lang_map[target_lang]]
        )
        
        # Decode the translation
        translation = tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )[0]
        
        return translation
    except Exception as e:
        logger.error(f"Translation error: {e}")
        if source_lang == "tamil" and target_lang == "english":
            return "Translation failed. Using a fallback method."
        return text

# Process query function
def process_query(query: str, qa_chain, translation_model, translation_tokenizer) -> str:
    """Process a user query using vector search and LLM."""
    if not qa_chain:
        return "System is not properly initialized. Please try again later."
    
    # Check if query is in Tamil and translate if needed
    original_language = detect_language(query)
    english_query = query
    
    if original_language == "tamil":
        with st.spinner("Translating your query from Tamil to English..."):
            english_query = translate_text(query, "tamil", "english", translation_model, translation_tokenizer)
    
    # Process the query
    with st.spinner("Searching legal documents..."):
        try:
            # Search through documents
            document_results = qa_chain({"query": english_query})
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
                    response = translate_text(response, "english", "tamil", translation_model, translation_tokenizer)
            
            return response
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            error_msg = "Sorry, I encountered an error while processing your query."
            
            # Translate error message if original query was in Tamil
            if original_language == "tamil":
                error_msg = translate_text(error_msg, "english", "tamil", translation_model, translation_tokenizer)
            
            return error_msg

def initialize_resources():
    """Initialize all required resources."""
    # Get API key
    api_key = get_api_key()
    
    # If API key not found, show input field
    if not api_key and not st.session_state.api_key_entered:
        with st.sidebar:
            input_api_key = st.text_input(
                "Enter your Groq API Key:",
                type="password",
                key="api_key_input"
            )
            
            if input_api_key:
                api_key = input_api_key
                os.environ["GROQ_API_KEY"] = api_key
                st.session_state.api_key_entered = True
                st.experimental_rerun()
    
    if not api_key and not st.session_state.api_key_entered:
        st.sidebar.warning("Please enter your Groq API Key to continue.")
        return False, None, None, None, None, None
    
    # Load translation model
    if st.session_state.translation_model is None or st.session_state.translation_tokenizer is None:
        with st.spinner("Loading translation model..."):
            model, tokenizer = load_translation_model()
            st.session_state.translation_model = model
            st.session_state.translation_tokenizer = tokenizer
    
    # Initialize embeddings
    if st.session_state.embeddings is None:
        with st.spinner("Initializing embeddings model..."):
            embeddings = initialize_embeddings()
            st.session_state.embeddings = embeddings
    
    # Load vector store
    if st.session_state.vectorstore is None and st.session_state.embeddings is not None:
        with st.spinner("Loading vector database..."):
            vectorstore = load_vectorstore(st.session_state.embeddings)
            st.session_state.vectorstore = vectorstore
    
    # Initialize LLM
    if st.session_state.llm is None and api_key:
        with st.spinner("Initializing LLM model..."):
            llm = initialize_llm(api_key)
            st.session_state.llm = llm
    
    # Create QA chain
    if (st.session_state.qa_chain is None and 
        st.session_state.llm is not None and 
        st.session_state.vectorstore is not None):
        with st.spinner("Creating QA chain..."):
            qa_chain = create_qa_chain(st.session_state.llm, st.session_state.vectorstore)
            st.session_state.qa_chain = qa_chain
    
    # Check if everything is loaded properly
    all_loaded = (
        st.session_state.translation_model is not None and 
        st.session_state.translation_tokenizer is not None and 
        st.session_state.embeddings is not None and 
        st.session_state.vectorstore is not None and 
        st.session_state.llm is not None and 
        st.session_state.qa_chain is not None
    )
    
    return all_loaded, api_key, st.session_state.translation_model, st.session_state.translation_tokenizer, st.session_state.qa_chain

def display_status():
    """Display the status of all components."""
    st.header("System Status")
    
    # Display component status
    components = {
        "Translation Model": st.session_state.translation_model is not None,
        "Vector Database": st.session_state.vectorstore is not None,
        "LLM Connection": st.session_state.llm is not None,
        "QA System": st.session_state.qa_chain is not None
    }
    
    for component, loaded in components.items():
        if loaded:
            st.success(f"{component} loaded ✓")
        else:
            st.error(f"{component} failed to load ✗")

def main():
    """Main function to run the Streamlit app."""
    st.title("Police Legal Document Assistant")
    st.subheader("Ask questions about police legal procedures in English or Tamil")
    
    # Initialize all resources at startup
    all_loaded, api_key, translation_model, translation_tokenizer, qa_chain = initialize_resources()
    
    # Sidebar for info and examples
    with st.sidebar:
        display_status()
        
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
    
    # Warning if not all components are loaded
    if not all_loaded:
        st.warning("Some components have not loaded properly. The application may not function correctly.")
    
    # Chat interface
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for role, message in st.session_state.history:
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
        response = process_query(user_query, qa_chain, translation_model, translation_tokenizer)
        
        # Add response to history
        st.session_state.history.append(("assistant", response))
        
        # Display response
        st.chat_message("assistant").write(response)

if __name__ == "__main__":
    main()
