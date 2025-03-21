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
from googleapiclient.discovery import build
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
GOOGLE_SEARCH_NUM_RESULTS = 5

# Set page config at the start
st.set_page_config(
    page_title="LexWay AI - Police Legal Document Assistant",
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
if 'google_search_api' not in st.session_state:
    st.session_state.google_search_api = None
if 'api_keys_entered' not in st.session_state:
    st.session_state.api_keys_entered = False

# Initialize global resources
@st.cache_resource
def get_api_keys():
    """Get API keys from environment variables or secrets."""
    keys = {
        "groq_api_key": os.environ.get("GROQ_API_KEY") or (st.secrets["GROQ_API_KEY"] if "GROQ_API_KEY" in st.secrets else None),
        "google_api_key": os.environ.get("GOOGLE_API_KEY") or (st.secrets["GOOGLE_API_KEY"] if "GOOGLE_API_KEY" in st.secrets else None),
        "search_engine_id": os.environ.get("SEARCH_ENGINE_ID") or (st.secrets["SEARCH_ENGINE_ID"] if "SEARCH_ENGINE_ID" in st.secrets else None)
    }
    return keys

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
        # Use 'cpu' by default to avoid CUDA issues
        device = 'cpu'
        model_kwargs = {'device': device}
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs=model_kwargs
        )
        logger.info(f"Initialized embeddings model: {EMBEDDING_MODEL} on {device}")
        return embeddings
    except Exception as e:
        logger.error(f"Error initializing embeddings model: {e}")
        return None

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

@st.cache_resource
def initialize_google_search(_api_key, _search_engine_id):
    """Initialize the Google Custom Search API."""
    if not _api_key or not _search_engine_id:
        logger.error("Missing Google API key or Search Engine ID")
        return None
    
    try:
        service = build("customsearch", "v1", developerKey=_api_key)
        logger.info("Initialized Google Custom Search API")
        return service, _search_engine_id
    except Exception as e:
        logger.error(f"Error initializing Google Search API: {e}")
        return None

@st.cache_resource
def create_qa_chain(_llm, _vectorstore):
    """Create a retrieval QA chain with the vector database and LLM."""
    try:
        # Create a custom prompt template for legal document retrieval
        prompt_template = """
        You are LexWay AI, an AI-powered police assistance chatbot designed to provide citizens with accurate and structured legal and procedural information. Your responses must strictly adhere to official police records and legal documents.

        Use the following pieces of context to answer the question. The context consists of both documents from our legal database and potentially relevant search results from authoritative sources.
        
        If you don't know the answer based on the context provided, say: "I'm sorry, but I cannot provide information on that. Please refer to the nearest police station or legal authority for assistance."
        
        Do not generate assumptions, opinions, or advice beyond the provided legal texts. Your response should be concise but complete.
        
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

def perform_google_search(query, google_search_api, search_engine_id):
    """Search Google for relevant information."""
    if not google_search_api or not search_engine_id:
        return []
    
    try:
        service, cse_id = google_search_api
        result = service.cse().list(
            q=query,
            cx=cse_id,
            num=GOOGLE_SEARCH_NUM_RESULTS
        ).execute()
        
        search_results = []
        if 'items' in result:
            for item in result['items']:
                search_results.append({
                    'title': item.get('title', ''),
                    'snippet': item.get('snippet', ''),
                    'link': item.get('link', '')
                })
        
        logger.info(f"Found {len(search_results)} Google search results")
        return search_results
    except Exception as e:
        logger.error(f"Error performing Google search: {e}")
        return []

# Process query function
def process_query(query: str, qa_chain, translation_model, translation_tokenizer, google_search_api=None, search_engine_id=None) -> str:
    """Process a user query using vector search, Google search, and LLM."""
    if not qa_chain:
        return "System is not properly initialized. Please try again later."
    
    # Check if query is in Tamil and translate if needed
    original_language = detect_language(query)
    english_query = query
    
    if original_language == "tamil":
        with st.spinner("Translating your query from Tamil to English..."):
            try:
                english_query = translate_text(query, "tamil", "english", translation_model, translation_tokenizer)
            except Exception as e:
                logger.error(f"Error translating query: {e}")
                return "Sorry, I encountered an error while translating your query."
    
    # Process the query with both vector search and Google search
    with st.spinner("Searching legal documents and online resources..."):
        try:
            # First search through local vector database
            document_results = qa_chain({"query": english_query})
            document_answer = document_results["result"]
            
            # Get sources from vector database
            sources_text = "\n\nDocument Sources:\n"
            for i, doc in enumerate(document_results["source_documents"]):
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "Unknown")
                sources_text += f"{i+1}. {source}, Page {page}\n"
            
            # Then search Google if API is available
            google_results_text = ""
            if google_search_api and search_engine_id:
                try:
                    search_results = perform_google_search(english_query, google_search_api, search_engine_id)
                    
                    if search_results:
                        google_results_text = "\n\nWeb Sources:\n"
                        for i, result in enumerate(search_results):
                            google_results_text += f"{i+1}. {result['title']}: {result['link']}\n"
                except Exception as e:
                    logger.error(f"Error performing Google search: {e}")
            
            # Format final response
            response = f"{document_answer}\n{sources_text}"
            if google_results_text:
                response += f"{google_results_text}"
            
            # Validate response against guidelines
            if "I don't know" in document_answer and not google_results_text:
                response = "I'm sorry, but I cannot provide information on that. Please refer to the nearest police station or legal authority for assistance."
            
            # Translate response back to Tamil if original query was in Tamil
            if original_language == "tamil":
                with st.spinner("Translating response to Tamil..."):
                    try:
                        response = translate_text(response, "english", "tamil", translation_model, translation_tokenizer)
                    except Exception as e:
                        logger.error(f"Error translating response: {e}")
                        return "Sorry, I encountered an error while translating the response."
            
            return response
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            error_msg = "Sorry, I encountered an error while processing your query."
            
            # Translate error message if original query was in Tamil
            if original_language == "tamil":
                try:
                    error_msg = translate_text(error_msg, "english", "tamil", translation_model, translation_tokenizer)
                except Exception as e:
                    logger.error(f"Error translating error message: {e}")
            
            return error_msg

def initialize_resources():
    """Initialize all required resources."""
    # Get API keys
    api_keys = get_api_keys()
    
    # If API keys not found, show input fields
    if (not api_keys["groq_api_key"] or not api_keys["google_api_key"] or not api_keys["search_engine_id"]) and not st.session_state.api_keys_entered:
        with st.sidebar:
            st.subheader("API Configuration")
            
            input_groq_api_key = st.text_input(
                "Enter your Groq API Key:",
                type="password",
                key="groq_api_key_input"
            )
            
            input_google_api_key = st.text_input(
                "Enter your Google API Key:",
                type="password",
                key="google_api_key_input"
            )
            
            input_search_engine_id = st.text_input(
                "Enter your Search Engine ID:",
                type="password",
                key="search_engine_id_input"
            )
            
            if st.button("Submit API Keys"):
                if input_groq_api_key and input_google_api_key and input_search_engine_id:
                    # Update environment variables
                    os.environ["GROQ_API_KEY"] = input_groq_api_key
                    os.environ["GOOGLE_API_KEY"] = input_google_api_key
                    os.environ["SEARCH_ENGINE_ID"] = input_search_engine_id
                    
                    # Update API keys
                    api_keys["groq_api_key"] = input_groq_api_key
                    api_keys["google_api_key"] = input_google_api_key
                    api_keys["search_engine_id"] = input_search_engine_id
                    
                    st.session_state.api_keys_entered = True
                    st.success("API keys successfully submitted!")
                    # Replace experimental_rerun with rerun
                    st.rerun()
                else:
                    st.error("Please enter all API keys.")
    
    if not api_keys["groq_api_key"] and not st.session_state.api_keys_entered:
        st.sidebar.warning("Please enter your API keys to continue.")
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
    if st.session_state.llm is None and api_keys["groq_api_key"]:
        with st.spinner("Initializing LLM model..."):
            llm = initialize_llm(api_keys["groq_api_key"])
            st.session_state.llm = llm
    
    # Initialize Google Search API
    if st.session_state.google_search_api is None and api_keys["google_api_key"] and api_keys["search_engine_id"]:
        with st.spinner("Initializing Google Search API..."):
            google_search_api = initialize_google_search(api_keys["google_api_key"], api_keys["search_engine_id"])
            st.session_state.google_search_api = google_search_api
    
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
    
    google_search_available = st.session_state.google_search_api is not None
    
    return (all_loaded, 
            api_keys, 
            st.session_state.translation_model, 
            st.session_state.translation_tokenizer, 
            st.session_state.qa_chain, 
            st.session_state.google_search_api)

def display_status():
    """Display the status of all components."""
    st.header("System Status")
    
    # Display component status
    components = {
        "Translation Model": st.session_state.translation_model is not None,
        "Vector Database": st.session_state.vectorstore is not None,
        "LLM Connection": st.session_state.llm is not None,
        "QA System": st.session_state.qa_chain is not None,
        "Google Search API": st.session_state.google_search_api is not None
    }
    
    for component, loaded in components.items():
        if loaded:
            st.success(f"{component} loaded ✓")
        else:
            st.error(f"{component} failed to load ✗")

def main():
    """Main function to run the Streamlit app."""
    st.title("LexWay AI - Police Legal Document Assistant")
    st.subheader("Ask questions about police legal procedures in English or Tamil")
    
    # Initialize all resources at startup
    all_loaded, api_keys, translation_model, translation_tokenizer, qa_chain, google_search_api = initialize_resources()
    
    # Sidebar for info and examples
    with st.sidebar:
        display_status()
        
        # Add online/offline mode toggle
        st.subheader("Search Settings")
        use_google_search = st.toggle("Use Google Search", value=google_search_api is not None)
        
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
        google_search_to_use = google_search_api if use_google_search else None
        search_engine_id_to_use = api_keys["search_engine_id"] if use_google_search else None
        
        response = process_query(
            user_query, 
            qa_chain, 
            translation_model, 
            translation_tokenizer,
            google_search_to_use,
            search_engine_id_to_use
        )
        
        # Add response to history
        st.session_state.history.append(("assistant", response))
        
        # Display response
        st.chat_message("assistant").write(response)

if __name__ == "__main__":
    main()
