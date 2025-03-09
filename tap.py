import streamlit as st
import logging
import os
import pandas as pd
import numpy as np
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from typing import Dict, List, Any, Tuple, Optional
import json
import traceback
import time
import faiss
import requests
from bs4 import BeautifulSoup
import html2text
from datetime import datetime
from data_loader import DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Tap Bonds AI Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "bond_details" not in st.session_state:
    st.session_state.bond_details = None
if "cashflow_details" not in st.session_state:
    st.session_state.cashflow_details = None
if "company_insights" not in st.session_state:
    st.session_state.company_insights = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "embeddings_dict" not in st.session_state:
    st.session_state.embeddings_dict = {}
if "data_loading_status" not in st.session_state:
    st.session_state.data_loading_status = {
        "bond": {"status": "not_started", "message": "Not loaded"},
        "cashflow": {"status": "not_started", "message": "Not loaded"},
        "company": {"status": "not_started", "message": "Not loaded"},
        "web": {"status": "not_started", "message": "Not loaded"}
    }
if "last_load_attempt" not in st.session_state:
    st.session_state.last_load_attempt = 0
if "search_results" not in st.session_state:
    st.session_state.search_results = {}
if "web_cache" not in st.session_state:
    st.session_state.web_cache = {}
if "data_loader" not in st.session_state:
    st.session_state.data_loader = DataLoader()

class VectorStore:
    """Vector store for efficient similarity search"""
    def __init__(self, dimension=768):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.texts = []

    def add_texts(self, texts: List[str], embeddings: List[List[float]]):
        if not texts or not embeddings:
            return
        embeddings_array = np.array(embeddings).astype('float32')
        self.index.add(embeddings_array)
        self.texts.extend(texts)

    def similarity_search(self, query_embedding: List[float], k: int = 5) -> List[str]:
        query_array = np.array([query_embedding]).astype('float32')
        D, I = self.index.search(query_array, k)
        return [self.texts[i] for i in I[0] if i < len(self.texts)]

@st.cache_data(ttl=3600)
def fetch_web_content(url: str) -> str:
    """Fetch and parse web content with caching"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        h = html2text.HTML2Text()
        h.ignore_links = True
        return h.handle(soup.get_text())
    except Exception as e:
        logger.error(f"Error fetching web content: {str(e)}")
        return ""

@st.cache_data(ttl=3600)
def perform_web_search(query: str, num_results: int = 3) -> List[Dict]:
    """Perform web search with content fetching"""
    try:
        search = DuckDuckGoSearchAPIWrapper()
        results = search.results(query, num_results)

        # Enhance results with content
        enhanced_results = []
        for result in results:
            if result["link"] not in st.session_state.web_cache:
                content = fetch_web_content(result["link"])
                st.session_state.web_cache[result["link"]] = {
                    "content": content,
                    "timestamp": datetime.now()
                }
            enhanced_results.append({
                **result,
                "content": st.session_state.web_cache[result["link"]]["content"]
            })
        return enhanced_results
    except Exception as e:
        logger.error(f"Error in web search: {str(e)}")
        return []

@st.cache_resource
def get_llm(api_key: str, model: str = "llama3-70b-8192"):
    """Initialize LLM with caching"""
    try:
        if not api_key:
            return None
        return ChatGroq(
            api_key=api_key,
            model=model,
            temperature=0.2,
            max_tokens=4000
        )
    except Exception as e:
        logger.error(f"Error initializing LLM: {str(e)}")
        return None

def process_query(query: str, context: Dict, llm) -> Dict:
    """Process query using LLM and available data"""
    try:
        if not llm:
            return {"error": "AI model not initialized. Please check API key."}

        # Perform web search for additional context
        web_results = perform_web_search(query)
        web_context = "\n".join(result.get("content", "") for result in web_results)

        # Create enhanced prompt
        template = """You are a financial expert specializing in bond analysis.

        User Query: {query}

        Available Data:
        Bond Data: {bond_data}
        Cashflow Data: {cashflow_data}
        Company Data: {company_data}

        Additional Web Context:
        {web_context}

        Please provide a comprehensive analysis based on all available information.
        Format your response using Markdown for better readability.
        Include relevant web information when appropriate.
        """

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()

        response = chain.invoke({
            "query": query,
            "bond_data": str(context.get("bond_data")),
            "cashflow_data": str(context.get("cashflow_data")),
            "company_data": str(context.get("company_data")),
            "web_context": web_context
        })

        return {
            "response": response,
            "web_results": web_results
        }

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return {"error": f"Error processing query: {str(e)}"}

def display_status():
    """Display data loading status"""
    cols = st.columns(len(st.session_state.data_loading_status))
    for col, (key, value) in zip(cols, st.session_state.data_loading_status.items()):
        with col:
            status_icon = "✅" if value["status"] == "success" else "❌" if value["status"] == "error" else "⏳"
            st.markdown(f"{status_icon} **{key.title()}:** {value['message']}")

def main():
    """Main application function"""
    try:
        # Sidebar configuration
        with st.sidebar:
            st.title("Configuration")

            # API Key input
            api_key = os.getenv("GROQ_API_KEY") or st.text_input(
                "Enter your GROQ API Key",
                type="password"
            )

            # Model selection
            model_option = st.selectbox(
                "Select Model",
                ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"],
                help="Choose the AI model for analysis"
            )

            # Data upload section
            st.markdown("### Data Management")

            # Clear data button
            if st.button("Clear All Data", type="secondary"):
                for key in ["bond_details", "cashflow_details", "company_insights", 
                          "vector_store", "embeddings_dict", "web_cache"]:
                    if key in st.session_state:
                        st.session_state[key] = None
                st.session_state.data_loading_status = {
                    k: {"status": "not_started", "message": "Not loaded"}
                    for k in st.session_state.data_loading_status
                }
                st.success("All data cleared!")
                st.rerun()

            # File uploaders
            st.markdown("#### Upload Files")
            bond_files = [
                st.file_uploader(f"Bond Details Part {i+1}", type=["csv"], key=f"bond_{i}")
                for i in range(2)
            ]

            cashflow_file = st.file_uploader(
                "Cashflow Details",
                type=["csv"],
                key="cashflow"
            )

            company_file = st.file_uploader(
                "Company Insights",
                type=["csv"],
                key="company"
            )

            # Chunk size for data loader
            chunk_size = st.number_input(
                "Processing Chunk Size", 
                min_value=10000, 
                max_value=200000, 
                value=50000,
                step=10000,
                help="Size of chunks for processing large files"
            )

            # Load data button
            if st.button("Load Data", type="primary"):
                with st.spinner("Processing data files..."):
                    # Use the DataLoader class
                    data_loader = st.session_state.data_loader
                    data_loader.set_chunk_size(chunk_size)

                    bond_details, cashflow_details, company_insights, status = data_loader.load_data(
                        bond_files, cashflow_file, company_file, chunk_size
                    )

                    # Update session state
                    st.session_state.bond_details = bond_details
                    st.session_state.cashflow_details = cashflow_details
                    st.session_state.company_insights = company_insights
                    st.session_state.data_loading_status.update(status)

                    if any(s["status"] == "success" for s in status.values()):
                        st.success("Data loaded successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to load data. Check status below.")

        # Main content area
        st.title("Tap Bonds AI Platform")
        st.markdown("""
        Welcome to the Tap Bonds AI Platform! 💼

        This platform provides AI-powered bond analysis and insights using GROQ's advanced language models.
        Upload your data files and ask questions about bonds, companies, or market trends.
        """)

        # Display data status
        st.markdown("### Data Status")
        display_status()

        # Query interface
        if any(value["status"] == "success" for value in st.session_state.data_loading_status.values()):
            query = st.text_input(
                "Ask about bonds, companies, or market trends",
                placeholder="e.g., Show details for ISIN INE001A01001 or Find bonds with yield above 8%"
            )

            if query:
                # Initialize LLM
                llm = get_llm(api_key, model_option)
                if not llm:
                    st.error("Please provide a valid GROQ API key to continue.")
                    return

                with st.spinner("Processing your query..."):
                    # Process query
                    context = {
                        "bond_data": st.session_state.bond_details,
                        "cashflow_data": st.session_state.cashflow_details,
                        "company_data": st.session_state.company_insights
                    }

                    result = process_query(query, context, llm)

                    if "error" in result:
                        st.error(result["error"])
                    else:
                        # Display AI response
                        st.markdown("### Analysis")
                        st.markdown(result["response"])

                        # Display web search results
                        if result.get("web_results"):
                            with st.expander("Web Search Results"):
                                for i, r in enumerate(result["web_results"], 1):
                                    st.markdown(f"**{i}. [{r['title']}]({r['link']})**")
                                    st.markdown(r['snippet'])

                        # Add to chat history
                        st.session_state.chat_history.append({
                            "query": query,
                            "response": result["response"]
                        })

            # Display chat history
            if st.session_state.chat_history:
                with st.expander("Chat History"):
                    for chat in st.session_state.chat_history:
                        st.markdown(f"**Q:** {chat['query']}")
                        st.markdown(f"**A:** {chat['response']}")
                        st.markdown("---")

            # Bond data exploration
            if st.session_state.bond_details is not None:
                with st.expander("Bond Data Explorer"):
                    st.write("### Bond Details")
                    if st.session_state.bond_details is not None:
                        bond_count = len(st.session_state.bond_details)
                        st.write(f"Total Bonds: {bond_count}")

                        # Display sample bonds
                        st.dataframe(st.session_state.bond_details.head(5))

                        # ISIN search
                        isin_search = st.text_input("Search by ISIN")
                        if isin_search:
                            results = st.session_state.bond_details[
                                st.session_state.bond_details['isin'].str.contains(isin_search, case=False, na=False)
                            ]
                            if not results.empty:
                                st.dataframe(results)
                            else:
                                st.info("No matching bonds found")
        else:
            st.info("Please upload and process data files to begin analysis.")

        # Example queries
        with st.expander("Example Queries"):
            st.markdown("""
            ### Bond Analysis
            - Show details for ISIN INE001A01001
            - Find bonds with coupon rate above 8%
            - What are the highest yielding bonds in the banking sector?

            ### Cash Flow Analysis
            - Show upcoming payments for ISIN INE001A01001
            - Which bonds have payments due in the next month?

            ### Company Analysis
            - Compare financial metrics for Company XYZ
            - Show all bonds from the technology sector
            """)

    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        logger.error(f"Error in main: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
