
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
import tempfile
import traceback
import time
import faiss
import requests
from bs4 import BeautifulSoup
import html2text
from datetime import datetime

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

def save_uploadedfile(uploadedfile):
    """Save uploaded file to a temporary location"""
    try:
        if uploadedfile is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as f:
                f.write(uploadedfile.getvalue())
                return f.name
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
    return None

def validate_csv_file(file_path: str, expected_columns: List[str]) -> Tuple[bool, str]:
    """Validate CSV file format"""
    try:
        df_header = pd.read_csv(file_path, nrows=0)
        missing_columns = [col for col in expected_columns if col not in df_header.columns]
        if missing_columns:
            return False, f"Missing columns: {', '.join(missing_columns)}"
        return True, "File validated successfully"
    except Exception as e:
        return False, f"Error validating file: {str(e)}"

def process_json_columns(df: pd.DataFrame, json_columns: List[str]) -> pd.DataFrame:
    """Process JSON columns in DataFrame"""
    for col in json_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 
                json.loads(x) if isinstance(x, str) and x.strip() and x.strip() != '{' else {})
    return df

def load_data(bond_files, cashflow_file, company_file) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Dict[str, Dict]]:
    """Load all data files with error handling"""
    status = {
        "bond": {"status": "not_started", "message": ""},
        "cashflow": {"status": "not_started", "message": ""},
        "company": {"status": "not_started", "message": ""}
    }

    try:
        # Process bond files
        bond_dfs = []
        if bond_files and any(bond_files):
            status["bond"]["status"] = "in_progress"

            for i, bf in enumerate(bond_files):
                if bf is None:
                    continue

                bond_path = save_uploadedfile(bf)
                if bond_path:
                    try:
                        is_valid, validation_message = validate_csv_file(
                            bond_path, ['isin', 'company_name']
                        )
                        if not is_valid:
                            status["bond"]["status"] = "error"
                            status["bond"]["message"] = f"Bond file {i+1}: {validation_message}"
                            continue

                        df = pd.read_csv(bond_path, low_memory=False)
                        if not df.empty:
                            json_columns = ['coupon_details', 'issuer_details', 'instrument_details', 
                                          'redemption_details', 'credit_rating_details', 'listing_details']
                            df = process_json_columns(df, json_columns)
                            bond_dfs.append(df)
                    finally:
                        try:
                            os.unlink(bond_path)
                        except Exception:
                            pass

        bond_details = pd.concat(bond_dfs, ignore_index=True) if bond_dfs else None
        if bond_details is not None:
            # Clean ISIN column by removing any whitespace
            if 'isin' in bond_details.columns:
                bond_details['isin'] = bond_details['isin'].astype(str).str.strip()
            
            status["bond"]["status"] = "success"
            status["bond"]["message"] = f"Loaded {len(bond_details)} bonds"

        # Process cashflow file
        cashflow_details = None
        if cashflow_file:
            status["cashflow"]["status"] = "in_progress"
            cashflow_path = save_uploadedfile(cashflow_file)
            if cashflow_path:
                try:
                    is_valid, validation_message = validate_csv_file(
                        cashflow_path, ['isin', 'cash_flow_date', 'cash_flow_amount']
                    )
                    if is_valid:
                        cashflow_details = pd.read_csv(cashflow_path, low_memory=False)
                        if not cashflow_details.empty:
                            # Clean ISIN column
                            if 'isin' in cashflow_details.columns:
                                cashflow_details['isin'] = cashflow_details['isin'].astype(str).str.strip()
                            
                            # Convert date columns
                            for col in ['cash_flow_date', 'record_date']:
                                if col in cashflow_details.columns:
                                    cashflow_details[col] = pd.to_datetime(
                                        cashflow_details[col], errors='coerce'
                                    )
                            
                            status["cashflow"]["status"] = "success"
                            status["cashflow"]["message"] = f"Loaded {len(cashflow_details)} records"
                    else:
                        status["cashflow"]["status"] = "error"
                        status["cashflow"]["message"] = validation_message
                finally:
                    try:
                        os.unlink(cashflow_path)
                    except Exception:
                        pass

        # Process company file
        company_insights = None
        if company_file:
            status["company"]["status"] = "in_progress"
            company_path = save_uploadedfile(company_file)
            if company_path:
                try:
                    is_valid, validation_message = validate_csv_file(
                        company_path, ['company_name']
                    )
                    if is_valid:
                        company_insights = pd.read_csv(company_path, low_memory=False)
                        json_columns = ['key_metrics', 'income_statement', 'balance_sheet', 
                                      'cashflow', 'lenders_profile']
                        company_insights = process_json_columns(company_insights, json_columns)
                        if not company_insights.empty:
                            status["company"]["status"] = "success"
                            status["company"]["message"] = f"Loaded {len(company_insights)} companies"
                    else:
                        status["company"]["status"] = "error"
                        status["company"]["message"] = validation_message
                finally:
                    try:
                        os.unlink(company_path)
                    except Exception:
                        pass

        return bond_details, cashflow_details, company_insights, status

    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        for key in status:
            if status[key]["status"] in ["not_started", "in_progress"]:
                status[key]["status"] = "error"
                status[key]["message"] = "Unexpected error during processing"
        return None, None, None, status

def find_bond_by_isin(isin: str) -> Dict:
    """Find bond details by ISIN"""
    if st.session_state.bond_details is None:
        return {"error": "Bond data not loaded"}
    
    # Clean the input ISIN
    isin = isin.strip().upper()
    
    # Search for the bond in the dataframe
    bond_data = st.session_state.bond_details
    matching_bonds = bond_data[bond_data['isin'].str.upper() == isin]
    
    if matching_bonds.empty:
        return {"error": f"No bond found with ISIN {isin}"}
    
    # Get the first matching bond
    bond = matching_bonds.iloc[0].to_dict()
    
    # Get associated cashflow data if available
    cashflow_data = None
    if st.session_state.cashflow_details is not None:
        cashflows = st.session_state.cashflow_details
        matching_cashflows = cashflows[cashflows['isin'].str.upper() == isin]
        if not matching_cashflows.empty:
            cashflow_data = matching_cashflows.to_dict('records')
    
    # Get company data if available
    company_data = None
    if st.session_state.company_insights is not None and 'company_name' in bond:
        company_name = bond['company_name']
        companies = st.session_state.company_insights
        matching_companies = companies[companies['company_name'].str.contains(company_name, case=False, na=False)]
        if not matching_companies.empty:
            company_data = matching_companies.iloc[0].to_dict()
    
    return {
        "bond": bond,
        "cashflow": cashflow_data,
        "company": company_data
    }

def process_query(query: str, context: Dict, llm) -> Dict:
    """Process query using LLM and available data"""
    try:
        if not llm:
            return {"error": "AI model not initialized. Please check API key."}
        
        # Check for ISIN specific query
        isin_search = None
        if "isin" in query.lower():
            import re
            # Try to extract ISIN pattern (usually 12 alphanumeric characters)
            isin_pattern = r'\b[A-Z]{2}[A-Z0-9]{9}[0-9]{1}\b'
            matches = re.findall(isin_pattern, query.upper())
            if matches:
                isin_search = matches[0]
            else:
                # If no pattern matched, look for words after "isin"
                isin_words = re.search(r'isin\s+([A-Za-z0-9]+)', query, re.IGNORECASE)
                if isin_words:
                    isin_search = isin_words.group(1).upper()
        
        specific_bond_data = None
        if isin_search:
            specific_bond_data = find_bond_by_isin(isin_search)
            if "error" not in specific_bond_data:
                # Found a specific bond, adjust context
                context["specific_bond"] = specific_bond_data

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
        
        Specific Bond Details: {specific_bond}

        Additional Web Context:
        {web_context}

        Please provide a comprehensive analysis based on all available information.
        Format your response using Markdown for better readability.
        Include relevant web information when appropriate.
        If the query relates to a specific bond by ISIN, focus on that bond's details.
        """

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()

        response = chain.invoke({
            "query": query,
            "bond_data": str(context.get("bond_data")),
            "cashflow_data": str(context.get("cashflow_data")),
            "company_data": str(context.get("company_data")),
            "specific_bond": str(context.get("specific_bond", "No specific bond details requested")),
            "web_context": web_context
        })

        return {
            "response": response,
            "web_results": web_results
        }

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        st.error(f"Error processing query: {str(e)}")
        return {"error": f"Error processing query: {str(e)}"}

def direct_bond_lookup(isin: str) -> str:
    """Perform a direct lookup for bond details without using LLM"""
    if not isin:
        return "No ISIN provided for lookup"
    
    result = find_bond_by_isin(isin)
    if "error" in result:
        return f"**Error:** {result['error']}"
    
    # Format bond details
    bond = result["bond"]
    bond_details = "## Bond Details\n\n"
    bond_details += f"**ISIN:** {bond.get('isin', 'N/A')}\n\n"
    bond_details += f"**Company:** {bond.get('company_name', 'N/A')}\n\n"
    bond_details += f"**Issue Size:** {bond.get('issue_size', 'N/A')}\n\n"
    
    # Add more details if available
    if bond.get('coupon_details'):
        bond_details += "### Coupon Details\n\n"
        for k, v in bond['coupon_details'].items():
            bond_details += f"**{k}:** {v}\n\n"
    
    # Add cashflow details if available
    if result.get("cashflow"):
        bond_details += "### Cashflow Details\n\n"
        bond_details += "| Date | Amount | Type |\n"
        bond_details += "|------|--------|------|\n"
        for cf in result["cashflow"]:
            date = cf.get('cash_flow_date', 'N/A')
            amount = cf.get('cash_flow_amount', 'N/A')
            cf_type = cf.get('cash_flow_type', 'N/A')
            bond_details += f"| {date} | {amount} | {cf_type} |\n"
    
    # Add company details if available
    if result.get("company"):
        company = result["company"]
        bond_details += "### Company Details\n\n"
        bond_details += f"**Company:** {company.get('company_name', 'N/A')}\n\n"
        bond_details += f"**Industry:** {company.get('industry', 'N/A')}\n\n"
    
    return bond_details

def display_status():
    """Display data loading status"""
    cols = st.columns(len(st.session_state.data_loading_status))
    for col, (key, value) in zip(cols, st.session_state.data_loading_status.items()):
        with col:
            status_icon = "✅" if value["status"] == "success" else "❌" if value["status"] == "error" else "⏳"
            st.markdown(f"{status_icon} **{key.title()}:** {value['message']}")

def display_bond_explorer():
    """Display a simple bond explorer UI"""
    if st.session_state.bond_details is None:
        return
    
    with st.expander("Bond Data Explorer"):
        # Show sample of bond data
        st.dataframe(st.session_state.bond_details.head(10), use_container_width=True)
        
        # Direct ISIN lookup
        col1, col2 = st.columns([3, 1])
        with col1:
            direct_isin = st.text_input("Look up bond by ISIN", key="direct_isin")
        with col2:
            if st.button("Look Up", key="lookup_button"):
                if direct_isin:
                    st.markdown(direct_bond_lookup(direct_isin))

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

            # Processing chunk size
            chunk_size = st.number_input(
                "Processing Chunk Size",
                min_value=1000,
                max_value=100000,
                value=50000,
                step=10000,
                help="Chunk size for processing large files"
            )

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

            # Load data button
            if st.button("Load Data", type="primary"):
                with st.spinner("Processing data files..."):
                    bond_details, cashflow_details, company_insights, status = load_data(
                        bond_files, cashflow_file, company_file
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
            # Display bond explorer
            display_bond_explorer()
            
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

                    # Check for direct ISIN lookup
                    import re
                    isin_direct = re.search(r'\b[A-Z]{2}[A-Z0-9]{9}[0-9]{1}\b', query.upper())
                    if "show details for isin" in query.lower() and isin_direct:
                        isin = isin_direct.group(0)
                        st.markdown("### Analysis")
                        st.markdown(direct_bond_lookup(isin))
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            "query": query,
                            "response": direct_bond_lookup(isin)
                        })
                    else:
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
