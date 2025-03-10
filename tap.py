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

# Initialize session state
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

# Vector store for efficient similarity search
class VectorStore:
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
        response = requests.get(url, timeout=10, verify=True)  # Added verify=True to handle SSL certificates
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
                json.loads(x) if isinstance(x, str) and x.strip() else {})
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

                        df = pd.read_csv(bond_path)
                        if not df.empty:
                            json_columns = ['coupon_details', 'issuer_details', 'instrument_details']
                            df = process_json_columns(df, json_columns)
                            bond_dfs.append(df)
                    finally:
                        try:
                            os.unlink(bond_path)
                        except Exception:
                            pass

        bond_details = pd.concat(bond_dfs, ignore_index=True) if bond_dfs else None
        if bond_details is not None:
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
                        cashflow_details = pd.read_csv(cashflow_path)
                        if not cashflow_details.empty:
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
                        company_insights = pd.read_csv(company_path)
                        json_columns = ['key_metrics', 'income_statement', 'balance_sheet', 'cashflow']
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

def lookup_bond_by_isin(isin: str) -> Dict:
    """Look up bond details by ISIN"""
    try:
        # Validate input
        if not isin or len(isin) < 2:
            return {"error": "Please enter a valid ISIN"}

        # Check if bond data is loaded
        if st.session_state.bond_details is None:
            return {"error": "Bond data not loaded"}

        # Convert to uppercase for case-insensitive matching
        isin = isin.strip().upper()
        
        # Try exact match first
        exact_match = st.session_state.bond_details[st.session_state.bond_details['isin'] == isin]
        
        if not exact_match.empty:
            bond_data = exact_match.iloc[0].to_dict()
            
            # Get corresponding cashflow data if available
            cashflow_data = []
            if st.session_state.cashflow_details is not None:
                cashflow_matches = st.session_state.cashflow_details[
                    st.session_state.cashflow_details['isin'] == isin
                ]
                if not cashflow_matches.empty:
                    cashflow_data = cashflow_matches.to_dict('records')
            
            # Get company data if available
            company_data = {}
            if st.session_state.company_insights is not None and 'company_name' in bond_data:
                company_matches = st.session_state.company_insights[
                    st.session_state.company_insights['company_name'] == bond_data['company_name']
                ]
                if not company_matches.empty:
                    company_data = company_matches.iloc[0].to_dict()
            
            return {
                "bond_data": bond_data,
                "cashflow_data": cashflow_data,
                "company_data": company_data
            }
        
        # Try partial match if exact match fails
        partial_matches = st.session_state.bond_details[
            st.session_state.bond_details['isin'].str.contains(isin, case=False, na=False)
        ]
        
        if not partial_matches.empty:
            # If there are multiple matches, return the first one
            bond_data = partial_matches.iloc[0].to_dict()
            matched_isin = bond_data['isin']
            
            # Get corresponding cashflow data if available
            cashflow_data = []
            if st.session_state.cashflow_details is not None:
                cashflow_matches = st.session_state.cashflow_details[
                    st.session_state.cashflow_details['isin'] == matched_isin
                ]
                if not cashflow_matches.empty:
                    cashflow_data = cashflow_matches.to_dict('records')
            
            # Get company data if available
            company_data = {}
            if st.session_state.company_insights is not None and 'company_name' in bond_data:
                company_matches = st.session_state.company_insights[
                    st.session_state.company_insights['company_name'] == bond_data['company_name']
                ]
                if not company_matches.empty:
                    company_data = company_matches.iloc[0].to_dict()
            
            return {
                "bond_data": bond_data,
                "cashflow_data": cashflow_data,
                "company_data": company_data,
                "is_partial_match": True
            }
        
        return {"error": f"No bond found with ISIN {isin}"}
    
    except Exception as e:
        logger.error(f"Error looking up bond: {str(e)}")
        return {"error": f"Error looking up bond: {str(e)}"}

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

        # Check if query is about a specific bond (ISIN lookup)
        isin_query = query.strip().upper()
        specific_bond_data = None
        specific_cashflow_data = None

        bond_df = context.get("bond_data")
        cashflow_df = context.get("cashflow_data")

        # If query looks like an ISIN, try to extract specific bond data
        if len(isin_query) >= 10 and bond_df is not None:
            # Try exact match first
            if isin_query in bond_df['isin'].values:
                specific_bond_data = bond_df[bond_df['isin'] == isin_query].to_dict('records')
                if cashflow_df is not None:
                    specific_cashflow_data = cashflow_df[cashflow_df['isin'] == isin_query].to_dict('records')
            # Try partial match
            else:
                matching_bonds = bond_df[bond_df['isin'].str.contains(isin_query, case=False, na=False)]
                if not matching_bonds.empty:
                    specific_bond_data = matching_bonds.to_dict('records')
                    if cashflow_df is not None:
                        matching_isins = matching_bonds['isin'].tolist()
                        specific_cashflow_data = cashflow_df[cashflow_df['isin'].isin(matching_isins)].to_dict('records')

        # Perform web search for additional context
        web_results = perform_web_search(query)
        web_context = "\n".join(result.get("content", "") for result in web_results)

        # Create enhanced prompt - with token reduction strategy
        template = """You are a financial expert specializing in bond analysis.

        User Query: {query}

        Available Data Summary:
        Bond Data: {bond_data_summary}
        Cashflow Data: {cashflow_data_summary}
        Company Data: {company_data_summary}

        Specific Bond Data: {specific_bond_data}
        Specific Cashflow Data: {specific_cashflow_data}

        Additional Web Context:
        {web_context}

        Please provide a comprehensive analysis based on all available information.
        Format your response using Markdown for better readability.
        Include relevant web information when appropriate.
        """

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()

        # Create summaries instead of sending full dataframes to reduce token count
        bond_data_summary = "No bond data available"
        cashflow_data_summary = "No cashflow data available"
        company_data_summary = "No company data available"
        
        if bond_df is not None:
            bond_data_summary = f"Dataset with {len(bond_df)} bonds. Sample columns: {', '.join(bond_df.columns[:5])}"
        
        if cashflow_df is not None:
            cashflow_data_summary = f"Dataset with {len(cashflow_df)} cashflow records. Sample columns: {', '.join(cashflow_df.columns[:5])}"
        
        if context.get("company_data") is not None:
            company_data_summary = f"Dataset with {len(context.get('company_data'))} companies. Sample columns: {', '.join(context.get('company_data').columns[:5])}"

        # Truncate web context if too large
        if web_context and len(web_context) > 2000:
            web_context = web_context[:2000] + "... [truncated due to length]"

        # Truncate specific data if too large
        specific_bond_str = str(specific_bond_data)
        if len(specific_bond_str) > 1000:
            specific_bond_str = specific_bond_str[:1000] + "... [truncated due to length]"
            
        specific_cashflow_str = str(specific_cashflow_data)
        if len(specific_cashflow_str) > 1000:
            specific_cashflow_str = specific_cashflow_str[:1000] + "... [truncated due to length]"

        response = chain.invoke({
            "query": query,
            "bond_data_summary": bond_data_summary,
            "cashflow_data_summary": cashflow_data_summary,
            "company_data_summary": company_data_summary,
            "specific_bond_data": specific_bond_str if specific_bond_data else "No specific bond data found",
            "specific_cashflow_data": specific_cashflow_str if specific_cashflow_data else "No specific cashflow data found",
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

# Cashflow Display Agent class
class CashflowDisplayAgent:
    def display_cashflow_summary(self, cashflow_df, bond_data):
        """Display cashflow summary with enhanced formatting"""
        try:
            # Ensure cashflow_df has the necessary columns
            required_columns = ['cash_flow_date', 'cash_flow_amount']
            if not all(col in cashflow_df.columns for col in required_columns):
                st.error("Cashflow data missing required columns")
                return
            
            # Convert date column to datetime if it's not already
            if 'cash_flow_date' in cashflow_df.columns:
                cashflow_df['cash_flow_date'] = pd.to_datetime(cashflow_df['cash_flow_date'])
                
            # Sort by date
            cashflow_df = cashflow_df.sort_values('cash_flow_date')
            
            # Display summary statistics
            total_cashflow = cashflow_df['cash_flow_amount'].sum()
            avg_cashflow = cashflow_df['cash_flow_amount'].mean()
            
            col1, col2 = st.columns(2)
            col1.metric("Total Cashflow", f"{total_cashflow:,.2f}")
            col2.metric("Average Cashflow", f"{avg_cashflow:,.2f}")
            
            # Display cashflow timeline
            st.subheader("Cashflow Timeline")
            st.dataframe(
                cashflow_df[['cash_flow_date', 'cash_flow_amount']].rename(
                    columns={'cash_flow_date': 'Date', 'cash_flow_amount': 'Amount'}
                ),
                use_container_width=True
            )
            
            # Display cashflow visualization
            st.subheader("Cashflow Visualization")
            chart_data = cashflow_df[['cash_flow_date', 'cash_flow_amount']].rename(
                columns={'cash_flow_date': 'Date', 'cash_flow_amount': 'Amount'}
            )
            st.bar_chart(chart_data.set_index('Date'))
            
        except Exception as e:
            st.error(f"Error displaying cashflow: {str(e)}")

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

        # Add Bond Explorer
        if any(value["status"] == "success" for value in st.session_state.data_loading_status.values()):
            tabs = st.tabs(["Bond Data Explorer", "AI Query"])

            # Bond Data Explorer tab
            with tabs[0]:
                st.subheader("Look up bond by ISIN")
                isin_input = st.text_input("Enter ISIN code", key="isin_lookup", label_visibility="collapsed")

                if isin_input:
                    with st.spinner("Looking up bond details..."):
                        bond_result = lookup_bond_by_isin(isin_input)

                        if "error" in bond_result:
                            st.error(bond_result["error"])
                        else:
                            bond_data = bond_result.get("bond_data", {})
                            cashflow_data = bond_result.get("cashflow_data", [])
                            company_data = bond_result.get("company_data", {})

                            # Display bond details
                            st.markdown("### Bond Details")
                            st.markdown(f"**ISIN:** {bond_data.get('isin', 'N/A')}")
                            st.markdown(f"**Company:** {bond_data.get('company_name', 'N/A')}")
                            st.markdown(f"**Issue Size:** {bond_data.get('issue_size', 'N/A')}")

                            # Display coupon details
                            st.markdown("### Coupon Details")
                            coupon_details = bond_data.get('coupon_details', {})
                            st.json(coupon_details)

                            # Display company details if available
                            if company_data:
                                st.markdown("### Company Details")
                                st.markdown(f"**Company:** {company_data.get('company_name', 'N/A')}")
                                st.markdown(f"**Industry:** {company_data.get('industry', 'N/A')}")

                            # Display cashflow details if available
                            if cashflow_data:
                                st.markdown("### Cashflow Details")
                                cashflow_df = pd.DataFrame(cashflow_data)

                                # Create cashflow display agent
                                cashflow_display = CashflowDisplayAgent()

                                # Display cashflow with enhanced formatting
                                cashflow_display.display_cashflow_summary(cashflow_df, bond_data)
                            else:
                                st.info("No cashflow details available")

            # AI Query tab
            with tabs[1]:
                query = st.text_input(
                    "Ask about bonds, companies, or market trends",
                    placeholder="e.g., Show details for ISIN INE001A01001 or Find bonds with yield above 8%",
                    label_visibility="collapsed"
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
