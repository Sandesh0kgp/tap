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

# Sample bond data as provided
SAMPLE_BOND_DATA = {
    "ISIN": "INE08XP07258",
    "Issuer": "AKARA CAPITAL ADVISORS PRIVATE LIMITED",
    "Cashflow": [
        {"Date": "22-Feb-2025", "Record Date": "17-Feb-2025", "Principal Amount": 0, "Interest Amount": 17003.2},
        {"Date": "03-Mar-2025", "Record Date": "", "Principal Amount": 0, "Interest Amount": 0},
        {"Date": "22-Mar-2025", "Record Date": "07-Mar-2025", "Principal Amount": 0, "Interest Amount": 15357.8},
        {"Date": "22-Apr-2025", "Record Date": "07-Apr-2025", "Principal Amount": 0, "Interest Amount": 17003.2},
        {"Date": "22-May-2025", "Record Date": "07-May-2025", "Principal Amount": 0, "Interest Amount": 16454.8},
        {"Date": "22-Jun-2025", "Record Date": "07-Jun-2025", "Principal Amount": 0, "Interest Amount": 17003.2},
        {"Date": "22-Jul-2025", "Record Date": "07-Jul-2025", "Principal Amount": 0, "Interest Amount": 16454.8},
        {"Date": "22-Aug-2025", "Record Date": "07-Aug-2025", "Principal Amount": 0, "Interest Amount": 17003.2},
        {"Date": "22-Sep-2025", "Record Date": "07-Sep-2025", "Principal Amount": 0, "Interest Amount": 17003.2},
        {"Date": "22-Oct-2025", "Record Date": "07-Oct-2025", "Principal Amount": 0, "Interest Amount": 16454.8},
        {"Date": "22-Nov-2025", "Record Date": "07-Nov-2025", "Principal Amount": 0, "Interest Amount": 17003.2},
        {"Date": "22-Dec-2025", "Record Date": "07-Dec-2025", "Principal Amount": 0, "Interest Amount": 16454.8},
        {"Date": "22-Jan-2026", "Record Date": "07-Jan-2026", "Principal Amount": 0, "Interest Amount": 17003.2},
        {"Date": "22-Feb-2026", "Record Date": "07-Feb-2026", "Principal Amount": 2000000, "Interest Amount": 17003.2}
    ],
    "Face Value": 100000.00,
    "Face Value After PO": 100000.00,
    "Trade FV": 2000000,
    "Units": 20,
    "Trade Date": "03-Mar-2025",
    "YTM": "15.1000%",
    "Coupon": "10.0100%",
    "NPV": 1929967.33,
    "Accrued Interest": 4936.44,
    "Clean Price": 96.2515,
    "Dirty Price": 96.4984,
    "Consideration": 1929966.44,
    "Stamp Duty": 2,
    "Total Consideration": 1929968.44
}

# Convert sample data to DataFrame format
bond_df = pd.DataFrame({
    "isin": [SAMPLE_BOND_DATA["ISIN"]],
    "company_name": [SAMPLE_BOND_DATA["Issuer"]],
    "face_value": [SAMPLE_BOND_DATA["Face Value"]],
    "ytm": [SAMPLE_BOND_DATA["YTM"]],
    "coupon_rate": [SAMPLE_BOND_DATA["Coupon"]],
    "npv": [SAMPLE_BOND_DATA["NPV"]],
    "trade_date": [SAMPLE_BOND_DATA["Trade Date"]]
})

cashflow_df = pd.DataFrame(SAMPLE_BOND_DATA["Cashflow"])
cashflow_df["isin"] = SAMPLE_BOND_DATA["ISIN"]
cashflow_df.rename(columns={
    "Date": "cash_flow_date",
    "Record Date": "record_date",
    "Principal Amount": "principal_amount",
    "Interest Amount": "interest_amount"
}, inplace=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "bond_details" not in st.session_state:
    st.session_state.bond_details = bond_df
if "cashflow_details" not in st.session_state:
    st.session_state.cashflow_details = cashflow_df
if "company_insights" not in st.session_state:
    st.session_state.company_insights = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "embeddings_dict" not in st.session_state:
    st.session_state.embeddings_dict = {}
if "data_loading_status" not in st.session_state:
    st.session_state.data_loading_status = {
        "bond": {"status": "success", "message": "Sample bond loaded"},
        "cashflow": {"status": "success", "message": "Sample cashflow loaded"},
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
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        h = html2text.HTML2Text()
        h.ignore_links = True
        content = h.handle(soup.get_text())
        return content[:5000]
    except Exception as e:
        logger.error(f"Error fetching web content: {str(e)}")
        return ""

@st.cache_data(ttl=3600)
def perform_web_search(query: str, num_results: int = 2) -> List[Dict]:
    try:
        search = DuckDuckGoSearchAPIWrapper()
        results = search.results(query, num_results)
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
                "content": st.session_state.web_cache[result["link"]]["content"][:2000]
            })
        return enhanced_results
    except Exception as e:
        logger.error(f"Error in web search: {str(e)}")
        return []

def save_uploadedfile(uploadedfile):
    try:
        if uploadedfile is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as f:
                f.write(uploadedfile.getvalue())
                return f.name
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
    return None

def validate_csv_file(file_path: str, expected_columns: List[str]) -> Tuple[bool, str]:
    try:
        df_header = pd.read_csv(file_path, nrows=0)
        missing_columns = [col for col in expected_columns if col not in df_header.columns]
        if missing_columns:
            return False, f"Missing columns: {', '.join(missing_columns)}"
        return True, "File validated successfully"
    except Exception as e:
        return False, f"Error validating file: {str(e)}"

def process_json_columns(df: pd.DataFrame, json_columns: List[str]) -> pd.DataFrame:
    for col in json_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 
                json.loads(x) if isinstance(x, str) and x.strip() else {})
    return df

def lookup_bond_by_isin(isin: str) -> Dict:
    try:
        if st.session_state.bond_details is None:
            return {"error": "Bond data not loaded"}
            
        bond_df = st.session_state.bond_details
        isin = isin.strip().upper()
        
        matching_bonds = bond_df[bond_df['isin'] == isin]
        
        if matching_bonds.empty:
            matching_bonds = bond_df[bond_df['isin'].str.contains(isin, case=False, na=False)]
            
            if matching_bonds.empty:
                min_match_length = min(5, len(isin))
                for i in range(min_match_length, len(isin) + 1):
                    partial_isin = isin[:i]
                    matching_bonds = bond_df[bond_df['isin'].str.contains(partial_isin, case=False, na=False)]
                    if not matching_bonds.empty:
                        break
        
        if not matching_bonds.empty:
            bond_data = matching_bonds.iloc[0].to_dict()
            
            cashflow_data = None
            if st.session_state.cashflow_details is not None:
                cashflow = st.session_state.cashflow_details[
                    st.session_state.cashflow_details['isin'] == matching_bonds.iloc[0]['isin']
                ]
                if not cashflow.empty:
                    cashflow_data = cashflow.to_dict('records')
            
            company_data = None
            if st.session_state.company_insights is not None:
                company = st.session_state.company_insights[
                    st.session_state.company_insights['company_name'] == matching_bonds.iloc[0]['company_name']
                ]
                if not company.empty:
                    company_data = company.iloc[0].to_dict()
            
            return {
                "bond_data": bond_data,
                "cashflow_data": cashflow_data,
                "company_data": company_data,
                "matches": len(matching_bonds)
            }
        
        return {"error": f"No bond found with ISIN: {isin}"}
    except Exception as e:
        logger.error(f"Error looking up bond: {str(e)}\n{traceback.format_exc()}")
        return {"error": f"Error looking up bond: {str(e)}"}

def load_data(bond_files, cashflow_file, company_file) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Dict[str, Dict]]:
    status = {
        "bond": {"status": "not_started", "message": ""},
        "cashflow": {"status": "not_started", "message": ""},
        "company": {"status": "not_started", "message": ""}
    }

    try:
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
                            if 'isin' in df.columns:
                                df['isin'] = df['isin'].astype(str)
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
                            if 'isin' in cashflow_details.columns:
                                cashflow_details['isin'] = cashflow_details['isin'].astype(str)
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

@st.cache_resource
def get_llm(api_key: str, model: str = "llama3-70b-8192"):
    try:
        if not api_key:
            return None
        return ChatGroq(
            api_key=api_key,
            model=model,
            temperature=0.2,
            max_tokens=3000
        )
    except Exception as e:
        logger.error(f"Error initializing LLM: {str(e)}")
        return None

def extract_bond_summary(data: Dict) -> str:
    if not data or "bond_data" not in data:
        return "No specific bond data found"
    
    bond = data["bond_data"]
    summary = {
        "isin": bond.get("isin", "N/A"),
        "company_name": bond.get("company_name", "N/A"),
        "face_value": bond.get("face_value", "N/A"),
        "ytm": bond.get("ytm", "N/A"),
        "coupon_rate": bond.get("coupon_rate", "N/A"),
        "npv": bond.get("npv", "N/A"),
        "trade_date": bond.get("trade_date", "N/A")
    }
    return json.dumps(summary, indent=2)

def extract_cashflow_summary(data: List) -> str:
    if not data:
        return "No specific cashflow data found"
    
    limited_data = data[:5]  # Show up to 5 cashflows
    summarized = []
    
    for cf in limited_data:
        summary = {
            "isin": cf.get("isin", "N/A"),
            "cash_flow_date": cf.get("cash_flow_date", "N/A"),
            "principal_amount": cf.get("principal_amount", "N/A"),
            "interest_amount": cf.get("interest_amount", "N/A"),
            "record_date": cf.get("record_date", "N/A")
        }
        summarized.append(summary)
    
    return json.dumps(summarized, indent=2)

def process_query(query: str, context: Dict, llm) -> Dict:
    try:
        if not llm:
            return {"error": "AI model not initialized. Please check API key."}

        isin_query = query.strip().upper()
        specific_bond_data = None
        
        if (len(isin_query) >= 5 and "bond_data" in context) or "ISIN" in isin_query:
            if "ISIN" in isin_query:
                import re
                isin_pattern = re.search(r'[A-Z0-9]{5,}', isin_query)
                if isin_pattern:
                    isin_query = isin_pattern.group(0)
            
            specific_bond_data = lookup_bond_by_isin(isin_query)

        web_results = perform_web_search(query, num_results=1)
        
        web_snippets = []
        for result in web_results:
            web_snippets.append({
                "title": result.get("title", ""),
                "snippet": result.get("snippet", "")
            })
        
        template = """You are a financial expert specializing in bond analysis.

        User Query: {query}

        {specific_bond_info}

        {web_info}

        Please provide a concise analysis based on the available information.
        Format your response using Markdown for better readability.
        If the user is looking for a specific bond by ISIN, focus on providing key details.
        """

        specific_bond_info = ""
        if specific_bond_data and "error" not in specific_bond_data:
            specific_bond_info = f"""
            Specific Bond Information:
            {extract_bond_summary(specific_bond_data)}
            
            Cashflow Information:
            {extract_cashflow_summary(specific_bond_data.get('cashflow_data', []))}
            """
        
        web_info = ""
        if web_snippets:
            web_info = "Relevant Web Information:\n"
            for i, result in enumerate(web_snippets):
                web_info += f"Source {i+1}: {result['title']}\n{result['snippet']}\n\n"

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()

        response = chain.invoke({
            "query": query,
            "specific_bond_info": specific_bond_info,
            "web_info": web_info
        })

        return {
            "response": response,
            "web_results": web_results
        }

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return {"error": f"Error processing query: {str(e)}"}

def display_status():
    cols = st.columns(len(st.session_state.data_loading_status))
    for col, (key, value) in zip(cols, st.session_state.data_loading_status.items()):
        with col:
            status_icon = "✅" if value["status"] == "success" else "❌"
            st.markdown(f"{status_icon} **{key.title()}:** {value['message']}")

def main():
    try:
        with st.sidebar:
            st.title("Configuration")
            api_key = os.getenv("GROQ_API_KEY") or st.text_input("Enter your GROQ API Key", type="password")
            model_option = st.selectbox(
                "Select Model",
                ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"],
                help="Choose the AI model for analysis"
            )
            st.markdown("### Data Management")
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

            st.markdown("#### Upload Files")
            bond_files = [
                st.file_uploader(f"Bond Details Part {i+1}", type=["csv"], key=f"bond_{i}")
                for i in range(2)
            ]
            cashflow_file = st.file_uploader("Cashflow Details", type=["csv"], key="cashflow")
            company_file = st.file_uploader("Company Insights", type=["csv"], key="company")

            if st.button("Load Data", type="primary"):
                with st.spinner("Processing data files..."):
                    bond_details, cashflow_details, company_insights, status = load_data(
                        bond_files, cashflow_file, company_file
                    )
                    st.session_state.bond_details = bond_details if bond_details is not None else bond_df
                    st.session_state.cashflow_details = cashflow_details if cashflow_details is not None else cashflow_df
                    st.session_state.company_insights = company_insights
                    st.session_state.data_loading_status.update(status)
                    if any(s["status"] == "success" for s in status.values()):
                        st.success("Data loaded successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to load data. Check status below.")

        st.title("Tap Bonds AI Platform")
        st.markdown("""
        Welcome to the Tap Bonds AI Platform! 💼
        This platform provides AI-powered bond analysis and insights using GROQ's advanced language models.
        Upload your data files and ask questions about bonds, companies, or market trends.
        Sample bond data (INE08XP07258) is pre-loaded for demonstration.
        """)

        st.markdown("### Data Status")
        display_status()

        if st.session_state.bond_details is not None:
            with st.expander("Debug Information (Data Overview)"):
                st.write("Bond Data Sample:")
                st.dataframe(st.session_state.bond_details)
                st.write("Cashflow Data Sample (first 5 rows):")
                st.dataframe(st.session_state.cashflow_details.head(5))

        if any(value["status"] == "success" for value in st.session_state.data_loading_status.values()):
            tabs = st.tabs(["Bond Data Explorer", "AI Query"])
            
            with tabs[0]:
                st.subheader("Look up bond by ISIN")
                isin_input = st.text_input("Enter ISIN", key="isin_lookup", 
                                         placeholder="e.g., INE08XP07258",
                                         help="Enter full or partial ISIN code")
                
                if isin_input:
                    with st.spinner("Looking up bond details..."):
                        bond_result = lookup_bond_by_isin(isin_input)
                        
                        if "error" in bond_result:
                            st.error(bond_result["error"])
                        else:
                            if bond_result.get("matches", 1) > 1:
                                st.info(f"Found {bond_result.get('matches')} bonds matching '{isin_input}'. Showing the first match.")
                            
                            bond_data = bond_result.get("bond_data", {})
                            cashflow_data = bond_result.get("cashflow_data", [])
                            
                            st.markdown("### Bond Details")
                            st.markdown(f"**ISIN:** {bond_data.get('isin', 'N/A')}")
                            st.markdown(f"**Company:** {bond_data.get('company_name', 'N/A')}")
                            st.markdown(f"**Face Value:** {bond_data.get('face_value', 'N/A')}")
                            st.markdown(f"**YTM:** {bond_data.get('ytm', 'N/A')}")
                            st.markdown(f"**Coupon Rate:** {bond_data.get('coupon_rate', 'N/A')}")
                            st.json(bond_data)
                            
                            if cashflow_data:
                                st.markdown("### Cashflow Details")
                                st.dataframe(pd.DataFrame(cashflow_data))
            
            with tabs[1]:
                query = st.text_input(
                    "Ask about bonds, companies, or market trends",
                    placeholder="e.g., Show details for ISIN INE08XP07258",
                    label_visibility="collapsed"
                )
                
                if query:
                    llm = get_llm(api_key, model_option)
                    if not llm:
                        st.error("Please provide a valid GROQ API key to continue.")
                        return

                    with st.spinner("Processing your query..."):
                        context = {
                            "bond_data": st.session_state.bond_details,
                            "cashflow_data": st.session_state.cashflow_details,
                            "company_data": st.session_state.company_insights
                        }
                        result = process_query(query, context, llm)

                        if "error" in result:
                            st.error(result["error"])
                        else:
                            st.markdown("### Analysis")
                            st.markdown(result["response"])
                            if result.get("web_results"):
                                with st.expander("Web Search Results"):
                                    for i, r in enumerate(result["web_results"], 1):
                                        st.markdown(f"**{i}. [{r['title']}]({r['link']})**")
                                        st.markdown(r['snippet'])
                            st.session_state.chat_history.append({
                                "query": query,
                                "response": result["response"]
                            })

            if st.session_state.chat_history:
                with st.expander("Chat History"):
                    for chat in st.session_state.chat_history:
                        st.markdown(f"**Q:** {chat['query']}")
                        st.markdown(f"**A:** {chat['response']}")
                        st.markdown("---")
        else:
            st.info("Please upload and process data files to begin analysis.")

        with st.expander("Example Queries"):
            st.markdown("""
            ### Bond Analysis
            - Show details for ISIN INE08XP07258
            - Find bonds with yield above 15%
            - What are the upcoming payments for INE08XP07258?

            ### Cash Flow Analysis
            - Show cashflows for ISIN INE08XP07258 in 2025
            - When is the principal repayment for INE08XP07258?
            """)

    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        logger.error(f"Error in main: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
