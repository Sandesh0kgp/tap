import streamlit as st
import logging
import os
import pandas as pd
import numpy as np
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from typing import Dict, List, Any, Tuple, Optional
import json
import tempfile
import traceback
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Tap Bonds AI Hackathon",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sample bond data (fallback)
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
    "YTM": "15.1000%",
    "Coupon": "10.0100%",
    "NPV": 1929967.33,
    "Accrued Interest": 4936.44,
    "Clean Price": 96.2515,
    "Dirty Price": 96.4984,
    "Consideration": 1929966.44,
    "Total Consideration": 1929968.44
}

bond_df = pd.DataFrame({
    "isin": [SAMPLE_BOND_DATA["ISIN"]],
    "company_name": [SAMPLE_BOND_DATA["Issuer"]],
    "face_value": [SAMPLE_BOND_DATA["Face Value"]],
    "ytm": [SAMPLE_BOND_DATA["YTM"]],
    "coupon_rate": [SAMPLE_BOND_DATA["Coupon"]],
    "npv": [SAMPLE_BOND_DATA["NPV"]],
    "credit_rating": ["AA"],
    "maturity_date": ["22-Feb-2026"],
    "security_type": ["Secured"]
})

cashflow_df = pd.DataFrame(SAMPLE_BOND_DATA["Cashflow"])
cashflow_df["isin"] = SAMPLE_BOND_DATA["ISIN"]
cashflow_df.rename(columns={
    "Date": "cash_flow_date",
    "Record Date": "record_date",
    "Principal Amount": "principal_amount",
    "Interest Amount": "interest_amount"
}, inplace=True)

# Mock additional data for demo
mock_bond_df = pd.DataFrame({
    "isin": ["INE123456789", "INE987654321"],
    "company_name": ["UGRO CAPITAL PRIVATE LIMITED", "UGRO CAPITAL PRIVATE LIMITED"],
    "face_value": [100000, 1000000],
    "ytm": ["9.2%", "10.0%"],
    "coupon_rate": ["9.2%", "10.0%"],
    "npv": [95000, 980000],
    "credit_rating": ["AAA", "AA+"],
    "maturity_date": ["10-12-2027", "05-09-2030"],
    "security_type": ["Secured", "Secured"]
})
bond_df = pd.concat([bond_df, mock_bond_df], ignore_index=True)

mock_finder_df = pd.DataFrame({
    "isin": ["INE123456789", "INE987654321"],
    "company_name": ["TATA CAPITAL", "INDIABULLS HOUSING FINANCE"],
    "credit_rating": ["AAA", "AA"],
    "yield_range": ["7.5%-8.0%", "9.2%-9.8%"],
    "platform": ["SMEST", "FixedIncome & SMEST"]
})

mock_screener_df = pd.DataFrame({
    "company_name": ["UGRO CAPITAL PRIVATE LIMITED", "ABC COMPANY"],
    "sector": ["Financial Services", "Financial Services"],
    "industry": ["NBFC", "Banking"],
    "credit_rating": ["AA+", "A"],
    "eps": [5.2, 3.8],
    "current_ratio": [1.5, 1.2],
    "debt_equity": [0.8, 1.1],
    "debt_ebitda": [2.5, 3.0],
    "interest_coverage": [4.0, 3.5]
})

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "bond_details" not in st.session_state:
    st.session_state.bond_details = bond_df
if "cashflow_details" not in st.session_state:
    st.session_state.cashflow_details = cashflow_df
if "finder_details" not in st.session_state:
    st.session_state.finder_details = mock_finder_df
if "screener_details" not in st.session_state:
    st.session_state.screener_details = mock_screener_df
if "data_loading_status" not in st.session_state:
    st.session_state.data_loading_status = {
        "bond": {"status": "success", "message": "Sample bond loaded"},
        "cashflow": {"status": "success", "message": "Sample cashflow loaded"},
        "finder": {"status": "success", "message": "Sample finder loaded"},
        "screener": {"status": "success", "message": "Sample screener loaded"}
    }

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

def save_uploadedfile(uploadedfile):
    try:
        if uploadedfile is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as f:
                f.write(uploadedfile.getvalue())
                return f.name
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
    return None

def load_data(bond_files, cashflow_file, finder_file, screener_file):
    status = {
        "bond": {"status": "not_started", "message": ""},
        "cashflow": {"status": "not_started", "message": ""},
        "finder": {"status": "not_started", "message": ""},
        "screener": {"status": "not_started", "message": ""}
    }
    try:
        bond_dfs = []
        if bond_files and any(bond_files):
            status["bond"]["status"] = "in_progress"
            for bf in bond_files:
                if bf is None:
                    continue
                bond_path = save_uploadedfile(bf)
                if bond_path:
                    df = pd.read_csv(bond_path)
                    if 'isin' in df.columns:
                        df['isin'] = df['isin'].astype(str)
                    if 'coupon_rate' in df.columns:
                        df['coupon_rate'] = pd.to_numeric(df['coupon_rate'].astype(str).str.replace('%', ''), errors='coerce')
                    bond_dfs.append(df)
                    os.unlink(bond_path)
        bond_details = pd.concat(bond_dfs, ignore_index=True) if bond_dfs else None
        if bond_details is not None:
            status["bond"]["status"] = "success"
            status["bond"]["message"] = f"Loaded {len(bond_details)} bonds"

        cashflow_details = None
        if cashflow_file:
            cashflow_path = save_uploadedfile(cashflow_file)
            if cashflow_path:
                cashflow_details = pd.read_csv(cashflow_path)
                if 'isin' in cashflow_details.columns:
                    cashflow_details['isin'] = cashflow_details['isin'].astype(str)
                status["cashflow"]["status"] = "success"
                status["cashflow"]["message"] = f"Loaded {len(cashflow_details)} records"
                os.unlink(cashflow_path)

        finder_details = None
        if finder_file:
            finder_path = save_uploadedfile(finder_file)
            if finder_path:
                finder_details = pd.read_csv(finder_path)
                status["finder"]["status"] = "success"
                status["finder"]["message"] = f"Loaded {len(finder_details)} finder records"
                os.unlink(finder_path)

        screener_details = None
        if screener_file:
            screener_path = save_uploadedfile(screener_file)
            if screener_path:
                screener_details = pd.read_csv(screener_path)
                status["screener"]["status"] = "success"
                status["screener"]["message"] = f"Loaded {len(screener_details)} screener records"
                os.unlink(screener_path)

        return bond_details, cashflow_details, finder_details, screener_details, status
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return None, None, None, None, status

# Simulated website retriever
def retrieve_website_content(query: str) -> str:
    # Mock retrieval from TapBonds.com
    mock_content = """
    Tap Bonds is a knowledge and research platform for bonds.
    Bond Directory: https://tapbonds.com/directory - 22k ISINs available.
    Bond Finder: https://tapbonds.com/finder - Compare bonds on SMEST and FixedIncome.
    Bond Screener: Analyze company financials at https://tapbonds.com/tools/screener.
    """
    return mock_content

# Bond Yield Calculator Logic
def calculate_bond_price(cashflows: List[Dict], yield_rate: float, investment_date: str) -> float:
    total_pv = 0
    inv_date = datetime.strptime(investment_date, "%d-%m-%Y")
    for cf in cashflows:
        cf_date = datetime.strptime(cf["cash_flow_date"], "%d-%m-%Y")
        if cf_date < inv_date:
            continue
        days_diff = (cf_date - inv_date).days
        years = days_diff / 365.0
        pv = (cf["principal_amount"] + cf["interest_amount"]) / ((1 + yield_rate / 100) ** years)
        total_pv += pv
    return round(total_pv, 2)

def calculate_bond_yield(cashflows: List[Dict], price: float, investment_date: str) -> float:
    def npv(yield_rate):
        return calculate_bond_price(cashflows, yield_rate, investment_date) - price
    
    # Simple binary search for yield
    low, high = 0.0, 50.0
    for _ in range(100):
        mid = (low + high) / 2
        if npv(mid) > 0:
            low = mid
        else:
            high = mid
        if abs(high - low) < 0.001:
            break
    return round(mid, 2)

# Agent Classes
class OrchestratorAgent:
    def __init__(self, llm, bond_agent, finder_agent, cashflow_agent, screener_agent):
        self.llm = llm
        self.bond_agent = bond_agent
        self.finder_agent = finder_agent
        self.cashflow_agent = cashflow_agent
        self.screener_agent = screener_agent

    def process_query(self, query: str) -> str:
        query_lower = query.lower()
        if "isin" in query_lower or "details for" in query_lower or "issuances" in query_lower:
            return self.bond_agent.handle_query(query)
        elif "available" in query_lower or "yield" in query_lower or "platform" in query_lower:
            return self.finder_agent.handle_query(query)
        elif "cash flow" in query_lower or "maturity" in query_lower or "price" in query_lower or "yield" in query_lower:
            return self.cashflow_agent.handle_query(query)
        elif "company" in query_lower or "ratio" in query_lower or "financial" in query_lower:
            return self.screener_agent.handle_query(query)
        else:
            return "Please specify a valid query related to bonds, finder, cash flow, or screener."

class BondsDirectoryAgent:
    def __init__(self, bond_df):
        self.bond_df = bond_df

    def handle_query(self, query: str) -> str:
        query_lower = query.lower()
        
        # Basic ISIN Lookup
        if "details for isin" in query_lower:
            isin = query.split()[-1].upper()
            bond = self.bond_df[self.bond_df['isin'] == isin]
            if not bond.empty:
                b = bond.iloc[0]
                return f"""
                ### Bond Details for {isin}
                - **Issuer Name:** {b['company_name']}
                - **Type of Issuer:** Non-PSU
                - **Sector:** Financial Services
                - **Coupon Rate:** {b['coupon_rate']}
                - **Instrument Name:** {b['coupon_rate']} Secured NCDs
                - **Face Value:** ₹{b['face_value']:,}
                - **Redemption Date:** {b['maturity_date']}
                - **Credit Rating:** {b['credit_rating']}
                """
            return f"No bond found with ISIN: {isin}"

        # Issuances by Company
        elif "issuances done by" in query_lower:
            company = " ".join(query.split()[4:]).title()
            bonds = self.bond_df[self.bond_df['company_name'].str.contains(company, case=False)]
            if not bonds.empty:
                active = bonds[bonds['maturity_date'] > "09-03-2025"]
                matured = len(bonds) - len(active)
                response = f"""
                ### Issuances by {company}
                {company} has issued {len(bonds)} bonds in total.
                - {matured} have matured
                - {len(active)} are active
                #### Active Bonds:
                | ISIN | Coupon Rate | Maturity Date | Face Value | Credit Rating |
                |------|-------------|---------------|------------|--------------|
                """
                for _, b in active.iterrows():
                    response += f"| {b['isin']} | {b['coupon_rate']} | {b['maturity_date']} | ₹{b['face_value']:,} | {b['credit_rating']} |\n"
                return response
            return f"No issuances found for {company}"

        # Filter-Based Search
        elif "coupon rate above" in query_lower:
            min_coupon = float(query_lower.split("coupon rate above")[1].split("%")[0].strip())
            filtered = self.bond_df[self.bond_df['coupon_rate'].str.replace('%', '').astype(float) > min_coupon]
            if not filtered.empty:
                response = f"### Bonds with Coupon Rate Above {min_coupon}%\nFound {len(filtered)} bonds:\n"
                for _, b in filtered.head(3).iterrows():
                    response += f"- **ISIN:** {b['isin']}  \n  **Issuer:** {b['company_name']}  \n  **Coupon Rate:** {b['coupon_rate']}  \n  **Maturity:** {b['maturity_date']}\n"
                return response
            return f"No bonds found with coupon rate above {min_coupon}%"

class BondFinderAgent:
    def __init__(self, finder_df):
        self.finder_df = finder_df

    def handle_query(self, query: str) -> str:
        query_lower = query.lower()
        
        # General Inquiry
        if "bonds are available in the bond finder" in query_lower:
            response = """
            ### Available Bonds in Bond Finder
            Currently showcasing bonds on SMEST and FixedIncome:
            | Issuer | Rating | Yield Range | Available At |
            |--------|--------|-------------|--------------|
            """
            for _, f in self.finder_df.iterrows():
                response += f"| {f['company_name']} | {f['credit_rating']} | {f['yield_range']} | {f['platform']} |\n"
            return response

        # Platform Availability
        elif "where can i buy bonds from" in query_lower:
            issuer = " ".join(query.split()[5:]).title()
            bonds = self.finder_df[self.finder_df['company_name'].str.contains(issuer, case=False)]
            if not bonds.empty:
                b = bonds.iloc[0]
                return f"### Bonds from {issuer}\nAvailable on {b['platform']} with yield range {b['yield_range']}."
            return f"Bonds from {issuer} are currently not available."

        # Yield-Based Search
        elif "yield of more than" in query_lower:
            min_yield = float(query_lower.split("yield of more than")[1].split("%")[0].strip())
            filtered = self.finder_df[self.finder_df['yield_range'].apply(lambda x: float(x.split('-')[0].replace('%', '')) > min_yield)]
            if not filtered.empty:
                response = f"### Bonds with Yield > {min_yield}%\n"
                for _, f in filtered.iterrows():
                    response += f"- **Issuer:** {f['company_name']}  \n  **Rating:** {f['credit_rating']}  \n  **Yield:** {f['yield_range']}  \n  **Available At:** {f['platform']}\n"
                return response
            return f"No bonds found with yield above {min_yield}%"

class CashFlowMaturityAgent:
    def __init__(self, cashflow_df, bond_df):
        self.cashflow_df = cashflow_df
        self.bond_df = bond_df

    def handle_query(self, query: str) -> str:
        query_lower = query.lower()
        
        # Cash Flow Schedule
        if "cash flow schedule for isin" in query_lower:
            isin = query.split()[-1].upper()
            cashflows = self.cashflow_df[self.cashflow_df['isin'] == isin]
            if not cashflows.empty:
                response = f"### Cash Flow Schedule for {isin}\n| Date | Type |\n|------|------|\n"
                for _, cf in cashflows.iterrows():
                    cf_type = "Principal + Interest" if cf['principal_amount'] > 0 else "Interest Payment"
                    response += f"| {cf['cash_flow_date']} | {cf_type} |\n"
                return response
            return f"No cash flow data found for ISIN: {isin}"

        # Yield/Price Calculation
        elif "price" in query_lower or "yield" in query_lower:
            if "what will be the consideration" in query_lower:
                return "Please use the calculator tool or provide ISIN, units, date, and yield/price to calculate."
            # Placeholder for UI-driven calculator
            return "Yield/Price calculator requires UI input. Please specify ISIN, investment date, units, and yield/price."

class BondScreenerAgent:
    def __init__(self, screener_df):
        self.screener_df = screener_df

    def handle_query(self, query: str) -> str:
        query_lower = query.lower()
        
        # Company Rating
        if "rating of" in query_lower:
            company = " ".join(query.split()[4:]).title()
            comp = self.screener_df[self.screener_df['company_name'].str.contains(company, case=False)]
            if not comp.empty:
                return f"### Rating of {company}\nCredit Rating: {comp.iloc[0]['credit_rating']}"
            return f"No data found for {company}"

        # Sector Inquiry
        elif "is in which sector" in query_lower:
            company = " ".join(query.split()[1:-3]).title()
            comp = self.screener_df[self.screener_df['company_name'].str.contains(company, case=False)]
            if not comp.empty:
                return f"### Sector of {company}\nSector: {comp.iloc[0]['sector']}"
            return f"No data found for {company}"

def main():
    st.title("Tap Bonds AI Hackathon")
    st.markdown("""
    Welcome to the Tap Bonds Hackathon! Build an AI-powered layer for bond discovery and research.
    """)

    with st.sidebar:
        api_key = os.getenv("GROQ_API_KEY") or st.text_input("Enter GROQ API Key", type="password")
        model_option = st.selectbox("Select Model", ["llama3-70b-8192", "llama3-8b-8192"])
        bond_files = [st.file_uploader(f"Bond Details {i+1}", type=["csv"], key=f"bond_{i}") for i in range(2)]
        cashflow_file = st.file_uploader("Cashflow Details", type=["csv"], key="cashflow")
        finder_file = st.file_uploader("Finder Details", type=["csv"], key="finder")
        screener_file = st.file_uploader("Screener Details", type=["csv"], key="screener")
        
        if st.button("Load Data"):
            with st.spinner("Loading data..."):
                bond_details, cashflow_details, finder_details, screener_details, status = load_data(
                    bond_files, cashflow_file, finder_file, screener_file
                )
                st.session_state.bond_details = bond_details if bond_details is not None else bond_df
                st.session_state.cashflow_details = cashflow_details if cashflow_details is not None else cashflow_df
                st.session_state.finder_details = finder_details if finder_details is not None else mock_finder_df
                st.session_state.screener_details = screener_details if screener_details is not None else mock_screener_df
                st.session_state.data_loading_status.update(status)
                st.success("Data loaded!")

    # Initialize agents
    llm = get_llm(api_key, model_option)
    bond_agent = BondsDirectoryAgent(st.session_state.bond_details)
    finder_agent = BondFinderAgent(st.session_state.finder_details)
    cashflow_agent = CashFlowMaturityAgent(st.session_state.cashflow_details, st.session_state.bond_details)
    screener_agent = BondScreenerAgent(st.session_state.screener_details)
    orchestrator = OrchestratorAgent(llm, bond_agent, finder_agent, cashflow_agent, screener_agent)

    # Basic UI
    query = st.text_input("Enter your query", placeholder="e.g., Show me details for ISIN INE08XP07258")
    if query:
        with st.spinner("Processing..."):
            response = orchestrator.process_query(query)
            st.markdown("### Response")
            st.markdown(response)

    # Display status
    st.markdown("### Data Status")
    cols = st.columns(len(st.session_state.data_loading_status))
    for col, (key, value) in zip(cols, st.session_state.data_loading_status.items()):
        with col:
            status_icon = "✅" if value["status"] == "success" else "❌"
            st.markdown(f"{status_icon} **{key.title()}:** {value['message']}")

if __name__ == "__main__":
    main()
