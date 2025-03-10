import pandas as pd
import logging
from typing import Optional, Dict, List, Tuple, Any
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime
import numpy as np
import faiss

class TapBondsDataLoader:
    """
    Optimized data loader for TapBonds platform handling bond data, cashflow details,
    and company financial information with efficient search capabilities.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.bond_details = None
        self.cashflow_details = None
        self.company_insights = None
        self._chunk_size = 50000
        self.vector_store = None
        self.web_cache = {}
        self.search_results = {}
        
    def initialize_vector_store(self, dimension=768):
        """Initialize vector store for similarity search"""
        self.vector_store = {
            "dimension": dimension,
            "index": faiss.IndexFlatL2(dimension),
            "texts": []
        }
        
    def add_vectors(self, texts: List[str], embeddings: List[List[float]]):
        """Add text embeddings to vector store"""
        if not self.vector_store:
            self.initialize_vector_store()
            
        if not texts or not embeddings:
            return
            
        embeddings_array = np.array(embeddings).astype('float32')
        self.vector_store["index"].add(embeddings_array)
        self.vector_store["texts"].extend(texts)
        
    def similarity_search(self, query_embedding: List[float], k: int = 5) -> List[str]:
        """Perform similarity search using vector store"""
        if not self.vector_store:
            return []
            
        query_array = np.array([query_embedding]).astype('float32')
        D, I = self.vector_store["index"].search(query_array, k)
        return [self.vector_store["texts"][i] for i in I[0] if i < len(self.vector_store["texts"])]

    def set_chunk_size(self, size: int):
        """Set the chunk size for processing large files"""
        self._chunk_size = max(1000, min(size, 100000))

    def load_data(self, bond_files: List, cashflow_file, company_file, chunk_size: Optional[int] = None) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Dict[str, Dict]]:
        """Load and process all data files"""
        if chunk_size:
            self.set_chunk_size(chunk_size)

        status = {
            "bond": {"status": "not_started", "message": ""}, 
            "cashflow": {"status": "not_started", "message": ""}, 
            "company": {"status": "not_started", "message": ""}
        }

        bond_details, cashflow_details, company_insights = None, None, None

        try:
            # Process bond files
            if bond_files and any(bond_files):
                bond_details, status["bond"] = self._process_bond_files(bond_files)
                if bond_details is not None:
                    self.bond_details = bond_details

            # Process cashflow file
            if cashflow_file:
                cashflow_details, status["cashflow"] = self._process_cashflow_file(cashflow_file)
                if cashflow_details is not None:
                    self.cashflow_details = cashflow_details

            # Process company file
            if company_file:
                company_insights, status["company"] = self._process_company_file(company_file)
                if company_insights is not None:
                    self.company_insights = company_insights

        except Exception as e:
            self.logger.error(f"Error in load_data: {str(e)}")
            # Update any unset status
            for key in status:
                if status[key]["status"] in ["not_started", "in_progress"]:
                    status[key]["status"] = "error"
                    status[key]["message"] = "Unexpected error during processing"

        return bond_details, cashflow_details, company_insights, status

    def _process_bond_files(self, bond_files: List) -> Tuple[Optional[pd.DataFrame], Dict[str, str]]:
        """Process multiple bond files with improved error handling"""
        status = {"status": "in_progress", "message": ""}
        try:
            bond_dfs = []
            for i, bf in enumerate(bond_files):
                if bf is None:
                    continue

                temp_path = self._save_temp_file(bf)
                if not temp_path:
                    continue

                try:
                    # Load and validate file
                    df = self._load_and_validate_csv(
                        temp_path, 
                        ['isin', 'company_name'],
                        f"Bond file {i+1}"
                    )
                    if df is not None:
                        bond_dfs.append(df)
                finally:
                    self._cleanup_temp_file(temp_path)

            if bond_dfs:
                bond_details = pd.concat(bond_dfs, ignore_index=True)
                bond_details = bond_details.drop_duplicates(subset=['isin'], keep='first')
                self._process_json_columns(bond_details)
                status = {
                    "status": "success",
                    "message": f"Loaded {len(bond_details)} bonds"
                }
                return bond_details, status
            else:
                status = {
                    "status": "error",
                    "message": "No valid bond files processed"
                }
                return None, status

        except Exception as e:
            self.logger.error(f"Error processing bond files: {str(e)}")
            status = {
                "status": "error",
                "message": f"Error processing bond files: {str(e)}"
            }
            return None, status

    def _process_cashflow_file(self, cashflow_file) -> Tuple[Optional[pd.DataFrame], Dict[str, str]]:
        """Process cashflow file with improved date and numeric handling"""
        status = {"status": "in_progress", "message": ""}
        temp_path = self._save_temp_file(cashflow_file)

        if not temp_path:
            status = {
                "status": "error",
                "message": "Failed to save cashflow file"
            }
            return None, status

        try:
            df = self._load_and_validate_csv(
                temp_path,
                ['isin', 'cash_flow_date', 'cash_flow_amount'],
                "Cashflow file"
            )
            if df is not None:
                # Convert date columns
                for col in ['cash_flow_date', 'record_date']:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors='coerce')

                # Convert numeric columns
                for col in ['cash_flow_amount', 'principal_amount', 'interest_amount']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Sort by date for easier analysis
                if 'cash_flow_date' in df.columns:
                    df = df.sort_values('cash_flow_date')

                status = {
                    "status": "success",
                    "message": f"Loaded {len(df)} cashflow records"
                }
                return df, status
            return None, {
                "status": "error",
                "message": "Failed to validate cashflow file"
            }

        except Exception as e:
            self.logger.error(f"Error processing cashflow file: {str(e)}")
            status = {
                "status": "error",
                "message": f"Error processing cashflow file: {str(e)}"
            }
            return None, status
        finally:
            self._cleanup_temp_file(temp_path)

    def _process_company_file(self, company_file) -> Tuple[Optional[pd.DataFrame], Dict[str, str]]:
        """Process company insights file with improved JSON parsing"""
        status = {"status": "in_progress", "message": ""}
        temp_path = self._save_temp_file(company_file)

        if not temp_path:
            status = {
                "status": "error",
                "message": "Failed to save company file"
            }
            return None, status

        try:
            df = self._load_and_validate_csv(
                temp_path,
                ['company_name'],
                "Company file"
            )
            if df is not None:
                # Process JSON columns
                json_columns = [
                    'key_metrics', 'income_statement', 'balance_sheet',
                    'cashflow', 'lenders_profile'
                ]
                for col in json_columns:
                    if col in df.columns:
                        df[col] = df[col].apply(self._safe_json_parse)

                status = {
                    "status": "success",
                    "message": f"Loaded {len(df)} company records"
                }
                return df, status
            return None, {
                "status": "error",
                "message": "Failed to validate company file"
            }

        except Exception as e:
            self.logger.error(f"Error processing company file: {str(e)}")
            status = {
                "status": "error",
                "message": f"Error processing company file: {str(e)}"
            }
            return None, status
        finally:
            self._cleanup_temp_file(temp_path)
    
    def _safe_json_parse(self, value):
        """Safely parse JSON strings with error handling"""
        if not isinstance(value, str) or not value.strip():
            return {}
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            self.logger.warning(f"Failed to parse JSON: {value[:50]}...")
            return {}

    def _save_temp_file(self, file) -> Optional[str]:
        """Save uploaded file to temporary location"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as f:
                f.write(file.getvalue())
                return f.name
        except Exception as e:
            self.logger.error(f"Error saving temporary file: {str(e)}")
            return None

    def _cleanup_temp_file(self, file_path: str) -> None:
        """Clean up temporary file"""
        try:
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            self.logger.error(f"Error cleaning up temporary file: {str(e)}")

    def _load_and_validate_csv(self, file_path: str, required_columns: List[str], file_desc: str) -> Optional[pd.DataFrame]:
        """Load and validate CSV file with chunk processing for large files"""
        try:
            # Load CSV in chunks
            chunks = []
            try:
                for chunk in pd.read_csv(file_path, chunksize=self._chunk_size):
                    if not chunk.empty:
                        chunks.append(chunk)
            except pd.errors.EmptyDataError:
                self.logger.error(f"{file_desc} is empty")
                return None
            except pd.errors.ParserError as e:
                self.logger.error(f"Parser error in {file_desc}: {str(e)}")
                return None

            if not chunks:
                self.logger.error(f"{file_desc} contains no valid data")
                return None

            df = pd.concat(chunks, ignore_index=True)

            # Validate required columns
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                self.logger.error(f"Missing columns in {file_desc}: {', '.join(missing_cols)}")
                return None

            return df
        except Exception as e:
            self.logger.error(f"Error loading {file_desc}: {str(e)}")
            return None

    def _process_json_columns(self, df: pd.DataFrame) -> None:
        """Process JSON columns in DataFrame"""
        json_columns = [
            'coupon_details', 'issuer_details', 'instrument_details',
            'redemption_details', 'credit_rating_details', 'listing_details'
        ]

        for col in json_columns:
            if col in df.columns:
                df[col] = df[col].apply(self._safe_json_parse)

    # ===== Bond Directory Functions =====
    
    def get_bond_details(self, isin: Optional[str] = None) -> pd.DataFrame:
        """Retrieve bond details, optionally filtered by ISIN"""
        if self.bond_details is None:
            raise ValueError("Bond data not loaded")
        if isin:
            return self.bond_details[self.bond_details['isin'] == isin]
        return self.bond_details
    
    def find_bonds_by_company(self, company_name: str) -> pd.DataFrame:
        """Find bonds issued by a specific company"""
        if self.bond_details is None:
            raise ValueError("Bond data not loaded")
        return self.bond_details[
            self.bond_details['company_name'].str.contains(company_name, case=False, na=False)
        ]
    
    def find_bonds_by_criteria(self, criteria: Dict[str, Any]) -> pd.DataFrame:
        """Find bonds based on multiple criteria like coupon rate, maturity, etc."""
        if self.bond_details is None:
            raise ValueError("Bond data not loaded")
        
        filtered_df = self.bond_details.copy()
        
        for key, value in criteria.items():
            if key not in filtered_df.columns:
                continue
                
            if key == 'maturity_date' and isinstance(value, dict):
                if 'start' in value and 'end' in value:
                    filtered_df = filtered_df[
                        (pd.to_datetime(filtered_df['maturity_date']) >= pd.to_datetime(value['start'])) & 
                        (pd.to_datetime(filtered_df['maturity_date']) <= pd.to_datetime(value['end']))
                    ]
            elif key == 'coupon' and isinstance(value, dict):
                if 'min' in value:
                    filtered_df = filtered_df[filtered_df['coupon'].astype(float) >= float(value['min'])]
                if 'max' in value:
                    filtered_df = filtered_df[filtered_df['coupon'].astype(float) <= float(value['max'])]
            elif key == 'credit_rating' and isinstance(value, list):
                filtered_df = filtered_df[filtered_df['credit_rating'].isin(value)]
            else:
                filtered_df = filtered_df[filtered_df[key] == value]
                
        return filtered_df
    
    def get_bonds_by_maturity_year(self, year: int) -> pd.DataFrame:
        """Get bonds maturing in a specific year"""
        if self.bond_details is None:
            raise ValueError("Bond data not loaded")
        
        if 'maturity_date' not in self.bond_details.columns:
            raise ValueError("Maturity date column not available")
            
        self.bond_details['maturity_year'] = pd.to_datetime(self.bond_details['maturity_date']).dt.year
        return self.bond_details[self.bond_details['maturity_year'] == year]

    # ===== Cashflow Functions =====
    
    def get_cashflow_details(self, isin: Optional[str] = None) -> pd.DataFrame:
        """Retrieve cashflow details, optionally filtered by ISIN"""
        if self.cashflow_details is None:
            raise ValueError("Cashflow data not loaded")
        if isin:
            return self.cashflow_details[self.cashflow_details['isin'] == isin]
        return self.cashflow_details
    
    def get_upcoming_cashflows(self, days: int = 30) -> pd.DataFrame:
        """Get upcoming cashflows within specified days"""
        if self.cashflow_details is None:
            raise ValueError("Cashflow data not loaded")
            
        today = pd.Timestamp.now()
        future_date = today + pd.Timedelta(days=days)
        
        return self.cashflow_details[
            (self.cashflow_details['cash_flow_date'] >= today) & 
            (self.cashflow_details['cash_flow_date'] <= future_date)
        ]

    # ===== Company Functions =====
    
    def get_company_insights(self, company_name: Optional[str] = None) -> pd.DataFrame:
        """Retrieve company insights, optionally filtered by company name"""
        if self.company_insights is None:
            raise ValueError("Company data not loaded")
        if company_name:
            return self.company_insights[
                self.company_insights['company_name'].str.contains(
                    company_name, case=False, na=False
                )
            ]
        return self.company_insights
    
    def compare_companies(self, company_names: List[str], metrics: List[str]) -> Dict[str, Dict]:
        """Compare multiple companies based on specified metrics"""
        if self.company_insights is None:
            raise ValueError("Company data not loaded")
            
        result = {}
        for company in company_names:
            company_data = self.get_company_insights(company)
            if company_data.empty:
                result[company] = {"error": "Company not found"}
                continue
                
            company_result = {}
            for metric in metrics:
                if metric in company_data.columns:
                    company_result[metric] = company_data.iloc[0][metric]
                elif metric in ['key_metrics', 'income_statement', 'balance_sheet', 'cashflow']:
                    if metric in company_data.columns:
                        company_result[metric] = company_data.iloc[0][metric]
                        
            result[company] = company_result
            
        return result

    # ===== Bond Finder & Screener Functions =====
    
    def lookup_bond_by_isin(self, isin: str) -> Dict:
        """Look up bond details by ISIN with structured output"""
        try:
            if self.bond_details is None:
                return {"error": "Bond data not loaded", "success": False}
                
            # Try exact match
            matching_bonds = self.bond_details[self.bond_details['isin'] == isin]
            
            if matching_bonds.empty:
                # Try partial match
                matching_bonds = self.bond_details[self.bond_details['isin'].str.contains(isin, case=False, na=False)]
            
            if matching_bonds.empty:
                # Suggest similar ISINs
                similar_isins = []
                if len(isin) >= 4:
                    similar_isins = self.bond_details[
                        self.bond_details['isin'].str.contains(isin[:4], case=False, na=False)
                    ]['isin'].unique().tolist()[:5]
                
                return {
                    "error": f"No bond found with ISIN: {isin}",
                    "similar_isins": similar_isins if similar_isins else None,
                    "success": False
                }
                
            # Get primary bond data
            bond_data = matching_bonds.iloc[0].to_dict()
            
            # Get cashflow data if available
            cashflow_data = None
            cashflow_metrics = None
            if self.cashflow_details is not None:
                cashflow = self.cashflow_details[
                    self.cashflow_details['isin'] == matching_bonds.iloc[0]['isin']
                ]
                
                if not cashflow.empty:
                    # Convert date columns
                    for col in ['cash_flow_date', 'record_date']:
                        if col in cashflow.columns:
                            cashflow[col] = pd.to_datetime(cashflow[col]).dt.strftime('%d-%b-%Y')
                    
                    # Format numeric columns
                    for col in ['cash_flow_amount', 'principal_amount', 'interest_amount']:
                        if col in cashflow.columns:
                            cashflow[col] = cashflow[col].apply(lambda x: float(x) if pd.notnull(x) else 0)
                    
                    # Sort by date if possible
                    if 'cash_flow_date' in cashflow.columns:
                        cashflow['date_for_sort'] = pd.to_datetime(cashflow['cash_flow_date'], format='%d-%b-%Y')
                        cashflow = cashflow.sort_values('date_for_sort')
                        cashflow = cashflow.drop('date_for_sort', axis=1)
                    
                    # Format as structured records
                    cashflow_data = cashflow.to_dict('records')
                    
                    # Calculate summary metrics
                    cashflow_metrics = {
                        "total_cashflow": sum(cf.get('cash_flow_amount', 0) for cf in cashflow_data),
                        "total_principal": sum(cf.get('principal_amount', 0) for cf in cashflow_data),
                        "total_interest": sum(cf.get('interest_amount', 0) for cf in cashflow_data),
                        "payment_count": len(cashflow_data),
                        "next_payment_date": cashflow_data[0]['cash_flow_date'] if cashflow_data else None,
                        "next_payment_amount": cashflow_data[0]['cash_flow_amount'] if cashflow_data else None
                    }
            
            # Get company data if available
            company_data = None
            if self.company_insights is not None and 'company_name' in bond_data:
                company = self.company_insights[
                    self.company_insights['company_name'] == bond_data['company_name']
                ]
                if not company.empty:
                    company_data = company.iloc[0].to_dict()
            
            # Extract key details from bond data for summary
            bond_summary = {}
            for key in ['isin', 'company_name', 'face_value', 'issue_size', 'trade_fv', 
                        'maturity_date', 'allotment_date', 'ytm', 'coupon']:
                if key in bond_data:
                    bond_summary[key] = bond_data[key]
            
            return {
                "bond_data": bond_data,
                "bond_summary": bond_summary,
                "cashflow_data": cashflow_data,
                "cashflow_metrics": cashflow_metrics,
                "company_data": company_data,
                "success": True
            }
        except Exception as e:
            self.logger.error(f"Error looking up bond: {str(e)}")
            return {"error": f"Error looking up bond: {str(e)}", "success": False}
    
    def find_high_yield_bonds(self, min_yield: float = 8.0, sector: Optional[str] = None) -> pd.DataFrame:
        """Find bonds with yield above specified threshold, optionally in a specific sector"""
        if self.bond_details is None:
            raise ValueError("Bond data not loaded")
            
        if 'ytm' not in self.bond_details.columns:
            raise ValueError("Yield to maturity column not available")
        
        # Convert YTM to numeric
        self.bond_details['ytm_numeric'] = pd.to_numeric(self.bond_details['ytm'], errors='coerce')
        
        filtered_df = self.bond_details[self.bond_details['ytm_numeric'] >= min_yield]
        
        if sector and 'sector' in self.bond_details.columns:
            filtered_df = filtered_df[filtered_df['sector'] == sector]
            
        return filtered_df.sort_values('ytm_numeric', ascending=False)
    
    def find_bonds_by_credit_rating(self, ratings: List[str]) -> pd.DataFrame:
        """Find bonds with specific credit ratings"""
        if self.bond_details is None:
            raise ValueError("Bond data not loaded")
            
        if 'credit_rating' not in self.bond_details.columns:
            raise ValueError("Credit rating column not available")
            
        return self.bond_details[self.bond_details['credit_rating'].isin(ratings)]
    
    # ===== Bond Calculator Functions =====
    
    def calculate_bond_price(self, face_value: float, coupon_rate: float, 
                           ytm: float, years_to_maturity: float, 
                           payments_per_year: int = 2) -> Dict[str, float]:
        """Calculate clean price of a bond based on its face value, coupon rate, YTM and maturity"""
        try:
            # Convert percentages to decimals
            coupon_rate_decimal = coupon_rate / 100
            ytm_decimal = ytm / 100
            
            # Calculate values
            periods = years_to_maturity * payments_per_year
            coupon_payment = (face_value * coupon_rate_decimal) / payments_per_year
            rate_per_period = ytm_decimal / payments_per_year
            
            # Present value of coupon payments
            pv_coupons = coupon_payment * (1 - (1 + rate_per_period) ** -periods) / rate_per_period
            
            # Present value of principal
            pv_principal = face_value / (1 + rate_per_period) ** periods
            
            # Total bond price (clean price)
            bond_price = pv_coupons + pv_principal
            
            return {
                "clean_price": round(bond_price, 2),
                "pv_coupons": round(pv_coupons, 2),
                "pv_principal": round(pv_principal, 2)
            }
        except Exception as e:
            self.logger.error(f"Error calculating bond price: {str(e)}")
            return {"error": f"Error calculating bond price: {str(e)}"}
    
    def calculate_consideration_amount(self, face_value: float, units: int, price: float) -> Dict[str, float]:
        """Calculate consideration amount for a bond transaction"""
        try:
            trade_value = (price / 100) * face_value * units
            return {
                "trade_value": round(trade_value, 2),
                "price_percentage": price,
                "units": units,
                "face_value": face_value
            }
        except Exception as e:
            self.logger.error(f"Error calculating consideration amount: {str(e)}")
            return {"error": f"Error calculating consideration amount: {str(e)}"}
    
    def calculate_ytm_from_price(self, face_value: float, coupon_rate: float, 
                               price: float, years_to_maturity: float,
                               payments_per_year: int = 2) -> Dict[str, float]:
        """Calculate yield to maturity from bond price using iterative approach"""
        try:
            # Convert percentages
            coupon_rate_decimal = coupon_rate / 100
            
            # Parameters
            periods = years_to_maturity * payments_per_year
            coupon_payment = (face_value * coupon_rate_decimal) / payments_per_year
            
            # Initial guesses for YTM
            lower_ytm = 0.001  # 0.1%
            upper_ytm = 0.50   # 50%
            
            # Target price
            target_price = price / 100 * face_value
            
            # Iterative approach to find YTM
            for _ in range(100):  # Max 100 iterations
                mid_ytm = (lower_ytm + upper_ytm) / 2
                rate_per_period = mid_ytm / payments_per_year
                
                # Calculate price at this YTM
                pv_coupons = coupon_payment * (1 - (1 + rate_per_period) ** -periods) / rate_per_period
                pv_principal = face_value / (1 + rate_per_period) ** periods
                calculated_price = pv_coupons + pv_principal
                
                # Check if we're close enough
                if abs(calculated_price - target_price) < 0.01:
                    return {
                        "ytm_percentage": round(mid_ytm * 100, 3),
                        "rate_per_period": round(rate_per_period * 100, 3),
                        "calculated_price": round(calculated_price, 2)
                    }
                
                # Adjust bounds
                if calculated_price > target_price:
                    lower_ytm = mid_ytm
                else:
                    upper_ytm = mid_ytm
            
            # If we didn't converge, return the best estimate
            return {
                "ytm_percentage": round(mid_ytm * 100, 3),
                "rate_per_period": round(rate_per_period * 100, 3),
                "calculated_price": round(calculated_price, 2),
                "note": "Result is approximate; didn't fully converge"
            }
        except Exception as e:
            self.logger.error(f"Error calculating YTM: {str(e)}")
            return {"error": f"Error calculating YTM: {str(e)}"}
