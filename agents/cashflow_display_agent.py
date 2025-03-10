import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import json
import streamlit as st

class CashflowDisplayAgent:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def format_cashflow_summary(self, cashflow_df: pd.DataFrame, bond_details: Dict[str, Any] = None) -> Dict:
        """Format cashflow data into a structured summary"""
        try:
            if cashflow_df.empty:
                return {
                    "error": "No cashflow data available",
                    "success": False
                }
            
            # Ensure date column is datetime
            if 'cash_flow_date' in cashflow_df.columns:
                cashflow_df['cash_flow_date'] = pd.to_datetime(cashflow_df['cash_flow_date'])
                cashflow_df = cashflow_df.sort_values('cash_flow_date')
            
            # Extract bond details
            bond_info = {}
            if 'isin' in cashflow_df.columns:
                bond_info['isin'] = cashflow_df['isin'].iloc[0]
            
            # Add additional bond details if provided
            if bond_details:
                if 'company_name' in bond_details:
                    bond_info['issuer'] = bond_details['company_name']
                if 'face_value' in bond_details:
                    bond_info['face_value'] = bond_details['face_value']
                if 'coupon_details' in bond_details and isinstance(bond_details['coupon_details'], dict):
                    if 'rate' in bond_details['coupon_details']:
                        bond_info['coupon'] = f"{float(bond_details['coupon_details']['rate']):.4f}%"
            
            # Calculate aggregate metrics
            total_cashflow = cashflow_df['cash_flow_amount'].sum() if 'cash_flow_amount' in cashflow_df.columns else 0
            total_principal = cashflow_df['principal_amount'].sum() if 'principal_amount' in cashflow_df.columns else 0
            total_interest = cashflow_df['interest_amount'].sum() if 'interest_amount' in cashflow_df.columns else 0
            
            # Format cashflow table
            cashflow_table = []
            for _, row in cashflow_df.iterrows():
                payment = {}
                
                # Format date
                if 'cash_flow_date' in row:
                    payment['date'] = row['cash_flow_date'].strftime('%d-%b-%Y')
                
                # Format amounts
                if 'cash_flow_amount' in row:
                    payment['cashflow'] = float(row['cash_flow_amount'])
                
                if 'record_date' in row and pd.notna(row['record_date']):
                    payment['record_date'] = pd.to_datetime(row['record_date']).strftime('%d-%b-%Y')
                else:
                    payment['record_date'] = ""
                    
                if 'principal_amount' in row:
                    payment['principal'] = float(row['principal_amount'])
                
                if 'interest_amount' in row:
                    payment['interest'] = float(row['interest_amount'])
                
                if 'remaining_principal' in row:
                    payment['remaining_principal'] = float(row['remaining_principal'])
                
                cashflow_table.append(payment)
            
            # Calculate YTM if trade details are available
            ytm = None
            npv = None
            if bond_details and 'trade_details' in bond_details:
                trade = bond_details['trade_details']
                if 'yield_to_maturity' in trade:
                    ytm = f"{float(trade['yield_to_maturity']):.4f}%"
                if 'npv' in trade:
                    npv = float(trade['npv'])
            
            return {
                "bond_info": bond_info,
                "cashflow_table": cashflow_table,
                "metrics": {
                    "total_cashflow": total_cashflow,
                    "total_principal": total_principal,
                    "total_interest": total_interest,
                    "ytm": ytm,
                    "npv": npv
                },
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"Error formatting cashflow summary: {str(e)}")
            return {
                "error": f"Error formatting cashflow summary: {str(e)}",
                "success": False
            }
    
    def display_cashflow_summary(self, cashflow_df: pd.DataFrame, bond_details: Dict[str, Any] = None):
        """Display formatted cashflow summary in Streamlit"""
        try:
            summary = self.format_cashflow_summary(cashflow_df, bond_details)
            
            if not summary["success"]:
                st.error(summary["error"])
                return
            
            # Display bond information
            st.subheader("Bond Details")
            col1, col2, col3 = st.columns(3)
            
            bond_info = summary["bond_info"]
            with col1:
                st.metric("ISIN", bond_info.get("isin", "N/A"))
            with col2:
                st.metric("Issuer", bond_info.get("issuer", "N/A"))
            with col3:
                st.metric("Face Value", bond_info.get("face_value", "N/A"))
            
            # Display metrics
            st.subheader("Cashflow Metrics")
            metrics = summary["metrics"]
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Interest", f"₹{metrics['total_interest']:,.2f}")
            with col2:
                st.metric("Total Principal", f"₹{metrics['total_principal']:,.2f}")
            with col3:
                st.metric("YTM", metrics.get("ytm", "N/A"))
            with col4:
                if metrics.get("npv"):
                    st.metric("NPV", f"₹{metrics['npv']:,.2f}")
            
            # Display cashflow table
            st.subheader("Cashflow Schedule")
            
            cashflow_table = summary["cashflow_table"]
            if cashflow_table:
                df = pd.DataFrame(cashflow_table)
                
                # Format the display columns
                display_columns = []
                column_configs = {}
                
                if 'date' in df.columns:
                    display_columns.append('date')
                    column_configs['date'] = st.column_config.TextColumn("Payment Date")
                
                if 'cashflow' in df.columns:
                    display_columns.append('cashflow')
                    column_configs['cashflow'] = st.column_config.NumberColumn(
                        "Cashflow Amount", format="₹%.2f"
                    )
                
                if 'record_date' in df.columns:
                    display_columns.append('record_date')
                    column_configs['record_date'] = st.column_config.TextColumn("Record Date")
                
                if 'principal' in df.columns:
                    display_columns.append('principal')
                    column_configs['principal'] = st.column_config.NumberColumn(
                        "Principal", format="₹%.2f"
                    )
                
                if 'interest' in df.columns:
                    display_columns.append('interest')
                    column_configs['interest'] = st.column_config.NumberColumn(
                        "Interest", format="₹%.2f"
                    )
                
                if 'remaining_principal' in df.columns:
                    display_columns.append('remaining_principal')
                    column_configs['remaining_principal'] = st.column_config.NumberColumn(
                        "Remaining Principal", format="₹%.2f"
                    )
                
                st.dataframe(df[display_columns], column_config=column_configs, use_container_width=True)
                
                # CSV Download option
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Cashflow CSV",
                    data=csv,
                    file_name=f"{bond_info.get('isin', 'cashflow')}_schedule.csv",
                    mime="text/csv"
                )
            else:
                st.info("No cashflow details available")
                
        except Exception as e:
            self.logger.error(f"Error displaying cashflow summary: {str(e)}")
            st.error(f"Error displaying cashflow: {str(e)}")
    
    def parse_cashflow_from_text(self, cashflow_text: str) -> pd.DataFrame:
        """Parse cashflow data from text format"""
        try:
            # Parse the cashflow table
            lines = cashflow_text.strip().split('\n')
            header = lines[0].split('\t')
            
            data = []
            for line in lines[1:]:
                if line.strip() and '\t' in line:
                    row = line.split('\t')
                    if len(row) > 0:
                        data.append(row)
            
            # Create dataframe
            df = pd.DataFrame(data)
            
            # If we have at least as many columns as in the header
            if df.shape[1] >= len(header):
                df = df.iloc[:, :len(header)]
                df.columns = header
                
                # Extract bond details (if available)
                bond_details = {}
                for i, line in enumerate(lines):
                    if '\t' in line:
                        parts = line.split('\t')
                        if len(parts) >= 3:
                            key, value = parts[0], parts[2]
                            if key == 'ISIN':
                                bond_details['isin'] = value
                            elif key == 'Issuer':
                                bond_details['issuer'] = value
                            elif key == 'Face Value':
                                try:
                                    bond_details['face_value'] = float(value.replace(',', ''))
                                except:
                                    bond_details['face_value'] = value
                            elif key == 'YTM':
                                bond_details['ytm'] = value
                            elif key == 'Coupon':
                                bond_details['coupon'] = value
                            elif key == 'NPV':
                                try:
                                    bond_details['npv'] = float(value.replace(',', ''))
                                except:
                                    bond_details['npv'] = value
                                    
                # Convert columns to appropriate types
                if 'Cashflow Date' in df.columns:
                    df['cash_flow_date'] = pd.to_datetime(df['Cashflow Date'], errors='coerce')
                    df = df.drop('Cashflow Date', axis=1)
                
                if 'Cashflow' in df.columns:
                    df['cash_flow_amount'] = pd.to_numeric(df['Cashflow'], errors='coerce')
                    df = df.drop('Cashflow', axis=1)
                
                if 'Record Date' in df.columns:
                    df['record_date'] = pd.to_datetime(df['Record Date'], errors='coerce')
                    df = df.drop('Record Date', axis=1)
                
                if 'Principal Amount' in df.columns:
                    df['principal_amount'] = pd.to_numeric(df['Principal Amount'], errors='coerce')
                    df = df.drop('Principal Amount', axis=1)
                
                if 'Interest Amount' in df.columns:
                    df['interest_amount'] = pd.to_numeric(df['Interest Amount'], errors='coerce')
                    df = df.drop('Interest Amount', axis=1)
                
                # Add bond details
                if bond_details.get('isin'):
                    df['isin'] = bond_details['isin']
                
                return df, bond_details
            
            return pd.DataFrame(), {}
            
        except Exception as e:
            self.logger.error(f"Error parsing cashflow text: {str(e)}")
            return pd.DataFrame(), {}
