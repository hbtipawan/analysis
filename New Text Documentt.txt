import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="Advanced Stock Analysis Dashboard", layout="wide")

class StockAnalysisDashboard:
    def __init__(self):
        self.stock_data = None
        self.master_data = None
        self.index_data = None
        self.merged_data = None
        
        # Define sector-specific valuation metrics
        self.sector_valuation_metrics = {
            'Hospitals': ['EV/EBITDA Ratio', 'Price / CFO'],
            'Platforms': ['Price / Free Cash Flow', 'PE Ratio', 'Price / CFO'],
            'Capital Markets': ['PE Ratio', 'Net Profit Margin'],
            'Capital Goods': ['PE Ratio', 'Net Profit Margin'],
            'FMCG': ['PE Ratio', '1Y Historical Revenue Growth'],
            'Consumer': ['PE Ratio', '1Y Historical Revenue Growth'],
            'Commodity': ['PB Ratio', 'PE Ratio', 'Net Profit Margin'],
            'Real Estate': ['PB Ratio', 'Net Profit Margin'],
            'Defence': ['PE Ratio', 'Net Profit Margin'],
            'Packaging': ['PS Ratio', 'PB Ratio', 'Net Profit Margin'],
            'Cement': ['EV/EBITDA Ratio'],
            'IT': ['PE Ratio', 'Price / CFO'],
            'IT Product': ['Price / Sales', 'Price / CFO'],
            'Hotels': ['EV/EBITDA Ratio', 'PE Ratio', 'Price / CFO'],
            'Auto': ['PE Ratio', 'Net Profit Margin', '1Y Historical Revenue Growth'],
            'Auto Ancillaries': ['Net Profit Margin', 'EV/EBITDA Ratio', 'PE Ratio'],
            'Retail': ['EV/EBITDA Ratio', 'PE Ratio'],
            'Pharma CDMO': ['Price / Sales', 'PE Ratio', 'EV/EBITDA Ratio', 'Net Profit Margin'],
            'Pharma Generic': ['PE Ratio', 'EV/EBITDA Ratio', 'Net Profit Margin'],
            'Pharma API': ['PE Ratio', 'Price / Sales', 'Net Profit Margin'],
            'EMS Players': ['PE Ratio', '1Y Historical Revenue Growth', 'EBITDA Margin']
        }
        
    def load_data(self, stock_file, master_file, index_file=None):
        """Load data from uploaded files"""
        try:
            # Load stock data with encoding handling
            if stock_file.name.endswith('.csv'):
                try:
                    self.stock_data = pd.read_csv(stock_file, encoding='utf-8')
                except UnicodeDecodeError:
                    self.stock_data = pd.read_csv(stock_file, encoding='latin1')
            else:
                self.stock_data = pd.read_excel(stock_file)
            
            # Load master data with encoding handling
            if master_file.name.endswith('.csv'):
                try:
                    self.master_data = pd.read_csv(master_file, encoding='utf-8')
                except UnicodeDecodeError:
                    self.master_data = pd.read_csv(master_file, encoding='latin1')
            else:
                self.master_data = pd.read_excel(master_file)
            
            # Load index data if provided with encoding handling
            if index_file:
                if index_file.name.endswith('.csv'):
                    try:
                        self.index_data = pd.read_csv(index_file, encoding='utf-8')
                    except UnicodeDecodeError:
                        self.index_data = pd.read_csv(index_file, encoding='latin1')
                else:
                    self.index_data = pd.read_excel(index_file)
            
            # Clean column names - remove extra spaces
            self.stock_data.columns = self.stock_data.columns.str.strip()
            self.master_data.columns = self.master_data.columns.str.strip()
            if self.index_data is not None:
                self.index_data.columns = self.index_data.columns.str.strip()
            
            # Clean data before merging
            # Remove any leading/trailing spaces in Company Name
            self.stock_data['Company Name'] = self.stock_data['Company Name'].str.strip()
            self.master_data['Company Name'] = self.master_data['Company Name'].str.strip()
            
            # Merge stock data with master data
            self.merged_data = pd.merge(
                self.stock_data, 
                self.master_data, 
                on='Company Name', 
                how='left'
            )
            
            # Fill missing Sector/Industry with 'Others'
            self.merged_data['Sector'] = self.merged_data['Sector'].fillna('Others')
            self.merged_data['Industry'] = self.merged_data['Industry'].fillna('Others')
            
            # Convert numeric columns - handle column names with spaces
            numeric_columns = ['Market Cap', 'PE Ratio', 'PB Ratio', 'Return on Equity', 
                             '1Y Return', 'Debt to Equity', 'ROCE', '1W Return', '1M Return', 
                             '6M Return', '3M Return', 'Close Price', 'Operating Cash Flow',
                             'Free Cash Flow', 'Net Income', 'Interest Coverage Ratio',
                             'Days of Sales Outstanding', 'Working Capital Turnover Ratio',
                             'Promoter Holding Change 6M', 'MF Holding Change 6M',
                             'MF Holding Change 3M', 'FII Holding Change 3M']
            
            for col in numeric_columns:
                if col in self.merged_data.columns:
                    self.merged_data[col] = pd.to_numeric(self.merged_data[col], errors='coerce')
            
            return True
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
    
    def calculate_forensic_scores(self, df):
        """Calculate forensic accounting scores"""
        forensic_scores = pd.DataFrame(index=df.index)
        
        # 1. Cash flow quality check
        forensic_scores['cash_flow_quality'] = np.where(
            (df['Operating Cash Flow'] > 0) & (df['Net Income'] > 0),
            df['Operating Cash Flow'] / df['Net Income'],
            0
        )
        
        # 2. Days Sales Outstanding trend
        forensic_scores['dso_flag'] = np.where(
            df['Days of Sales Outstanding'] > df['Days of Sales Outstanding'].quantile(0.75),
            -1, 1
        )
        
        # 3. Inventory turnover check
        forensic_scores['inventory_flag'] = np.where(
            df['Inventory Turnover Ratio'] < df['Inventory Turnover Ratio'].quantile(0.25),
            -1, 1
        )
        
        # 4. Debt to equity warning
        forensic_scores['debt_flag'] = np.where(
            df['Debt to Equity'] > 2, -1, 1
        )
        
        # 5. Interest coverage check
        forensic_scores['interest_coverage_flag'] = np.where(
            df['Interest Coverage Ratio'] < 1.5, -1, 1
        )
        
        # 6. Working capital efficiency
        forensic_scores['working_capital_flag'] = np.where(
            df['Working Capital Turnover Ratio'] < df['Working Capital Turnover Ratio'].quantile(0.25),
            -1, 1
        )
        
        # 7. Asset quality check
        forensic_scores['asset_quality'] = df['Return on Assets'] / df['5Y Avg Return on Assets']
        forensic_scores['asset_quality_flag'] = np.where(
            forensic_scores['asset_quality'] < 0.8, -1, 1
        )
        
        # Calculate overall forensic score
        forensic_scores['forensic_score'] = (
            forensic_scores['cash_flow_quality'].fillna(0) * 0.2 +
            forensic_scores['dso_flag'] * 0.15 +
            forensic_scores['inventory_flag'] * 0.15 +
            forensic_scores['debt_flag'] * 0.15 +
            forensic_scores['interest_coverage_flag'] * 0.15 +
            forensic_scores['working_capital_flag'] * 0.1 +
            forensic_scores['asset_quality_flag'] * 0.1
        )
        
        return forensic_scores['forensic_score']
    
    def calculate_sector_specific_scores(self, df):
        """Calculate scores based on sector-specific valuation metrics"""
        sector_scores = pd.Series(index=df.index, dtype=float)
        
        for idx, row in df.iterrows():
            sector = row.get('Sector', 'General')
            score = 0
            count = 0
            
            # Get sector-specific metrics
            metrics = self.sector_valuation_metrics.get(sector, ['PE Ratio', 'PB Ratio', 'Net Profit Margin'])
            
            # Calculate percentile scores for each metric
            for metric in metrics:
                if metric in df.columns and pd.notna(row[metric]):
                    # Get sector peers
                    sector_data = df[df['Sector'] == sector][metric].dropna()
                    if len(sector_data) > 0:
                        percentile = (sector_data < row[metric]).sum() / len(sector_data)
                        # Invert percentile for metrics where lower is better
                        if metric in ['PE Ratio', 'PB Ratio', 'EV/EBITDA Ratio', 'Price / Sales']:
                            percentile = 1 - percentile
                        score += percentile
                        count += 1
            
            sector_scores[idx] = score / count if count > 0 else 0.5
        
        return sector_scores
    
    def calculate_momentum_scores(self, df, period):
        """Calculate momentum scores for different time periods"""
        momentum_scores = pd.DataFrame(index=df.index)
        
        # Define weights for different periods
        weights = {
            'very_short': {'1D Return': 0.3, '1W Return': 0.4, '1M Return': 0.3},
            'short': {'1W Return': 0.2, '1M Return': 0.5, '6M Return': 0.3},
            'medium': {'1M Return': 0.2, '6M Return': 0.5, '1Y Return': 0.3},
            'long': {'6M Return': 0.3, '1Y Return': 0.4, '5Y CAGR': 0.3}
        }
        
        period_weights = weights.get(period, weights['medium'])
        
        score = 0
        for metric, weight in period_weights.items():
            if metric in df.columns:
                # Normalize returns to 0-1 scale
                normalized = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())
                score += normalized.fillna(0.5) * weight
        
        # Add technical indicators
        if 'RSI 14D' in df.columns:
            rsi_score = np.where(
                (df['RSI 14D'] > 30) & (df['RSI 14D'] < 70),
                0.5 + (df['RSI 14D'] - 50) / 100,
                np.where(df['RSI 14D'] <= 30, 1, 0)
            )
            score += rsi_score * 0.2
        
        return score
    
    def calculate_multibagger_scores(self, df):
        """Identify potential multibagger stocks"""
        mb_scores = pd.DataFrame(index=df.index)
        
        # High growth with reasonable valuation
        mb_scores['growth_score'] = (
            df['5Y Historical Revenue Growth'].fillna(0) * 0.3 +
            df['5Y Historical EPS Growth'].fillna(0) * 0.3 +
            df['5Y Historical EBITDA Growth'].fillna(0) * 0.2 +
            df['1Y Forward Revenue Growth'].fillna(0) * 0.2
        ) / 100
        
        # Low debt, high efficiency
        mb_scores['quality_score'] = (
            (1 - df['Debt to Equity'].fillna(0) / df['Debt to Equity'].max()) * 0.3 +
            df['Return on Equity'].fillna(0) / 100 * 0.3 +
            df['ROCE'].fillna(0) / 100 * 0.2 +
            df['Free Cash Flow'].fillna(0) / df['Free Cash Flow'].max() * 0.2
        )
        
        # Reasonable valuation
        mb_scores['valuation_score'] = (
            (1 - df['PE Ratio'].fillna(50) / 100) * 0.5 +
            (1 - df['PB Ratio'].fillna(5) / 10) * 0.5
        )
        
        # Institutional interest
        mb_scores['institutional_score'] = (
            df['MF Holding Change 6M'].fillna(0) / 100 * 0.5 +
            df['FII Holding Change 6M'].fillna(0) / 100 * 0.5
        )
        
        # Combined score
        mb_scores['total_score'] = (
            mb_scores['growth_score'] * 0.35 +
            mb_scores['quality_score'] * 0.35 +
            mb_scores['valuation_score'] * 0.2 +
            mb_scores['institutional_score'] * 0.1
        )
        
        return mb_scores['total_score']
    
    def calculate_turnaround_scores(self, df):
        """Identify turnaround candidates"""
        ta_scores = pd.DataFrame(index=df.index)
        
        # Improving fundamentals
        ta_scores['improvement_score'] = (
            np.where(df['1Y Historical Revenue Growth'] > df['5Y Historical Revenue Growth'], 1, 0) * 0.25 +
            np.where(df['Net Profit Margin'] > df['5Y Avg Net Profit Margin'], 1, 0) * 0.25 +
            np.where(df['Return on Equity'] > df['5Y Avg Return on Equity'], 1, 0) * 0.25 +
            np.where(df['EBITDA Margin'] > df['5Y Avg EBITDA Margin'], 1, 0) * 0.25
        )
        
        # Recovery from lows
        ta_scores['recovery_score'] = (100 - df['% Away From 52W Low'].fillna(50)) / 100
        
        # Increasing institutional interest
        ta_scores['interest_score'] = (
            df['MF Holding Change 3M'].fillna(0) / 100 * 0.5 +
            df['FII Holding Change 3M'].fillna(0) / 100 * 0.5
        )
        
        # Combined score
        ta_scores['total_score'] = (
            ta_scores['improvement_score'] * 0.5 +
            ta_scores['recovery_score'] * 0.3 +
            ta_scores['interest_score'] * 0.2
        )
        
        return ta_scores['total_score']
    
    def calculate_overall_rank(self, df):
        """Calculate comprehensive ranking for all stocks"""
        # Calculate individual component scores
        forensic_scores = self.calculate_forensic_scores(df)
        sector_scores = self.calculate_sector_specific_scores(df)
        
        # Quality metrics
        quality_score = (
            df['Return on Equity'].fillna(0) / 100 * 0.2 +
            df['ROCE'].fillna(0) / 100 * 0.2 +
            df['Net Profit Margin'].fillna(0) / 100 * 0.2 +
            (1 - df['Debt to Equity'].fillna(0) / df['Debt to Equity'].max()) * 0.2 +
            df['Interest Coverage Ratio'].fillna(0) / df['Interest Coverage Ratio'].max() * 0.2
        )
        
        # Growth metrics
        growth_score = (
            df['5Y CAGR'].fillna(0) / 100 * 0.3 +
            df['1Y Historical Revenue Growth'].fillna(0) / 100 * 0.3 +
            df['1Y Historical EPS Growth'].fillna(0) / 100 * 0.2 +
            df['1Y Forward Revenue Growth'].fillna(0) / 100 * 0.2
        )
        
        # Combined score with weights
        overall_score = (
            sector_scores * 0.35 +          # Sector-specific valuation
            quality_score * 0.25 +          # Quality metrics
            growth_score * 0.20 +           # Growth metrics
            forensic_scores * 0.20          # Forensic accounting
        )
        
        # Create ranking (higher score = better rank)
        df['Overall_Score'] = overall_score
        df['Overall_Rank'] = df['Overall_Score'].rank(ascending=False, method='min')
        
        return df
    
    def display_ranking_tab(self, df):
        """Display the main ranking tab"""
        st.header("📊 Overall Stock Rankings")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            sectors = ['All'] + sorted(df['Sector'].dropna().unique().tolist())
            selected_sector = st.selectbox("Filter by Sector", sectors)
        
        with col2:
            if selected_sector != 'All':
                industries = ['All'] + sorted(df[df['Sector'] == selected_sector]['Industry'].dropna().unique().tolist())
            else:
                industries = ['All'] + sorted(df['Industry'].dropna().unique().tolist())
            selected_industry = st.selectbox("Filter by Industry", industries)
        
        # Apply filters
        filtered_df = df.copy()
        if selected_sector != 'All':
            filtered_df = filtered_df[filtered_df['Sector'] == selected_sector]
        if selected_industry != 'All':
            filtered_df = filtered_df[filtered_df['Industry'] == selected_industry]
        
        # Display table
        display_cols = ['Overall_Rank', 'Company Name', 'Sector', 'Industry', 'Market Cap', 
                       'PE Ratio', 'PB Ratio', 'Return on Equity', '1Y Return', 'Overall_Score']
        
        filtered_df = filtered_df.sort_values('Overall_Rank')
        st.dataframe(
            filtered_df[display_cols].head(100),
            use_container_width=True,
            hide_index=True
        )
        
        # Download button
        csv = filtered_df[display_cols].to_csv(index=False)
        st.download_button(
            label="Download Rankings as CSV",
            data=csv,
            file_name=f"stock_rankings_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    def display_momentum_tab(self, df, period, title, days):
        """Display momentum-based rankings"""
        st.header(f"📈 {title}")
        st.write(f"Top 50 stocks with potential upmove in the next {days}")
        
        # Calculate momentum scores
        momentum_scores = self.calculate_momentum_scores(df, period)
        df['Momentum_Score'] = momentum_scores
        df['Momentum_Rank'] = df['Momentum_Score'].rank(ascending=False, method='min')
        
        # Get top 50
        top_stocks = df.nsmallest(50, 'Momentum_Rank')
        
        # Display columns based on period
        if period == 'very_short':
            display_cols = ['Momentum_Rank', 'Company Name', 'Sector', '1D Return', '1W Return', 
                           'RSI 14D', 'Relative Volume', 'Momentum_Score']
        elif period == 'short':
            display_cols = ['Momentum_Rank', 'Company Name', 'Sector', '1W Return', '1M Return', 
                           'RSI 14D', 'MACD Line 1 Trend Indicator', 'Momentum_Score']
        elif period == 'medium':
            display_cols = ['Momentum_Rank', 'Company Name', 'Sector', '1M Return', '6M Return', 
                           'PE Ratio', 'PB Ratio', 'Momentum_Score']
        else:  # long
            display_cols = ['Momentum_Rank', 'Company Name', 'Sector', '6M Return', '1Y Return', 
                           '5Y CAGR', 'Return on Equity', 'Momentum_Score']
        
        st.dataframe(
            top_stocks[display_cols],
            use_container_width=True,
            hide_index=True
        )
        
        # Download button
        csv = top_stocks[display_cols].to_csv(index=False)
        st.download_button(
            label=f"Download {title} as CSV",
            data=csv,
            file_name=f"{title.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    def display_multibagger_tab(self, df):
        """Display potential multibagger stocks"""
        st.header("🚀 Potential Multibaggers")
        st.write("Top 50 stocks with multibagger potential based on growth, quality, and valuation")
        
        # Calculate multibagger scores
        mb_scores = self.calculate_multibagger_scores(df)
        df['Multibagger_Score'] = mb_scores
        df['Multibagger_Rank'] = df['Multibagger_Score'].rank(ascending=False, method='min')
        
        # Get top 50
        top_stocks = df.nsmallest(50, 'Multibagger_Rank')
        
        display_cols = ['Multibagger_Rank', 'Company Name', 'Sector', 'Market Cap',
                       '5Y Historical Revenue Growth', '5Y Historical EPS Growth',
                       'Return on Equity', 'Debt to Equity', 'PE Ratio', 'Multibagger_Score']
        
        st.dataframe(
            top_stocks[display_cols],
            use_container_width=True,
            hide_index=True
        )
        
        # Download button
        csv = top_stocks[display_cols].to_csv(index=False)
        st.download_button(
            label="Download Multibaggers as CSV",
            data=csv,
            file_name=f"multibaggers_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    def display_turnaround_tab(self, df):
        """Display turnaround candidates"""
        st.header("🔄 Turnaround Candidates")
        st.write("Top 25 companies showing signs of turnaround")
        
        # Calculate turnaround scores
        ta_scores = self.calculate_turnaround_scores(df)
        df['Turnaround_Score'] = ta_scores
        df['Turnaround_Rank'] = df['Turnaround_Score'].rank(ascending=False, method='min')
        
        # Get top 25
        top_stocks = df.nsmallest(25, 'Turnaround_Rank')
        
        display_cols = ['Turnaround_Rank', 'Company Name', 'Sector', 
                       '% Away From 52W Low', '1Y Historical Revenue Growth',
                       'Net Profit Margin', '5Y Avg Net Profit Margin',
                       'MF Holding Change 3M', 'Turnaround_Score']
        
        st.dataframe(
            top_stocks[display_cols],
            use_container_width=True,
            hide_index=True
        )
        
        # Download button
        csv = top_stocks[display_cols].to_csv(index=False)
        st.download_button(
            label="Download Turnaround Stocks as CSV",
            data=csv,
            file_name=f"turnaround_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    def display_stock_deep_dive(self, df):
        """Display detailed analysis for a selected stock"""
        st.header("🔍 Stock Deep Dive")
        
        # Use session state to maintain selected stock
        if 'selected_stock' not in st.session_state:
            st.session_state.selected_stock = None
        
        # Stock search
        stock_names = sorted(df['Company Name'].tolist())
        
        # Use a unique key for the selectbox
        selected_stock = st.selectbox(
            "Select or Search for a Stock", 
            stock_names,
            index=stock_names.index(st.session_state.selected_stock) if st.session_state.selected_stock in stock_names else 0,
            key='stock_selector'
        )
        
        # Update session state
        st.session_state.selected_stock = selected_stock
        
        if selected_stock:
            stock_data = df[df['Company Name'] == selected_stock].iloc[0]
            
            # Basic info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Sector", stock_data['Sector'])
                st.metric("Market Cap (Cr)", f"₹{float(stock_data['Market Cap']):.0f}")
            with col2:
                st.metric("Industry", stock_data['Industry'])
                st.metric("PE Ratio", f"{float(stock_data['PE Ratio']):.2f}")
            with col3:
                st.metric("Overall Rank", f"{int(stock_data['Overall_Rank'])}")
                st.metric("PB Ratio", f"{float(stock_data['PB Ratio']):.2f}")
            with col4:
                st.metric("Close Price", f"₹{float(stock_data['Close Price']):.2f}")
                st.metric("1Y Return", f"{float(stock_data['1Y Return']):.2f}%")
            
            # Tabs for different analyses
            tab1, tab2, tab3, tab4 = st.tabs(["Key Metrics", "Strengths", "Cautions", "Charts"])
            
            with tab1:
                # Sector-specific metrics
                sector = stock_data['Sector']
                sector_metrics = self.sector_valuation_metrics.get(sector, ['PE Ratio', 'PB Ratio'])
                
                st.subheader(f"Key Valuation Metrics for {sector} Sector")
                metric_cols = st.columns(len(sector_metrics))
                for i, metric in enumerate(sector_metrics):
                    if metric in stock_data:
                        metric_cols[i].metric(metric, f"{stock_data[metric]:.2f}")
                
                # Other important metrics
                st.subheader("Financial Health Indicators")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("ROE", f"{stock_data['Return on Equity']:.2f}%")
                    st.metric("ROCE", f"{stock_data['ROCE']:.2f}%")
                with col2:
                    st.metric("Debt to Equity", f"{stock_data['Debt to Equity']:.2f}")
                    st.metric("Interest Coverage", f"{stock_data['Interest Coverage Ratio']:.2f}")
                with col3:
                    st.metric("Operating Cash Flow", f"₹{stock_data['Operating Cash Flow']:,.0f}")
                    st.metric("Free Cash Flow", f"₹{stock_data['Free Cash Flow']:,.0f}")
            
            with tab2:
                st.subheader("✅ Key Strengths")
                strengths = []
                
                # Analyze strengths
                if stock_data['Return on Equity'] > 15:
                    strengths.append(f"Strong ROE of {stock_data['Return on Equity']:.2f}% indicates efficient use of shareholder equity")
                
                if stock_data['Debt to Equity'] < 0.5:
                    strengths.append(f"Low debt-to-equity ratio of {stock_data['Debt to Equity']:.2f} shows conservative capital structure")
                
                if stock_data['5Y CAGR'] > 15:
                    strengths.append(f"Impressive 5-year CAGR of {stock_data['5Y CAGR']:.2f}% demonstrates consistent growth")
                
                if stock_data['Operating Cash Flow'] > stock_data['Net Income']:
                    strengths.append("Strong cash generation with operating cash flow exceeding net income")
                
                if stock_data['MF Holding Change 6M'] > 0:
                    strengths.append(f"Increasing institutional interest with {stock_data['MF Holding Change 6M']:.2f}% rise in MF holdings")
                
                for i, strength in enumerate(strengths[:5], 1):
                    st.write(f"{i}. {strength}")
            
            with tab3:
                st.subheader("⚠️ Areas of Caution")
                cautions = []
                
                # Forensic checks
                if stock_data['Days of Sales Outstanding'] > 90:
                    cautions.append(f"High DSO of {stock_data['Days of Sales Outstanding']:.0f} days may indicate collection issues")
                
                if stock_data['Interest Coverage Ratio'] < 2:
                    cautions.append(f"Low interest coverage of {stock_data['Interest Coverage Ratio']:.2f}x poses solvency risk")
                
                if stock_data['PE Ratio'] > stock_data['Sector PE'] * 1.5:
                    cautions.append("Trading at significant premium to sector PE, may be overvalued")
                
                if stock_data['Promoter Holding Change 6M'] < -5:
                    cautions.append(f"Promoter stake decreased by {abs(stock_data['Promoter Holding Change 6M']):.2f}% in last 6 months")
                
                if stock_data['Working Capital Turnover Ratio'] < 2:
                    cautions.append("Low working capital efficiency may impact operational performance")
                
                for i, caution in enumerate(cautions[:5], 1):
                    st.write(f"{i}. {caution}")
            
            with tab4:
                # Create charts
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Revenue Trend', 'Profitability Trend', 
                                   'Margin Analysis', 'Return Metrics'),
                    specs=[[{'type': 'bar'}, {'type': 'bar'}],
                          [{'type': 'bar'}, {'type': 'bar'}]]
                )
                
                # Revenue trend
                revenue_metrics = ['Total Revenue', 'EBITDA', 'Net Income']
                revenue_values = [stock_data[m] for m in revenue_metrics if m in stock_data]
                fig.add_trace(
                    go.Bar(x=revenue_metrics, y=revenue_values, name="Financial Metrics"),
                    row=1, col=1
                )
                
                # Profitability margins
                margin_metrics = ['Net Profit Margin', 'EBITDA Margin', 'ROCE']
                margin_values = [stock_data[m] for m in margin_metrics if m in stock_data]
                fig.add_trace(
                    go.Bar(x=margin_metrics, y=margin_values, name="Margins %"),
                    row=1, col=2
                )
                
                # Valuation metrics
                val_metrics = ['PE Ratio', 'PB Ratio', 'EV/EBITDA Ratio']
                val_values = [stock_data[m] for m in val_metrics if m in stock_data]
                fig.add_trace(
                    go.Bar(x=val_metrics, y=val_values, name="Valuation"),
                    row=2, col=1
                )
                
                # Returns
                return_metrics = ['1W Return', '1M Return', '6M Return', '1Y Return']
                return_values = [stock_data[m] for m in return_metrics if m in stock_data]
                fig.add_trace(
                    go.Bar(x=return_metrics, y=return_values, name="Returns %"),
                    row=2, col=2
                )
                
                fig.update_layout(height=800, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    
    def display_sector_rotation(self):
        """Display sector rotation analysis"""
        st.header("📊 Sector Rotation Analysis")
        
        # Filter out rows with missing Sector values
        sector_data = self.merged_data.dropna(subset=['Sector'])
        
        if len(sector_data) > 0:
            # Calculate sector performance metrics
            # First check which columns exist
            available_metrics = {}
            metric_columns = {
                'Market Cap': 'sum',
                'Company Name': 'count',
                '1W Return': 'mean',
                '1M Return': 'mean',
                '6M Return': 'mean',
                '1Y Return': 'mean',
                'PE Ratio': 'median',
                'PB Ratio': 'median',
                'Return on Equity': 'mean',
                'Debt to Equity': 'median'
            }
            
            # Only include columns that exist in the data
            for col, agg_func in metric_columns.items():
                if col in sector_data.columns:
                    available_metrics[col] = agg_func
            
            sector_summary = sector_data.groupby('Sector').agg(available_metrics).round(2)
            
            # Rename columns for clarity - only rename columns that exist
            rename_dict = {}
            if 'Company Name' in sector_summary.columns:
                rename_dict['Company Name'] = 'No. of Stocks'
            if 'Market Cap' in sector_summary.columns:
                rename_dict['Market Cap'] = 'Total Market Cap (Cr)'
            if '1W Return' in sector_summary.columns:
                rename_dict['1W Return'] = 'Avg 1W Return %'
            if '1M Return' in sector_summary.columns:
                rename_dict['1M Return'] = 'Avg 1M Return %'
            if '6M Return' in sector_summary.columns:
                rename_dict['6M Return'] = 'Avg 6M Return %'
            if '1Y Return' in sector_summary.columns:
                rename_dict['1Y Return'] = 'Avg 1Y Return %'
            if 'PE Ratio' in sector_summary.columns:
                rename_dict['PE Ratio'] = 'Median PE'
            if 'PB Ratio' in sector_summary.columns:
                rename_dict['PB Ratio'] = 'Median PB'
            if 'Return on Equity' in sector_summary.columns:
                rename_dict['Return on Equity'] = 'Avg ROE %'
            if 'Debt to Equity' in sector_summary.columns:
                rename_dict['Debt to Equity'] = 'Median D/E'
            
            sector_summary = sector_summary.rename(columns=rename_dict)
            
            # Sort by 1M return if it exists
            if 'Avg 1M Return %' in sector_summary.columns:
                sector_summary = sector_summary.sort_values('Avg 1M Return %', ascending=False)
            
            # Display the table
            st.subheader("Sector Performance Summary")
            
            # Create format dict only for columns that exist
            format_dict = {}
            if 'Total Market Cap (Cr)' in sector_summary.columns:
                format_dict['Total Market Cap (Cr)'] = '₹{:,.0f}'
            if 'Avg 1W Return %' in sector_summary.columns:
                format_dict['Avg 1W Return %'] = '{:.2f}%'
            if 'Avg 1M Return %' in sector_summary.columns:
                format_dict['Avg 1M Return %'] = '{:.2f}%'
            if 'Avg 6M Return %' in sector_summary.columns:
                format_dict['Avg 6M Return %'] = '{:.2f}%'
            if 'Avg 1Y Return %' in sector_summary.columns:
                format_dict['Avg 1Y Return %'] = '{:.2f}%'
            if 'Median PE' in sector_summary.columns:
                format_dict['Median PE'] = '{:.2f}'
            if 'Median PB' in sector_summary.columns:
                format_dict['Median PB'] = '{:.2f}'
            if 'Avg ROE %' in sector_summary.columns:
                format_dict['Avg ROE %'] = '{:.2f}%'
            if 'Median D/E' in sector_summary.columns:
                format_dict['Median D/E'] = '{:.2f}'
            
            # Apply formatting with background gradient only if column exists
            styled_df = sector_summary.style.format(format_dict)
            if 'Avg 1M Return %' in sector_summary.columns:
                styled_df = styled_df.background_gradient(subset=['Avg 1M Return %'], cmap='RdYlGn')
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Create a simple bar chart for sector returns
            st.subheader("Sector Returns Comparison")
            
            # Prepare data for plotting
            returns_data = sector_summary[['Avg 1W Return %', 'Avg 1M Return %', 
                                          'Avg 6M Return %', 'Avg 1Y Return %']].reset_index()
            
            # Create tabs for different time periods
            tab1, tab2, tab3, tab4 = st.tabs(["1 Week", "1 Month", "6 Months", "1 Year"])
            
            with tab1:
                fig1 = go.Figure(data=[
                    go.Bar(x=returns_data['Sector'], y=returns_data['Avg 1W Return %'],
                          marker_color=returns_data['Avg 1W Return %'],
                          marker_colorscale='RdYlGn',
                          text=returns_data['Avg 1W Return %'].round(2),
                          textposition='auto')
                ])
                fig1.update_layout(title="1 Week Returns by Sector", yaxis_title="Return %", height=400)
                st.plotly_chart(fig1, use_container_width=True)
            
            with tab2:
                fig2 = go.Figure(data=[
                    go.Bar(x=returns_data['Sector'], y=returns_data['Avg 1M Return %'],
                          marker_color=returns_data['Avg 1M Return %'],
                          marker_colorscale='RdYlGn',
                          text=returns_data['Avg 1M Return %'].round(2),
                          textposition='auto')
                ])
                fig2.update_layout(title="1 Month Returns by Sector", yaxis_title="Return %", height=400)
                st.plotly_chart(fig2, use_container_width=True)
            
            with tab3:
                fig3 = go.Figure(data=[
                    go.Bar(x=returns_data['Sector'], y=returns_data['Avg 6M Return %'],
                          marker_color=returns_data['Avg 6M Return %'],
                          marker_colorscale='RdYlGn',
                          text=returns_data['Avg 6M Return %'].round(2),
                          textposition='auto')
                ])
                fig3.update_layout(title="6 Month Returns by Sector", yaxis_title="Return %", height=400)
                st.plotly_chart(fig3, use_container_width=True)
            
            with tab4:
                fig4 = go.Figure(data=[
                    go.Bar(x=returns_data['Sector'], y=returns_data['Avg 1Y Return %'],
                          marker_color=returns_data['Avg 1Y Return %'],
                          marker_colorscale='RdYlGn',
                          text=returns_data['Avg 1Y Return %'].round(2),
                          textposition='auto')
                ])
                fig4.update_layout(title="1 Year Returns by Sector", yaxis_title="Return %", height=400)
                st.plotly_chart(fig4, use_container_width=True)
            
            # Top performing stocks in each sector
            st.subheader("Top Performing Stocks by Sector")
            
            # Create a unique key for this selectbox
            selected_sector_for_stocks = st.selectbox(
                "Select Sector to View Top Stocks",
                sorted(sector_data['Sector'].unique()),
                key='sector_rotation_selector'
            )
            
            if selected_sector_for_stocks:
                sector_stocks = sector_data[sector_data['Sector'] == selected_sector_for_stocks].nsmallest(10, 'Overall_Rank')
                display_cols = ['Overall_Rank', 'Company Name', 'Market Cap', '1M Return', '6M Return', 
                               '1Y Return', 'PE Ratio', 'Return on Equity']
                
                # Check which columns exist before displaying
                available_cols = [col for col in display_cols if col in sector_stocks.columns]
                
                # Create format dict only for available columns
                format_dict = {}
                if 'Market Cap' in available_cols:
                    format_dict['Market Cap'] = '₹{:,.0f}'
                if '1M Return' in available_cols:
                    format_dict['1M Return'] = '{:.2f}%'
                if '6M Return' in available_cols:
                    format_dict['6M Return'] = '{:.2f}%'
                if '1Y Return' in available_cols:
                    format_dict['1Y Return'] = '{:.2f}%'
                if 'PE Ratio' in available_cols:
                    format_dict['PE Ratio'] = '{:.2f}'
                if 'Return on Equity' in available_cols:
                    format_dict['Return on Equity'] = '{:.2f}%'
                
                st.dataframe(
                    sector_stocks[available_cols].style.format(format_dict),
                    use_container_width=True,
                    hide_index=True
                )
            
            # Download sector summary
            csv = sector_summary.to_csv()
            st.download_button(
                label="Download Sector Summary as CSV",
                data=csv,
                file_name=f"sector_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No sector data available for analysis.")
    
    def run(self):
        """Main dashboard execution"""
        st.title("🎯 Advanced Stock Analysis Dashboard")
        st.markdown("---")
        
        # Initialize session state
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'merged_data' not in st.session_state:
            st.session_state.merged_data = None
        if 'filtered_data' not in st.session_state:
            st.session_state.filtered_data = None
        
        # Sidebar for file upload and filters
        with st.sidebar:
            st.header("📁 Data Upload")
            
            stock_file = st.file_uploader(
                "Upload Stock Data File",
                type=['csv', 'xlsx'],
                help="Upload the comprehensive stock data file"
            )
            
            master_file = st.file_uploader(
                "Upload Master Data File",
                type=['csv', 'xlsx'],
                help="Upload the sector/industry mapping file"
            )
            
            index_file = st.file_uploader(
                "Upload Index Data File (Optional)",
                type=['csv', 'xlsx'],
                help="Upload for sector rotation analysis"
            )
            
            if stock_file and master_file:
                if st.button("Load Data", type="primary"):
                    with st.spinner("Loading and processing data..."):
                        if self.load_data(stock_file, master_file, index_file):
                            st.success("Data loaded successfully!")
                            # Calculate rankings
                            self.merged_data = self.calculate_overall_rank(self.merged_data)
                            # Store in session state
                            st.session_state.merged_data = self.merged_data
                            st.session_state.filtered_data = self.merged_data.copy()
                            st.session_state.data_loaded = True
            
            # Common Filters Section
            if st.session_state.data_loaded and st.session_state.merged_data is not None:
                st.markdown("---")
                st.header("🔍 Global Filters")
                
                # Market Cap Filter
                st.subheader("Market Cap Filter")
                
                # Get min and max market cap
                min_mcap = float(st.session_state.merged_data['Market Cap'].min())
                max_mcap = float(st.session_state.merged_data['Market Cap'].max())
                
                # Market cap range selector
                mcap_range = st.slider(
                    "Select Market Cap Range (in Crores)",
                    min_value=min_mcap,
                    max_value=max_mcap,
                    value=(min_mcap, max_mcap),
                    format="₹%.0f Cr",
                    key='market_cap_filter'
                )
                
                # Quick filters for market cap
                st.write("Quick Filters:")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Large Cap\n(>₹20,000 Cr)", key='large_cap'):
                        st.session_state.market_cap_filter = (20000.0, max_mcap)
                    if st.button("Mid Cap\n(₹5,000-20,000 Cr)", key='mid_cap'):
                        st.session_state.market_cap_filter = (5000.0, 20000.0)
                with col2:
                    if st.button("Small Cap\n(₹500-5,000 Cr)", key='small_cap'):
                        st.session_state.market_cap_filter = (500.0, 5000.0)
                    if st.button("Micro Cap\n(<₹500 Cr)", key='micro_cap'):
                        st.session_state.market_cap_filter = (min_mcap, 500.0)
                
                if st.button("Reset All Filters", type="secondary", key='reset_filters'):
                    st.session_state.market_cap_filter = (min_mcap, max_mcap)
                
                # Apply filters
                filtered_data = st.session_state.merged_data.copy()
                filtered_data = filtered_data[
                    (filtered_data['Market Cap'] >= mcap_range[0]) & 
                    (filtered_data['Market Cap'] <= mcap_range[1])
                ]
                
                # Show filter summary
                st.markdown("---")
                st.info(f"**Active Filters:**\n- Stocks: {len(filtered_data)} / {len(st.session_state.merged_data)}\n- Market Cap: ₹{mcap_range[0]:,.0f} Cr - ₹{mcap_range[1]:,.0f} Cr")
                
                # Update filtered data in session state
                st.session_state.filtered_data = filtered_data
        
        # Use filtered data from session state
        if st.session_state.data_loaded and st.session_state.filtered_data is not None:
            self.merged_data = st.session_state.filtered_data
            
            # Create tabs
            tabs = st.tabs([
                "📊 Rankings",
                "⚡ Very Short Term (15d)",
                "📈 Short Term (1m)",
                "📉 Medium Short Term (45d)",
                "📊 Medium Term (3m)",
                "📈 Long Term (1y)",
                "🚀 Multibaggers",
                "🔄 Turnarounds",
                "🔍 Stock Analysis",
                "📊 Sector Rotation"
            ])
            
            with tabs[0]:
                self.display_ranking_tab(self.merged_data)
            
            with tabs[1]:
                self.display_momentum_tab(self.merged_data, 'very_short', 
                                        "Very Short Term Opportunities", "15 days")
            
            with tabs[2]:
                self.display_momentum_tab(self.merged_data, 'short', 
                                        "Short Term Opportunities", "1 month")
            
            with tabs[3]:
                self.display_momentum_tab(self.merged_data, 'short', 
                                        "Medium Short Term Opportunities", "45 days")
            
            with tabs[4]:
                self.display_momentum_tab(self.merged_data, 'medium', 
                                        "Medium Term Opportunities", "3 months")
            
            with tabs[5]:
                self.display_momentum_tab(self.merged_data, 'long', 
                                        "Long Term Opportunities", "1 year")
            
            with tabs[6]:
                self.display_multibagger_tab(self.merged_data)
            
            with tabs[7]:
                self.display_turnaround_tab(self.merged_data)
            
            with tabs[8]:
                self.display_stock_deep_dive(self.merged_data)
            
            with tabs[9]:
                self.display_sector_rotation()
        
        else:
            # Welcome screen
            st.info("👈 Please upload your data files using the sidebar to get started")
            
            # Instructions
            with st.expander("📖 How to Use This Dashboard"):
                st.markdown("""
                ### Data Requirements:
                
                1. **Stock Data File**: Contains comprehensive fundamental and technical data
                2. **Master Data File**: Maps stocks to their sectors and industries
                3. **Index Data File** (Optional): For sector rotation analysis
                
                ### Features:
                
                - **Global Market Cap Filter**: Filter stocks across all tabs by market capitalization
                - **Overall Rankings**: Comprehensive ranking based on multiple factors
                - **Time-based Opportunities**: Stocks with potential for different time horizons
                - **Multibagger Identification**: High-growth potential stocks
                - **Turnaround Candidates**: Companies showing improvement signs
                - **Stock Deep Dive**: Detailed analysis of individual stocks
                - **Sector Rotation**: Visual analysis of sector performance
                
                ### Ranking Methodology:
                
                The ranking system incorporates:
                - Sector-specific valuation metrics
                - Forensic accounting checks
                - Quality and growth metrics
                - Technical indicators
                - Institutional holding patterns
                """)
            
            # Methodology explanation
            with st.expander("🔬 Ranking Methodology"):
                st.markdown("""
                ### Overall Ranking Components:
                
                1. **Sector-Specific Valuation (35%)**
                   - Uses appropriate metrics for each sector
                   - Compares stocks within their peer group
                
                2. **Quality Metrics (25%)**
                   - Return on Equity (ROE)
                   - Return on Capital Employed (ROCE)
                   - Profit margins
                   - Debt ratios
                   - Interest coverage
                
                3. **Growth Metrics (20%)**
                   - Historical revenue and earnings growth
                   - Forward growth expectations
                   - 5-year CAGR
                
                4. **Forensic Accounting (20%)**
                   - Cash flow quality
                   - Working capital efficiency
                   - Asset quality trends
                   - Debt sustainability
                
                ### Special Rankings:
                
                - **Momentum Rankings**: Based on price performance and technical indicators
                - **Multibagger Scores**: High growth + reasonable valuation + quality
                - **Turnaround Scores**: Improving fundamentals + recovery signals
                """)

# Run the dashboard
if __name__ == "__main__":
    dashboard = StockAnalysisDashboard()
    dashboard.run()