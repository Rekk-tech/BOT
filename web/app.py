import os
import json
import time
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Optional

# ================== Config ==================
DEFAULT_API_URL = "http://localhost:8000"
API_URL = os.getenv("API_URL", DEFAULT_API_URL)

# Page configuration
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .risk-high {
        background-color: #ff4444;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-weight: bold;
    }
    .risk-medium {
        background-color: #ffaa00;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-weight: bold;
    }
    .risk-low {
        background-color: #00aa44;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🚨 Advanced Fraud Detection Dashboard</h1>', unsafe_allow_html=True)

# ================== Sidebar Configuration ==================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Configuration
    api_url = st.text_input(
        "API URL", 
        value=API_URL, 
        help="FastAPI endpoint URL"
    )
    
    # Connection test
    if st.button("🔍 Test Connection"):
        try:
            with st.spinner("Testing connection..."):
                response = requests.get(f"{api_url}/health", timeout=10)
                response.raise_for_status()
                data = response.json()
                
            st.success("✅ Connection successful!")
            st.json({
                "Status": data.get("status"),
                "Threshold": data.get("threshold"),
                "Features": data.get("features_count"),
                "Version": data.get("version")
            })
            
        except requests.exceptions.ConnectionError:
            st.error("❌ Connection failed: Cannot reach API server")
        except requests.exceptions.Timeout:
            st.error("❌ Connection failed: Request timeout")
        except Exception as e:
            st.error(f"❌ Connection failed: {str(e)}")
    
    st.divider()
    
    # Model Information
    if st.button("📊 Get Model Info"):
        try:
            response = requests.get(f"{api_url}/model_info", timeout=10)
            response.raise_for_status()
            model_info = response.json()
            st.subheader("Model Information")
            st.json(model_info)
        except Exception as e:
            st.error(f"Failed to get model info: {str(e)}")
    
    st.divider()
    
    # Threshold Management
    st.subheader("🎯 Threshold Control")
    new_threshold = st.slider(
        "Fraud Detection Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.8,
        step=0.05,
        help="Probability threshold for flagging transactions as fraud"
    )
    
    if st.button("Update Threshold"):
        try:
            response = requests.post(
                f"{api_url}/update_threshold",
                params={"new_threshold": new_threshold},
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            st.success(f"✅ Threshold updated: {result['old_threshold']} → {result['new_threshold']}")
        except Exception as e:
            st.error(f"Failed to update threshold: {str(e)}")

# ================== Helper Functions ==================
TXN_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

def predict_single(payload: Dict) -> Dict:
    """Make single prediction API call"""
    url = f"{api_url}/predict"
    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()

def predict_batch(df: pd.DataFrame) -> Dict:
    """Make batch prediction API call"""
    required_cols = [
        "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrg",
        "nameDest", "oldbalanceDest", "newbalanceDest", "isFlaggedFraud"
    ]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Prepare data
    records = df[required_cols].to_dict(orient="records")
    payload = {"transactions": records}
    
    url = f"{api_url}/batch_predict"
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()

def format_risk_level(risk: str) -> str:
    """Format risk level with colored badge"""
    if risk == "HIGH":
        return f'<span class="risk-high">{risk}</span>'
    elif risk == "MEDIUM":
        return f'<span class="risk-medium">{risk}</span>'
    else:
        return f'<span class="risk-low">{risk}</span>'

def create_score_distribution_plot(scores: List[float]) -> go.Figure:
    """Create fraud score distribution plot"""
    fig = go.Figure(data=[go.Histogram(
        x=scores,
        nbinsx=30,
        name="Score Distribution",
        marker_color='rgba(55, 83, 109, 0.7)',
        marker_line_color='rgba(55, 83, 109, 1.0)',
        marker_line_width=1
    )])
    
    fig.update_layout(
        title="Fraud Score Distribution",
        xaxis_title="Fraud Score",
        yaxis_title="Frequency",
        showlegend=False,
        height=400
    )
    
    return fig

def create_risk_level_pie_chart(risk_levels: List[str]) -> go.Figure:
    """Create risk level distribution pie chart"""
    risk_counts = pd.Series(risk_levels).value_counts()
    colors = {'HIGH': '#ff4444', 'MEDIUM': '#ffaa00', 'LOW': '#00aa44'}
    
    fig = go.Figure(data=[go.Pie(
        labels=risk_counts.index,
        values=risk_counts.values,
        marker_colors=[colors.get(level, '#cccccc') for level in risk_counts.index],
        hole=0.4
    )])
    
    fig.update_layout(
        title="Risk Level Distribution",
        height=400
    )
    
    return fig

# ================== Main Tabs ==================
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Single Transaction", 
    "📦 Batch Processing", 
    "📊 Analytics Dashboard",
    "📋 Transaction Builder"
])

# ================== Tab 1: Single Transaction ==================
with tab1:
    st.subheader("Single Transaction Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.form("single_transaction_form"):
            # Transaction Details
            st.markdown("**Transaction Details**")
            tcol1, tcol2, tcol3 = st.columns(3)
            
            with tcol1:
                step = st.number_input("Time Step", min_value=0, value=500, step=1)
                transaction_type = st.selectbox("Transaction Type", TXN_TYPES, index=4)
            
            with tcol2:
                amount = st.number_input("Amount", min_value=0.0, value=200000.0, step=1000.0)
                is_flagged = st.selectbox("Initially Flagged", [0, 1], index=0)
            
            with tcol3:
                name_orig = st.text_input("Origin Account", value="C12345")
                name_dest = st.text_input("Destination Account", value="M98765")
            
            st.markdown("**Account Balances**")
            bcol1, bcol2 = st.columns(2)
            
            with bcol1:
                st.markdown("*Origin Account*")
                old_balance_org = st.number_input("Old Balance (Origin)", min_value=0.0, value=0.0, step=100.0)
                new_balance_org = st.number_input("New Balance (Origin)", min_value=0.0, value=0.0, step=100.0)
            
            with bcol2:
                st.markdown("*Destination Account*")
                old_balance_dest = st.number_input("Old Balance (Dest)", min_value=0.0, value=0.0, step=100.0)
                new_balance_dest = st.number_input("New Balance (Dest)", min_value=0.0, value=0.0, step=100.0)
            
            submitted = st.form_submit_button("🔍 Analyze Transaction", use_container_width=True)
    
    if submitted:
        payload = {
            "step": int(step),
            "type": transaction_type,
            "amount": float(amount),
            "nameOrig": name_orig or None,
            "oldbalanceOrg": float(old_balance_org),
            "newbalanceOrg": float(new_balance_org),
            "nameDest": name_dest or None,
            "oldbalanceDest": float(old_balance_dest),
            "newbalanceDest": float(new_balance_dest),
            "isFlaggedFraud": int(is_flagged),
        }
        
        try:
            with st.spinner("Analyzing transaction..."):
                result = predict_single(payload)
            
            with col2:
                st.markdown("### 📊 Analysis Results")
                
                # Main metrics
                score = result["score"]
                flagged = result["flagged"]
                risk_level = result["risk_level"]
                
                st.metric("Fraud Score", f"{score:.4f}")
                st.metric("Risk Level", risk_level)
                st.metric("Flagged", "🚨 YES" if flagged else "✅ NO")
                st.metric("Threshold", f"{result['threshold']:.3f}")
                
                # Risk gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Fraud Risk"},
                    gauge = {
                        'axis': {'range': [None, 1]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 0.3], 'color': "lightgreen"},
                            {'range': [0.3, 0.6], 'color': "yellow"},
                            {'range': [0.6, 1], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': result['threshold']
                        }
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Transaction details
            st.markdown("### 📄 Transaction Details")
            st.json(payload)
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")

# ================== Tab 2: Batch Processing ==================
with tab2:
    st.subheader("Batch Transaction Processing")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV file with transactions",
        type=["csv"],
        help="Required columns: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrg, nameDest, oldbalanceDest, newbalanceDest, isFlaggedFraud"
    )
    
    # Sample data download
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("📥 Download Sample CSV"):
            sample_data = pd.DataFrame({
                'step': [1, 2, 3],
                'type': ['TRANSFER', 'CASH_OUT', 'PAYMENT'],
                'amount': [9000.60, 181.00, 1864.28],
                'nameOrig': ['C1231006815', 'C1666544295', 'C1305486145'],
                'oldbalanceOrg': [9000.60, 181.00, 1864.28],
                'newbalanceOrg': [0.00, 0.00, 0.00],
                'nameDest': ['M1979787155', 'C553264065', 'M1144492040'],
                'oldbalanceDest': [0.00, 0.00, 0.00],
                'newbalanceDest': [0.00, 0.00, 0.00],
                'isFlaggedFraud': [0, 1, 0]
            })
            csv = sample_data.to_csv(index=False)
            st.download_button(
                "Download",
                csv,
                "sample_transactions.csv",
                "text/csv"
            )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File loaded successfully: {len(df)} transactions")
            
            # Data preview
            st.markdown("### 📊 Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Data validation
            required_cols = [
                "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrg",
                "nameDest", "oldbalanceDest", "newbalanceDest", "isFlaggedFraud"
            ]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}")
            else:
                # Data statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Transactions", len(df))
                with col2:
                    st.metric("Transaction Types", df['type'].nunique())
                with col3:
                    st.metric("Avg Amount", f"${df['amount'].mean():,.2f}")
                with col4:
                    st.metric("Initially Flagged", df['isFlaggedFraud'].sum())
                
                # Processing options
                st.markdown("### ⚙️ Processing Options")
                process_col1, process_col2 = st.columns(2)
                
                with process_col1:
                    batch_size = st.number_input("Batch Size", min_value=10, max_value=1000, value=100)
                    show_warnings = st.checkbox("Show Validation Warnings", value=True)
                
                with process_col2:
                    auto_download = st.checkbox("Auto-download Results", value=True)
                    detailed_analysis = st.checkbox("Include Detailed Analysis", value=True)
                
                # Process button
                if st.button("🚀 Process All Transactions", use_container_width=True):
                    try:
                        with st.spinner(f"Processing {len(df)} transactions..."):
                            # Process in batches if large dataset
                            if len(df) <= batch_size:
                                result = predict_batch(df)
                            else:
                                st.info(f"Processing in batches of {batch_size}...")
                                results = []
                                progress_bar = st.progress(0)
                                
                                for i in range(0, len(df), batch_size):
                                    batch_df = df.iloc[i:i+batch_size]
                                    batch_result = predict_batch(batch_df)
                                    results.extend(batch_result['results'])
                                    progress_bar.progress((i + batch_size) / len(df))
                                
                                # Combine results
                                result = {
                                    'results': results,
                                    'summary': {
                                        'total_transactions': len(df),
                                        'flagged_count': sum(1 for r in results if r['flagged']),
                                        'flagged_percentage': sum(1 for r in results if r['flagged']) / len(results) * 100,
                                        'avg_score': sum(r['score'] for r in results) / len(results),
                                        'warnings': []
                                    }
                                }
                        
                        # Display results
                        st.success(f"✅ Processing complete!")
                        
                        # Summary metrics
                        summary = result['summary']
                        st.markdown("### 📈 Results Summary")
                        
                        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                        with metric_col1:
                            st.metric("Processed", summary['total_transactions'])
                        with metric_col2:
                            st.metric("Flagged as Fraud", summary['flagged_count'])
                        with metric_col3:
                            st.metric("Fraud Rate", f"{summary['flagged_percentage']:.1f}%")
                        with metric_col4:
                            st.metric("Avg Score", f"{summary['avg_score']:.3f}")
                        
                        # Create results dataframe
                        results_df = df.copy()
                        results_df['fraud_score'] = [r['score'] for r in result['results']]
                        results_df['fraud_flagged'] = [r['flagged'] for r in result['results']]
                        results_df['risk_level'] = [r['risk_level'] for r in result['results']]
                        
                        if detailed_analysis:
                            # Visualizations
                            st.markdown("### 📊 Analysis Visualizations")
                            
                            viz_col1, viz_col2 = st.columns(2)
                            
                            with viz_col1:
                                scores = [r['score'] for r in result['results']]
                                fig_dist = create_score_distribution_plot(scores)
                                st.plotly_chart(fig_dist, use_container_width=True)
                            
                            with viz_col2:
                                risk_levels = [r['risk_level'] for r in result['results']]
                                fig_pie = create_risk_level_pie_chart(risk_levels)
                                st.plotly_chart(fig_pie, use_container_width=True)
                            
                            # Transaction type analysis
                            type_analysis = results_df.groupby('type').agg({
                                'fraud_flagged': ['count', 'sum', 'mean']
                            }).round(3)
                            type_analysis.columns = ['Total', 'Flagged', 'Fraud_Rate']
                            
                            st.markdown("### 📋 Transaction Type Analysis")
                            st.dataframe(type_analysis, use_container_width=True)
                        
                        # Results display options
                        st.markdown("### 🔍 Results Explorer")
                        
                        filter_col1, filter_col2, filter_col3 = st.columns(3)
                        with filter_col1:
                            show_flagged_only = st.checkbox("Show Only Flagged", value=False)
                        with filter_col2:
                            risk_filter = st.selectbox("Risk Level Filter", 
                                                     ['All', 'HIGH', 'MEDIUM', 'LOW'], 
                                                     index=0)
                        with filter_col3:
                            min_score = st.slider("Minimum Score", 0.0, 1.0, 0.0, 0.1)
                        
                        # Apply filters
                        filtered_df = results_df.copy()
                        if show_flagged_only:
                            filtered_df = filtered_df[filtered_df['fraud_flagged'] == True]
                        if risk_filter != 'All':
                            filtered_df = filtered_df[filtered_df['risk_level'] == risk_filter]
                        if min_score > 0:
                            filtered_df = filtered_df[filtered_df['fraud_score'] >= min_score]
                        
                        st.markdown(f"**Showing {len(filtered_df)} of {len(results_df)} transactions**")
                        
                        # Enhanced results display
                        display_df = filtered_df.copy()
                        display_df['fraud_score'] = display_df['fraud_score'].round(4)
                        display_df['risk_level_formatted'] = display_df['risk_level'].apply(
                            lambda x: format_risk_level(x)
                        )
                        
                        st.dataframe(
                            display_df.drop('risk_level', axis=1),
                            use_container_width=True,
                            column_config={
                                "fraud_score": st.column_config.ProgressColumn(
                                    "Fraud Score",
                                    help="Probability of fraud (0-1)",
                                    min_value=0,
                                    max_value=1,
                                ),
                                "fraud_flagged": st.column_config.CheckboxColumn(
                                    "Flagged",
                                    help="Flagged as potential fraud",
                                ),
                                "risk_level_formatted": st.column_config.Column(
                                    "Risk Level",
                                    help="Risk assessment level",
                                )
                            }
                        )
                        
                        # Download options
                        st.markdown("### 💾 Download Results")
                        download_col1, download_col2 = st.columns(2)
                        
                        with download_col1:
                            # Full results
                            csv_data = results_df.to_csv(index=False)
                            st.download_button(
                                "📊 Download Full Results",
                                csv_data,
                                f"fraud_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                "text/csv",
                                use_container_width=True
                            )
                        
                        with download_col2:
                            # Flagged only
                            flagged_df = results_df[results_df['fraud_flagged'] == True]
                            if len(flagged_df) > 0:
                                flagged_csv = flagged_df.to_csv(index=False)
                                st.download_button(
                                    "🚨 Download Flagged Only",
                                    flagged_csv,
                                    f"flagged_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    "text/csv",
                                    use_container_width=True
                                )
                            else:
                                st.info("No flagged transactions to download")
                        
                        # Show warnings if any
                        if show_warnings and summary.get('warnings'):
                            st.warning("⚠️ Validation Warnings:")
                            for warning in summary['warnings']:
                                st.text(f"• {warning}")
                                
                    except Exception as e:
                        st.error(f"❌ Processing failed: {str(e)}")
                        st.exception(e)
        
        except Exception as e:
            st.error(f"❌ File processing failed: {str(e)}")

# ================== Tab 3: Analytics Dashboard ==================
with tab3:
    st.subheader("📊 System Analytics Dashboard")
    
    # Mock analytics data (in real implementation, this would come from logs/database)
    st.info("📝 Note: This is a demo analytics dashboard. In production, connect to your logging system.")
    
    # Generate sample analytics data
    if st.button("🔄 Generate Sample Analytics"):
        # Sample data for demonstration
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        np.random.seed(42)
        
        daily_stats = pd.DataFrame({
            'date': dates,
            'total_predictions': np.random.poisson(100, len(dates)),
            'fraud_detected': np.random.poisson(15, len(dates)),
            'avg_score': np.random.beta(2, 5, len(dates)),
            'api_response_time': np.random.gamma(2, 0.1, len(dates))
        })
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Predictions", f"{daily_stats['total_predictions'].sum():,}")
        with col2:
            st.metric("Fraud Detected", f"{daily_stats['fraud_detected'].sum():,}")
        with col3:
            avg_fraud_rate = daily_stats['fraud_detected'].sum() / daily_stats['total_predictions'].sum()
            st.metric("Avg Fraud Rate", f"{avg_fraud_rate:.1%}")
        with col4:
            st.metric("Avg Response Time", f"{daily_stats['api_response_time'].mean():.2f}s")
        
        # Charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Daily predictions trend
            fig_trend = px.line(
                daily_stats, 
                x='date', 
                y=['total_predictions', 'fraud_detected'],
                title="Daily Predictions Trend",
                labels={'value': 'Count', 'date': 'Date'}
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        
        with chart_col2:
            # Fraud rate over time
            daily_stats['fraud_rate'] = daily_stats['fraud_detected'] / daily_stats['total_predictions']
            fig_rate = px.line(
                daily_stats,
                x='date',
                y='fraud_rate',
                title="Fraud Detection Rate Over Time",
                labels={'fraud_rate': 'Fraud Rate', 'date': 'Date'}
            )
            st.plotly_chart(fig_rate, use_container_width=True)
        
        # Performance metrics
        st.markdown("### ⚡ Performance Metrics")
        perf_col1, perf_col2 = st.columns(2)
        
        with perf_col1:
            fig_response = px.histogram(
                daily_stats,
                x='api_response_time',
                title="API Response Time Distribution",
                labels={'api_response_time': 'Response Time (seconds)', 'count': 'Frequency'}
            )
            st.plotly_chart(fig_response, use_container_width=True)
        
        with perf_col2:
            fig_score_dist = px.histogram(
                daily_stats,
                x='avg_score',
                title="Average Daily Score Distribution",
                labels={'avg_score': 'Average Score', 'count': 'Frequency'}
            )
            st.plotly_chart(fig_score_dist, use_container_width=True)

# ================== Tab 4: Transaction Builder ==================
with tab4:
    st.subheader("📋 Advanced Transaction Builder")
    st.markdown("Build and test complex transaction scenarios")
    
    # Scenario templates
    st.markdown("### 🎯 Quick Scenarios")
    scenario_col1, scenario_col2, scenario_col3 = st.columns(3)
    
    scenarios = {
        "Normal Transfer": {
            "step": 1,
            "type": "TRANSFER",
            "amount": 5000.0,
            "oldbalanceOrg": 10000.0,
            "newbalanceOrg": 5000.0,
            "oldbalanceDest": 2000.0,
            "newbalanceDest": 7000.0,
            "isFlaggedFraud": 0
        },
        "Suspicious Large Transfer": {
            "step": 100,
            "type": "TRANSFER",
            "amount": 500000.0,
            "oldbalanceOrg": 500000.0,
            "newbalanceOrg": 0.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "isFlaggedFraud": 0
        },
        "Cash Out Anomaly": {
            "step": 50,
            "type": "CASH_OUT",
            "amount": 100000.0,
            "oldbalanceOrg": 100000.0,
            "newbalanceOrg": 0.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 100000.0,
            "isFlaggedFraud": 0
        }
    }
    
    selected_scenario = None
    with scenario_col1:
        if st.button("📈 Load Normal Transfer", use_container_width=True):
            selected_scenario = "Normal Transfer"
    with scenario_col2:
        if st.button("⚠️ Load Suspicious Transfer", use_container_width=True):
            selected_scenario = "Suspicious Large Transfer"
    with scenario_col3:
        if st.button("🚨 Load Cash Out Anomaly", use_container_width=True):
            selected_scenario = "Cash Out Anomaly"
    
    # Transaction builder form
    st.markdown("### 🔧 Custom Transaction Builder")
    
    # Initialize session state
    if 'transactions' not in st.session_state:
        st.session_state.transactions = []
    
    with st.form("transaction_builder"):
        if selected_scenario:
            scenario_data = scenarios[selected_scenario]
            st.info(f"Loaded scenario: {selected_scenario}")
        else:
            scenario_data = {}
        
        builder_col1, builder_col2 = st.columns(2)
        
        with builder_col1:
            step = st.number_input("Step", value=scenario_data.get("step", 1))
            txn_type = st.selectbox("Type", TXN_TYPES, 
                                   index=TXN_TYPES.index(scenario_data.get("type", "TRANSFER")))
            amount = st.number_input("Amount", value=scenario_data.get("amount", 1000.0))
            old_org = st.number_input("Old Balance Org", value=scenario_data.get("oldbalanceOrg", 1000.0))
            new_org = st.number_input("New Balance Org", value=scenario_data.get("newbalanceOrg", 0.0))
        
        with builder_col2:
            name_orig = st.text_input("Name Orig", value="C12345")
            name_dest = st.text_input("Name Dest", value="M98765")
            old_dest = st.number_input("Old Balance Dest", value=scenario_data.get("oldbalanceDest", 0.0))
            new_dest = st.number_input("New Balance Dest", value=scenario_data.get("newbalanceDest", 1000.0))
            is_flagged = st.selectbox("Is Flagged", [0, 1], 
                                     index=scenario_data.get("isFlaggedFraud", 0))
        
        add_to_batch = st.checkbox("Add to batch for testing")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            submit_single = st.form_submit_button("🔍 Test Single", use_container_width=True)
        with col2:
            add_to_list = st.form_submit_button("➕ Add to List", use_container_width=True)
        with col3:
            clear_list = st.form_submit_button("🗑️ Clear List", use_container_width=True)
    
    # Handle form submissions
    transaction_data = {
        "step": int(step),
        "type": txn_type,
        "amount": float(amount),
        "nameOrig": name_orig,
        "oldbalanceOrg": float(old_org),
        "newbalanceOrg": float(new_org),
        "nameDest": name_dest,
        "oldbalanceDest": float(old_dest),
        "newbalanceDest": float(new_dest),
        "isFlaggedFraud": int(is_flagged)
    }
    
    if submit_single:
        try:
            with st.spinner("Testing transaction..."):
                result = predict_single(transaction_data)
            
            st.success("✅ Analysis Complete!")
            
            result_col1, result_col2, result_col3, result_col4 = st.columns(4)
            with result_col1:
                st.metric("Score", f"{result['score']:.4f}")
            with result_col2:
                st.metric("Flagged", "🚨 YES" if result['flagged'] else "✅ NO")
            with result_col3:
                st.metric("Risk", result['risk_level'])
            with result_col4:
                st.metric("Threshold", f"{result['threshold']:.3f}")
            
        except Exception as e:
            st.error(f"❌ Test failed: {str(e)}")
    
    if add_to_list:
        st.session_state.transactions.append(transaction_data.copy())
        st.success(f"✅ Added transaction to list (Total: {len(st.session_state.transactions)})")
    
    if clear_list:
        st.session_state.transactions = []
        st.success("✅ Transaction list cleared!")
    
    # Display transaction list
    if st.session_state.transactions:
        st.markdown("### 📋 Transaction List for Batch Testing")
        
        transactions_df = pd.DataFrame(st.session_state.transactions)
        st.dataframe(transactions_df, use_container_width=True)
        
        if st.button("🚀 Test All Transactions", use_container_width=True):
            try:
                with st.spinner(f"Testing {len(st.session_state.transactions)} transactions..."):
                    result = predict_batch(transactions_df)
                
                st.success("✅ Batch testing complete!")
                
                # Display results
                results_list = result['results']
                summary = result['summary']
                
                # Summary metrics
                batch_col1, batch_col2, batch_col3, batch_col4 = st.columns(4)
                with batch_col1:
                    st.metric("Total", summary['total_transactions'])
                with batch_col2:
                    st.metric("Flagged", summary['flagged_count'])
                with batch_col3:
                    st.metric("Fraud Rate", f"{summary['flagged_percentage']:.1f}%")
                with batch_col4:
                    st.metric("Avg Score", f"{summary['avg_score']:.3f}")
                
                # Detailed results
                results_df = transactions_df.copy()
                results_df['fraud_score'] = [r['score'] for r in results_list]
                results_df['fraud_flagged'] = [r['flagged'] for r in results_list]
                results_df['risk_level'] = [r['risk_level'] for r in results_list]
                
                st.dataframe(results_df, use_container_width=True)
                
                # Download results
                csv_results = results_df.to_csv(index=False)
                st.download_button(
                    "💾 Download Results",
                    csv_results,
                    f"batch_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"❌ Batch testing failed: {str(e)}")

# ================== Footer ==================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🚨 <strong>Advanced Fraud Detection System</strong> 🚨</p>
        <p>Built with Streamlit • Powered by Machine Learning • Version 2.0</p>
        <p><em>For support, contact your system administrator</em></p>
    </div>
    """,
    unsafe_allow_html=True
)