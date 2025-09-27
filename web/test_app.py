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
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with modern design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Custom Header */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem 0;
        border-bottom: 2px solid #f0f2f6;
    }
    
    /* Enhanced Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.4);
    }
    
    /* Status Indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
        transition: all 0.2s ease;
    }
    
    .status-online {
        background: linear-gradient(135deg, #4ade80 0%, #22c55e 100%);
        color: white;
        box-shadow: 0 4px 12px rgba(74, 222, 128, 0.3);
    }
    
    .status-offline {
        background: linear-gradient(135deg, #f87171 0%, #ef4444 100%);
        color: white;
        box-shadow: 0 4px 12px rgba(248, 113, 113, 0.3);
    }
    
    /* Risk Level Badges */
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(255, 107, 107, 0.3);
        animation: pulse 2s infinite;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #feca57 0%, #ff9ff3 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(254, 202, 87, 0.3);
    }
    
    .risk-low {
        background: linear-gradient(135deg, #48cab2 0%, #2dd4bf 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(72, 202, 178, 0.3);
    }
    
    /* Animated Elements */
    @keyframes pulse {
        0% { box-shadow: 0 4px 12px rgba(255, 107, 107, 0.3); }
        50% { box-shadow: 0 8px 20px rgba(255, 107, 107, 0.5); }
        100% { box-shadow: 0 4px 12px rgba(255, 107, 107, 0.3); }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Card Containers */
    .card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(0, 0, 0, 0.05);
        animation: fadeInUp 0.6s ease-out;
    }
    
    .dark-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
        color: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(30, 58, 138, 0.3);
    }
    
    /* Sidebar Enhancements */
    .sidebar .stSelectbox > div > div {
        background-color: #f8fafc;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
    }
    
    /* Button Enhancements */
    .stButton > button {
        border-radius: 10px;
        border: none;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Form Enhancements */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select,
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem;
        transition: all 0.3s ease;
    }
    
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Success/Error Messages */
    .element-container .stAlert {
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Tabs Enhancement */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f8fafc;
        border-radius: 10px;
        color: #64748b;
        font-weight: 500;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    /* Loading Spinner */
    .stSpinner > div {
        border-color: #667eea transparent transparent transparent;
    }
    
    /* Dataframe Styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 2px solid #f1f5f9;
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 15px;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        .metric-card {
            padding: 1rem;
        }
        
        .card, .dark-card {
            padding: 1rem;
        }
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
    }
</style>
""", unsafe_allow_html=True)

# Enhanced header with status indicator
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<h1 class="main-header">🛡️ Advanced Fraud Detection System</h1>', unsafe_allow_html=True)

# Quick status check
try:
    response = requests.get(f"{API_URL}/health", timeout=5)
    if response.status_code == 200:
        status_html = '<div class="status-indicator status-online">🟢 System Online</div>'
    else:
        status_html = '<div class="status-indicator status-offline">🔴 System Offline</div>'
except:
    status_html = '<div class="status-indicator status-offline">🔴 Connection Failed</div>'

st.markdown(f'<div style="text-align: center; margin-bottom: 2rem;">{status_html}</div>', unsafe_allow_html=True)

# ================== Enhanced Sidebar ==================
with st.sidebar:
    st.markdown("### ⚙️ System Configuration")
    
    # API Configuration with enhanced styling
    api_url = st.text_input(
        "🔗 API Endpoint", 
        value=API_URL,
        help="FastAPI server endpoint URL",
        placeholder="http://localhost:8000"
    )
    
    # Enhanced connection test
    if st.button("🔍 Test Connection", use_container_width=True):
        try:
            with st.spinner("Testing connection..."):
                response = requests.get(f"{api_url}/health", timeout=10)
                response.raise_for_status()
                data = response.json()
                
            st.balloons()
            st.success("✅ Connection established successfully!")
            
            # Enhanced info display
            with st.container():
                st.markdown("**System Information**")
                info_cols = st.columns(2)
                with info_cols[0]:
                    st.metric("Status", data.get("status", "Unknown"))
                    st.metric("Features", data.get("features_count", "N/A"))
                with info_cols[1]:
                    st.metric("Threshold", f"{data.get('threshold', 0):.3f}")
                    st.metric("Version", data.get("version", "N/A"))
                    
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot reach API server")
            st.info("💡 Make sure the API server is running")
        except requests.exceptions.Timeout:
            st.error("❌ Request timeout")
            st.info("💡 Server might be overloaded")
        except Exception as e:
            st.error(f"❌ Connection failed: {str(e)}")
    
    st.divider()
    
    # Enhanced Model Information
    with st.expander("📊 Model Information", expanded=False):
        if st.button("🔄 Refresh Model Info"):
            try:
                response = requests.get(f"{api_url}/model_info", timeout=10)
                response.raise_for_status()
                model_info = response.json()
                
                # Create a nice display for model info
                if "model_type" in model_info:
                    st.metric("Model Type", model_info["model_type"])
                if "accuracy" in model_info:
                    st.metric("Accuracy", f"{model_info['accuracy']:.2%}")
                if "last_trained" in model_info:
                    st.metric("Last Trained", model_info["last_trained"])
                    
                with st.expander("Full Details"):
                    st.json(model_info)
                    
            except Exception as e:
                st.error(f"Failed to get model info: {str(e)}")
    
    st.divider()
    
    # Enhanced Threshold Management
    st.markdown("### 🎯 Detection Threshold")
    
    # Current threshold display
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            current_threshold = response.json().get("threshold", 0.8)
        else:
            current_threshold = 0.8
    except:
        current_threshold = 0.8
    
    st.info(f"Current: {current_threshold:.3f}")
    
    new_threshold = st.slider(
        "New Threshold",
        min_value=0.0,
        max_value=1.0,
        value=current_threshold,
        step=0.01,
        help="Lower = more sensitive (more flags), Higher = less sensitive (fewer flags)"
    )
    
    # Threshold impact indicator
    if new_threshold < 0.5:
        st.warning("⚠️ High sensitivity - Many transactions will be flagged")
    elif new_threshold > 0.9:
        st.warning("⚠️ Low sensitivity - Only obvious fraud will be flagged")
    else:
        st.success("✅ Balanced threshold setting")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📈 Update", use_container_width=True):
            try:
                response = requests.post(
                    f"{api_url}/update_threshold",
                    params={"new_threshold": new_threshold},
                    timeout=10
                )
                response.raise_for_status()
                result = response.json()
                st.success(f"✅ Updated: {result['old_threshold']:.3f} → {result['new_threshold']:.3f}")
                st.rerun()
            except Exception as e:
                st.error(f"Update failed: {str(e)}")
    
    with col2:
        if st.button("🔄 Reset", use_container_width=True):
            st.info("Reset to default: 0.800")
    
    st.divider()
    
    # System Statistics (Mock data)
    st.markdown("### 📈 Quick Stats")
    stats_data = {
        "Today's Predictions": np.random.randint(450, 600),
        "Fraud Detected": np.random.randint(15, 35),
        "Avg Response Time": f"{np.random.uniform(0.1, 0.3):.2f}s",
        "System Uptime": "99.9%"
    }
    
    for label, value in stats_data.items():
        st.metric(label, value)

# ================== Helper Functions (Enhanced) ==================
TXN_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

def predict_single(payload: Dict) -> Dict:
    """Make single prediction API call with enhanced error handling"""
    try:
        url = f"{api_url}/predict"
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        raise Exception("Request timeout - server may be overloaded")
    except requests.exceptions.ConnectionError:
        raise Exception("Cannot connect to API server")
    except requests.exceptions.HTTPError as e:
        raise Exception(f"API error: {e.response.status_code}")

def predict_batch(df: pd.DataFrame) -> Dict:
    """Make batch prediction API call with progress tracking"""
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
    """Format risk level with enhanced colored badge"""
    if risk == "HIGH":
        return f'<span class="risk-high">🚨 {risk}</span>'
    elif risk == "MEDIUM":
        return f'<span class="risk-medium">⚠️ {risk}</span>'
    else:
        return f'<span class="risk-low">✅ {risk}</span>'

def create_enhanced_gauge(score: float, threshold: float, title: str = "Fraud Risk") -> go.Figure:
    """Create enhanced fraud risk gauge"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 20}},
        delta = {'reference': threshold, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue", 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 0.3], 'color': 'rgba(72, 202, 178, 0.3)'},
                {'range': [0.3, 0.6], 'color': 'rgba(254, 202, 87, 0.3)'},
                {'range': [0.6, 1], 'color': 'rgba(255, 107, 107, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold
            }
        }
    ))
    
    fig.update_layout(
        height=350,
        font={'color': "darkblue", 'family': "Arial"},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_score_distribution_plot(scores: List[float]) -> go.Figure:
    """Create enhanced fraud score distribution plot"""
    fig = go.Figure()
    
    # Add histogram
    fig.add_trace(go.Histogram(
        x=scores,
        nbinsx=30,
        name="Score Distribution",
        marker_color='rgba(102, 126, 234, 0.7)',
        marker_line_color='rgba(102, 126, 234, 1.0)',
        marker_line_width=2,
        opacity=0.8
    ))
    
    # Add mean line
    mean_score = np.mean(scores)
    fig.add_vline(x=mean_score, line_dash="dash", line_color="red", 
                  annotation_text=f"Mean: {mean_score:.3f}")
    
    fig.update_layout(
        title={
            'text': "Fraud Score Distribution",
            'x': 0.5,
            'font': {'size': 18}
        },
        xaxis_title="Fraud Score",
        yaxis_title="Frequency",
        showlegend=False,
        height=400,
        plot_bgcolor='rgba(248, 250, 252, 0.8)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter'}
    )
    
    return fig

def create_risk_level_pie_chart(risk_levels: List[str]) -> go.Figure:
    """Create enhanced risk level distribution pie chart"""
    risk_counts = pd.Series(risk_levels).value_counts()
    colors = {
        'HIGH': 'rgba(255, 107, 107, 0.8)',
        'MEDIUM': 'rgba(254, 202, 87, 0.8)',
        'LOW': 'rgba(72, 202, 178, 0.8)'
    }
    
    fig = go.Figure(data=[go.Pie(
        labels=risk_counts.index,
        values=risk_counts.values,
        marker_colors=[colors.get(level, '#cccccc') for level in risk_counts.index],
        hole=0.5,
        textinfo='label+percent+value',
        textfont={'size': 12, 'family': 'Inter'},
        marker_line_color='white',
        marker_line_width=2
    )])
    
    fig.update_layout(
        title={
            'text': "Risk Level Distribution",
            'x': 0.5,
            'font': {'size': 18}
        },
        height=400,
        font={'family': 'Inter'},
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# ================== Enhanced Main Tabs ==================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Single Analysis", 
    "📦 Batch Processing", 
    "📊 Analytics Dashboard",
    "📋 Transaction Builder",
    "⚡ Real-time Monitor"
])

# ================== Tab 1: Enhanced Single Transaction ==================
with tab1:
    st.markdown("### 🔍 Single Transaction Analysis")
    st.markdown("Analyze individual transactions for fraud risk assessment")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Enhanced form with better organization
        with st.container():
            st.markdown("#### 📝 Transaction Details")
            
            with st.form("single_transaction_form", clear_on_submit=False):
                # Basic info
                info_col1, info_col2, info_col3 = st.columns(3)
                
                with info_col1:
                    step = st.number_input("⏰ Time Step", 
                                         min_value=0, 
                                         value=500, 
                                         step=1,
                                         help="Transaction sequence number")
                    
                with info_col2:
                    transaction_type = st.selectbox("📊 Transaction Type", 
                                                  TXN_TYPES, 
                                                  index=4,
                                                  help="Type of financial transaction")
                    
                with info_col3:
                    amount = st.number_input("💰 Amount", 
                                           min_value=0.0, 
                                           value=200000.0, 
                                           step=1000.0,
                                           help="Transaction amount")
                
                # Account information
                st.markdown("#### 👤 Account Information")
                account_col1, account_col2 = st.columns(2)
                
                with account_col1:
                    name_orig = st.text_input("🏦 Origin Account ID", 
                                            value="C12345",
                                            help="Source account identifier")
                    
                with account_col2:
                    name_dest = st.text_input("🎯 Destination Account ID", 
                                            value="M98765",
                                            help="Target account identifier")
                
                # Balance information
                st.markdown("#### 💳 Balance Information")
                balance_col1, balance_col2 = st.columns(2)
                
                with balance_col1:
                    st.markdown("**Origin Account Balances**")
                    old_balance_org = st.number_input("Before Transaction", 
                                                    min_value=0.0, 
                                                    value=0.0, 
                                                    step=100.0,
                                                    key="old_orig")
                    new_balance_org = st.number_input("After Transaction", 
                                                    min_value=0.0, 
                                                    value=0.0, 
                                                    step=100.0,
                                                    key="new_orig")
                
                with balance_col2:
                    st.markdown("**Destination Account Balances**")
                    old_balance_dest = st.number_input("Before Transaction", 
                                                     min_value=0.0, 
                                                     value=0.0, 
                                                     step=100.0,
                                                     key="old_dest")
                    new_balance_dest = st.number_input("After Transaction", 
                                                     min_value=0.0, 
                                                     value=0.0, 
                                                     step=100.0,
                                                     key="new_dest")
                
                # Additional flags
                st.markdown("#### 🚩 Additional Information")
                is_flagged = st.selectbox("Initially Flagged by System", 
                                        [0, 1], 
                                        index=0,
                                        format_func=lambda x: "Yes" if x else "No",
                                        help="Whether this transaction was flagged by basic rules")
                
                submitted = st.form_submit_button("🔍 Analyze Transaction", 
                                                use_container_width=True,
                                                type="primary")
    
    # Results panel
    with col2:
        st.markdown("### 📊 Analysis Results")
        
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
                with st.spinner("🔄 Analyzing transaction..."):
                    result = predict_single(payload)
                
                # Enhanced results display
                score = result["score"]
                flagged = result["flagged"]
                risk_level = result["risk_level"]
                threshold = result["threshold"]
                
                # Key metrics with enhanced styling
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("🎯 Fraud Score", 
                            f"{score:.4f}",
                            delta=f"{score-threshold:.4f}",
                            delta_color="inverse")
                with metric_col2:
                    st.metric("📊 Risk Level", risk_level)
                
                # Status indicator
                if flagged:
                    st.markdown('<div class="status-indicator" style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); color: white;">🚨 FRAUD DETECTED</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="status-indicator status-online">✅ TRANSACTION OK</div>', unsafe_allow_html=True)
                
                # Enhanced gauge
                fig_gauge = create_enhanced_gauge(score, threshold)
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Confidence and recommendation
                st.markdown("#### 🎯 AI Confidence")
                confidence = abs(score - threshold) * 100
                st.progress(min(confidence / 50, 1.0))
                st.caption(f"Confidence: {confidence:.1f}%")
                
                # Recommendations
                st.markdown("#### 💡 Recommendations")
                if flagged:
                    if risk_level == "HIGH":
                        st.error("🚨 **Immediate Action Required**\n- Block transaction\n- Investigate accounts\n- Contact customer")
                    elif risk_level == "MEDIUM":
                        st.warning("⚠️ **Review Required**\n- Manual review\n- Additional verification\n- Monitor pattern")
                    else:
                        st.info("ℹ️ **Low Risk Flag**\n- Standard processing\n- Log for analysis")
                else:
                    st.success("✅ **Transaction Approved**\n- Process normally\n- Standard monitoring")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.info("💡 Check API connection and try again")
        else:
            # Placeholder content
            st.info("👆 Fill out the form and click 'Analyze Transaction' to see results")
            
            # Sample metrics placeholders
            st.metric("🎯 Fraud Score", "0.0000")
            st.metric("📊 Risk Level", "UNKNOWN")
            
            # Empty gauge placeholder
            fig_empty = go.Figure(go.Indicator(
                mode = "gauge",
                value = 0,
                title = {'text': "Fraud Risk"},
                gauge = {'axis': {'range': [None, 1]}}
            ))
            fig_empty.update_layout(height=350)
            st.plotly_chart(fig_empty, use_container_width=True)

# ================== Tab 2: Enhanced Batch Processing ==================
with tab2:
    st.markdown("### 📦 Batch Transaction Processing")
    st.markdown("Process multiple transactions simultaneously for efficient fraud detection")
    
    # Enhanced file upload section
    upload_col1, upload_col2 = st.columns([2, 1])
    
    with upload_col1:
        uploaded_file = st.file_uploader(
            "📁 Upload CSV file with transactions",
            type=["csv"],
            help="Upload a CSV file containing transaction data for batch analysis"
        )
        
        # File format help
        with st.expander("📋 Required CSV Format", expanded=False):
            st.markdown("""
            **Required Columns:**
            - `step`: Transaction sequence number (integer)
            - `type`: Transaction type (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER)
            - `amount`: Transaction amount (float)
            - `nameOrig`: Origin account ID (string)
            - `oldbalanceOrg`: Origin account balance before (float)
            - `newbalanceOrg`: Origin account balance after (float)
            - `nameDest`: Destination account ID (string)
            - `oldbalanceDest`: Destination account balance before (float)
            - `newbalanceDest`: Destination account balance after (float)
            - `isFlaggedFraud`: Initial fraud flag (0 or 1)
            """)
    
    with upload_col2:
        st.markdown("#### 📥 Sample Data")
        if st.button("📄 Download Sample CSV", use_container_width=True):
            sample_data = pd.DataFrame({
                'step': [1, 2, 3, 4, 5],
                'type': ['TRANSFER', 'CASH_OUT', 'PAYMENT', 'CASH_IN', 'DEBIT'],
                'amount': [9000.60, 181.00, 1864.28, 5000.00, 250.75],
                'nameOrig': ['C1231006815', 'C1666544295', 'C1305486145', 'C1234567890', 'C9876543210'],
                'oldbalanceOrg': [9000.60, 181.00, 1864.28, 0.00, 1000.00],
                'newbalanceOrg': [0.00, 0.00, 0.00, 5000.00, 749.25],
                'nameDest': ['M1979787155', 'C553264065', 'M1144492040', 'M5555555555', 'M1111111111'],
                'oldbalanceDest': [0.00, 0.00, 0.00, 1000.00, 500.00],
                'newbalanceDest': [0.00, 0.00, 0.00, 6000.00, 750.75],
                'isFlaggedFraud': [0, 1, 0, 0, 0]
            })
            csv = sample_data.to_csv(index=False)
            st.download_button(
                "💾 Download",
                csv,
                "sample_transactions.csv",
                "text/csv",
                use_container_width=True
            )
        
        st.markdown("#### 🎯 Quick Actions")
        if st.button("🔄 Clear All Data", use_container_width=True, type="secondary"):
            st.rerun()
    
    if uploaded_file is not None:
        try:
            # Load and validate data
            df = pd.read_csv(uploaded_file)
            file_size = len(df)
            
            # Success message with animation
            st.success(f"✅ File loaded successfully!")
            
            # Enhanced data overview
            st.markdown("#### 📊 Data Overview")
            overview_col1, overview_col2, overview_col3, overview_col4, overview_col5 = st.columns(5)
            
            with overview_col1:
                st.metric("📄 Total Records", f"{file_size:,}")
            with overview_col2:
                st.metric("💱 Transaction Types", df['type'].nunique())
            with overview_col3:
                st.metric("💰 Total Value", f"${df['amount'].sum():,.2f}")
            with overview_col4:
                st.metric("📊 Avg Amount", f"${df['amount'].mean():,.2f}")
            with overview_col5:
                st.metric("🚩 Pre-flagged", df['isFlaggedFraud'].sum())
            
            # Data preview with enhanced styling
            st.markdown("#### 👀 Data Preview")
            preview_tab1, preview_tab2, preview_tab3 = st.tabs(["📋 Sample Data", "📈 Summary Stats", "🔍 Data Quality"])
            
            with preview_tab1:
                st.dataframe(
                    df.head(10), 
                    use_container_width=True,
                    column_config={
                        "amount": st.column_config.NumberColumn(
                            "Amount",
                            format="$%.2f"
                        ),
                        "type": st.column_config.SelectboxColumn(
                            "Type",
                            options=TXN_TYPES
                        )
                    }
                )
            
            with preview_tab2:
                stats_df = df.describe()
                st.dataframe(stats_df, use_container_width=True)
                
            with preview_tab3:
                # Data quality checks
                quality_col1, quality_col2 = st.columns(2)
                
                with quality_col1:
                    st.markdown("**Missing Values**")
                    missing_data = df.isnull().sum()
                    missing_df = pd.DataFrame({
                        'Column': missing_data.index,
                        'Missing Count': missing_data.values,
                        'Missing %': (missing_data.values / len(df) * 100).round(2)
                    })
                    st.dataframe(missing_df[missing_df['Missing Count'] > 0])
                
                with quality_col2:
                    st.markdown("**Data Types**")
                    dtypes_df = pd.DataFrame({
                        'Column': df.dtypes.index,
                        'Data Type': df.dtypes.values.astype(str)
                    })
                    st.dataframe(dtypes_df)
            
            # Validation
            required_cols = [
                "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrg",
                "nameDest", "oldbalanceDest", "newbalanceDest", "isFlaggedFraud"
            ]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                st.info("💡 Please ensure your CSV file contains all required columns")
            else:
                # Processing configuration
                st.markdown("#### ⚙️ Processing Configuration")
                config_col1, config_col2, config_col3 = st.columns(3)
                
                with config_col1:
                    batch_size = st.number_input(
                        "📦 Batch Size", 
                        min_value=10, 
                        max_value=min(1000, len(df)), 
                        value=min(100, len(df)),
                        help="Number of transactions to process at once"
                    )
                    
                with config_col2:
                    show_progress = st.checkbox("📊 Show Progress", value=True)
                    generate_report = st.checkbox("📋 Generate Report", value=True)
                    
                with config_col3:
                    auto_download = st.checkbox("💾 Auto-download Results", value=True)
                    include_visualization = st.checkbox("📈 Include Charts", value=True)
                
                # Processing button with enhanced styling
                st.markdown("---")
                process_col1, process_col2, process_col3 = st.columns([1, 2, 1])
                
                with process_col2:
                    if st.button("🚀 Start Batch Processing", use_container_width=True, type="primary"):
                        # Processing logic
                        try:
                            start_time = time.time()
                            
                            with st.spinner("🔄 Processing transactions..."):
                                if show_progress:
                                    progress_bar = st.progress(0)
                                    status_text = st.empty()
                                
                                # Process in batches
                                if len(df) <= batch_size:
                                    if show_progress:
                                        status_text.text("Processing single batch...")
                                    result = predict_batch(df)
                                    if show_progress:
                                        progress_bar.progress(1.0)
                                else:
                                    if show_progress:
                                        status_text.text("Processing in batches...")
                                    results = []
                                    
                                    for i in range(0, len(df), batch_size):
                                        batch_df = df.iloc[i:i+batch_size]
                                        batch_result = predict_batch(batch_df)
                                        results.extend(batch_result['results'])
                                        
                                        if show_progress:
                                            progress = min((i + batch_size) / len(df), 1.0)
                                            progress_bar.progress(progress)
                                            status_text.text(f"Processed {min(i + batch_size, len(df))}/{len(df)} transactions")
                                    
                                    # Combine results
                                    result = {
                                        'results': results,
                                        'summary': {
                                            'total_transactions': len(df),
                                            'flagged_count': sum(1 for r in results if r['flagged']),
                                            'flagged_percentage': sum(1 for r in results if r['flagged']) / len(results) * 100,
                                            'avg_score': sum(r['score'] for r in results) / len(results),
                                            'processing_time': time.time() - start_time,
                                            'warnings': []
                                        }
                                    }
                            
                            # Clear progress indicators
                            if show_progress:
                                progress_bar.empty()
                                status_text.empty()
                            
                            processing_time = time.time() - start_time
                            st.success(f"✅ Processing complete in {processing_time:.2f} seconds!")
                            
                            # Enhanced results display
                            summary = result['summary']
                            st.markdown("#### 📈 Processing Results")
                            
                            # Key metrics with enhanced styling
                            result_col1, result_col2, result_col3, result_col4, result_col5 = st.columns(5)
                            
                            with result_col1:
                                st.metric("✅ Processed", f"{summary['total_transactions']:,}")
                            with result_col2:
                                st.metric("🚨 Flagged", f"{summary['flagged_count']:,}")
                            with result_col3:
                                fraud_rate = summary['flagged_percentage']
                                st.metric("📊 Fraud Rate", f"{fraud_rate:.1f}%")
                            with result_col4:
                                st.metric("🎯 Avg Score", f"{summary['avg_score']:.3f}")
                            with result_col5:
                                st.metric("⚡ Time", f"{processing_time:.2f}s")
                            
                            # Create enhanced results dataframe
                            results_df = df.copy()
                            results_df['fraud_score'] = [r['score'] for r in result['results']]
                            results_df['fraud_flagged'] = [r['flagged'] for r in result['results']]
                            results_df['risk_level'] = [r['risk_level'] for r in result['results']]
                            
                            # Visualizations
                            if include_visualization:
                                st.markdown("#### 📊 Analysis Visualizations")
                                
                                viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
                                    "📈 Score Distribution", 
                                    "🥧 Risk Levels", 
                                    "📊 Type Analysis",
                                    "💰 Amount vs Risk"
                                ])
                                
                                with viz_tab1:
                                    scores = [r['score'] for r in result['results']]
                                    fig_dist = create_score_distribution_plot(scores)
                                    st.plotly_chart(fig_dist, use_container_width=True)
                                
                                with viz_tab2:
                                    risk_levels = [r['risk_level'] for r in result['results']]
                                    fig_pie = create_risk_level_pie_chart(risk_levels)
                                    st.plotly_chart(fig_pie, use_container_width=True)
                                
                                with viz_tab3:
                                    # Transaction type analysis
                                    type_analysis = results_df.groupby('type').agg({
                                        'fraud_flagged': ['count', 'sum'],
                                        'fraud_score': 'mean'
                                    }).round(3)
                                    type_analysis.columns = ['Total', 'Flagged', 'Avg_Score']
                                    type_analysis['Fraud_Rate'] = (type_analysis['Flagged'] / type_analysis['Total'] * 100).round(1)
                                    
                                    fig_type = px.bar(
                                        type_analysis.reset_index(),
                                        x='type',
                                        y='Fraud_Rate',
                                        title='Fraud Rate by Transaction Type',
                                        color='Fraud_Rate',
                                        color_continuous_scale='Reds'
                                    )
                                    st.plotly_chart(fig_type, use_container_width=True)
                                
                                with viz_tab4:
                                    # Amount vs Risk scatter plot
                                    fig_scatter = px.scatter(
                                        results_df,
                                        x='amount',
                                        y='fraud_score',
                                        color='risk_level',
                                        size='fraud_score',
                                        title='Transaction Amount vs Fraud Score',
                                        color_discrete_map={
                                            'HIGH': '#ff6b6b',
                                            'MEDIUM': '#feca57',
                                            'LOW': '#48cab2'
                                        }
                                    )
                                    fig_scatter.update_layout(height=400)
                                    st.plotly_chart(fig_scatter, use_container_width=True)
                            
                            # Results exploration
                            st.markdown("#### 🔍 Results Explorer")
                            
                            # Enhanced filtering options
                            filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
                            
                            with filter_col1:
                                show_flagged_only = st.checkbox("🚨 Show Only Flagged", value=False)
                            with filter_col2:
                                risk_filter = st.selectbox(
                                    "📊 Risk Level", 
                                    ['All', 'HIGH', 'MEDIUM', 'LOW'], 
                                    index=0
                                )
                            with filter_col3:
                                type_filter = st.selectbox(
                                    "💱 Transaction Type",
                                    ['All'] + TXN_TYPES,
                                    index=0
                                )
                            with filter_col4:
                                min_score = st.slider(
                                    "🎯 Min Score", 
                                    0.0, 1.0, 0.0, 0.1,
                                    help="Filter by minimum fraud score"
                                )
                            
                            # Apply filters
                            filtered_df = results_df.copy()
                            
                            if show_flagged_only:
                                filtered_df = filtered_df[filtered_df['fraud_flagged'] == True]
                            if risk_filter != 'All':
                                filtered_df = filtered_df[filtered_df['risk_level'] == risk_filter]
                            if type_filter != 'All':
                                filtered_df = filtered_df[filtered_df['type'] == type_filter]
                            if min_score > 0:
                                filtered_df = filtered_df[filtered_df['fraud_score'] >= min_score]
                            
                            st.info(f"📊 Showing {len(filtered_df):,} of {len(results_df):,} transactions")
                            
                            # Enhanced results table
                            if len(filtered_df) > 0:
                                display_df = filtered_df.copy()
                                display_df['fraud_score'] = display_df['fraud_score'].round(4)
                                
                                st.dataframe(
                                    display_df,
                                    use_container_width=True,
                                    column_config={
                                        "fraud_score": st.column_config.ProgressColumn(
                                            "Fraud Score",
                                            help="AI-calculated fraud probability",
                                            min_value=0,
                                            max_value=1,
                                        ),
                                        "fraud_flagged": st.column_config.CheckboxColumn(
                                            "Flagged",
                                            help="Flagged as potential fraud",
                                        ),
                                        "risk_level": st.column_config.SelectboxColumn(
                                            "Risk Level",
                                            help="Risk assessment category",
                                            options=['LOW', 'MEDIUM', 'HIGH']
                                        ),
                                        "amount": st.column_config.NumberColumn(
                                            "Amount",
                                            format="$%.2f"
                                        )
                                    }
                                )
                            else:
                                st.warning("No transactions match the current filters")
                            
                            # Enhanced download section
                            st.markdown("#### 💾 Export Results")
                            download_col1, download_col2, download_col3 = st.columns(3)
                            
                            with download_col1:
                                # Full results
                                csv_data = results_df.to_csv(index=False)
                                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                st.download_button(
                                    "📊 Download All Results",
                                    csv_data,
                                    f"fraud_analysis_full_{timestamp}.csv",
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
                                        f"flagged_transactions_{timestamp}.csv",
                                        "text/csv",
                                        use_container_width=True
                                    )
                                else:
                                    st.info("No flagged transactions")
                            
                            with download_col3:
                                # Filtered results
                                if len(filtered_df) > 0:
                                    filtered_csv = filtered_df.to_csv(index=False)
                                    st.download_button(
                                        "🔍 Download Filtered",
                                        filtered_csv,
                                        f"filtered_results_{timestamp}.csv",
                                        "text/csv",
                                        use_container_width=True
                                    )
                                else:
                                    st.info("No filtered results")
                            
                            # Generate report
                            if generate_report:
                                st.markdown("#### 📋 Executive Summary Report")
                                
                                report_data = {
                                    "Processing Summary": {
                                        "Total Transactions Processed": f"{summary['total_transactions']:,}",
                                        "Processing Time": f"{processing_time:.2f} seconds",
                                        "Transactions per Second": f"{summary['total_transactions']/processing_time:.1f}",
                                        "Batch Size Used": f"{batch_size:,}"
                                    },
                                    "Fraud Detection Results": {
                                        "Total Flagged": f"{summary['flagged_count']:,}",
                                        "Fraud Detection Rate": f"{fraud_rate:.2f}%",
                                        "Average Fraud Score": f"{summary['avg_score']:.4f}",
                                        "High Risk Transactions": f"{len(results_df[results_df['risk_level'] == 'HIGH']):,}"
                                    },
                                    "Risk Distribution": {
                                        "High Risk": f"{len(results_df[results_df['risk_level'] == 'HIGH']):,}",
                                        "Medium Risk": f"{len(results_df[results_df['risk_level'] == 'MEDIUM']):,}",
                                        "Low Risk": f"{len(results_df[results_df['risk_level'] == 'LOW']):,}"
                                    }
                                }
                                
                                for section, data in report_data.items():
                                    with st.expander(f"📊 {section}", expanded=True):
                                        for key, value in data.items():
                                            col1, col2 = st.columns([2, 1])
                                            col1.write(f"**{key}:**")
                                            col2.write(value)
                            
                        except Exception as e:
                            st.error(f"❌ Processing failed: {str(e)}")
                            with st.expander("🔍 Error Details"):
                                st.exception(e)
                            st.info("💡 Please check your data format and API connection")
        
        except Exception as e:
            st.error(f"❌ File processing failed: {str(e)}")
            st.info("💡 Please ensure your CSV file is properly formatted")

# ================== Tab 3: Enhanced Analytics Dashboard ==================
with tab3:
    st.markdown("### 📊 System Analytics Dashboard")
    st.markdown("Monitor system performance and fraud detection trends")
    
    # Enhanced analytics with real-time feel
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 📈 Performance Overview")
    with col2:
        if st.button("🔄 Refresh Analytics", use_container_width=True):
            st.rerun()
    
    # Generate enhanced sample analytics
    if st.button("🎯 Generate Sample Analytics", use_container_width=True, type="primary"):
        # Enhanced sample data generation
        with st.spinner("📊 Generating analytics data..."):
            dates = pd.date_range(start='2024-01-01', end='2024-02-29', freq='D')
            np.random.seed(42)
            
            # More realistic data patterns
            base_predictions = 100
            weekend_factor = np.where(pd.to_datetime(dates).weekday >= 5, 0.7, 1.0)
            trend_factor = np.linspace(1.0, 1.3, len(dates))
            
            daily_stats = pd.DataFrame({
                'date': dates,
                'total_predictions': np.random.poisson(base_predictions * weekend_factor * trend_factor),
                'fraud_detected': np.random.poisson(15 * weekend_factor),
                'avg_score': np.random.beta(2, 5, len(dates)),
                'api_response_time': np.random.gamma(2, 0.1, len(dates)),
                'system_load': np.random.uniform(0.3, 0.9, len(dates)),
                'accuracy': np.random.uniform(0.92, 0.98, len(dates))
            })
            
            # Calculate additional metrics
            daily_stats['fraud_rate'] = daily_stats['fraud_detected'] / daily_stats['total_predictions']
            daily_stats['day_of_week'] = pd.to_datetime(daily_stats['date']).dt.day_name()
            
        # Enhanced key metrics with trends
        st.markdown("#### 📊 Key Performance Indicators")
        
        kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
        
        total_predictions = daily_stats['total_predictions'].sum()
        total_fraud = daily_stats['fraud_detected'].sum()
        avg_response_time = daily_stats['api_response_time'].mean()
        avg_accuracy = daily_stats['accuracy'].mean()
        avg_system_load = daily_stats['system_load'].mean()
        
        # Calculate trends (last 7 days vs previous 7 days)
        recent_fraud_rate = daily_stats.tail(7)['fraud_rate'].mean()
        previous_fraud_rate = daily_stats.tail(14).head(7)['fraud_rate'].mean()
        fraud_rate_trend = recent_fraud_rate - previous_fraud_rate
        
        with kpi_col1:
            st.metric(
                "🔍 Total Predictions", 
                f"{total_predictions:,}",
                delta=f"+{daily_stats.tail(1)['total_predictions'].values[0]}"
            )
        
        with kpi_col2:
            st.metric(
                "🚨 Fraud Detected", 
                f"{total_fraud:,}",
                delta=f"{fraud_rate_trend:.2%}" if fraud_rate_trend != 0 else None,
                delta_color="inverse"
            )
        
        with kpi_col3:
            st.metric(
                "⚡ Avg Response Time", 
                f"{avg_response_time:.3f}s",
                delta=f"{(daily_stats.tail(1)['api_response_time'].values[0] - avg_response_time):.3f}s",
                delta_color="inverse"
            )
        
        with kpi_col4:
            st.metric(
                "🎯 Model Accuracy", 
                f"{avg_accuracy:.1%}",
                delta=f"{(daily_stats.tail(1)['accuracy'].values[0] - avg_accuracy):.2%}"
            )
        
        with kpi_col5:
            st.metric(
                "💻 System Load", 
                f"{avg_system_load:.1%}",
                delta=f"{(daily_stats.tail(1)['system_load'].values[0] - avg_system_load):.2%}",
                delta_color="inverse"
            )
        
        # Enhanced visualizations
        st.markdown("#### 📈 Trend Analysis")
        
        chart_tab1, chart_tab2, chart_tab3, chart_tab4 = st.tabs([
            "📊 Daily Trends", 
            "🔍 Performance Metrics", 
            "📅 Weekly Patterns",
            "🎯 System Health"
        ])
        
        with chart_tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Enhanced daily predictions trend
                fig_trend = px.line(
                    daily_stats, 
                    x='date', 
                    y=['total_predictions', 'fraud_detected'],
                    title="📈 Daily Predictions vs Fraud Detection",
                    color_discrete_map={
                        'total_predictions': '#667eea',
                        'fraud_detected': '#ff6b6b'
                    }
                )
                fig_trend.update_layout(
                    height=400,
                    hovermode='x unified',
                    font={'family': 'Inter'}
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            
            with col2:
                # Fraud rate over time with threshold line
                fig_rate = px.line(
                    daily_stats,
                    x='date',
                    y='fraud_rate',
                    title="📊 Fraud Detection Rate Trend"
                )
                fig_rate.add_hline(
                    y=daily_stats['fraud_rate'].mean(),
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Average Rate"
                )
                fig_rate.update_layout(height=400)
                st.plotly_chart(fig_rate, use_container_width=True)
        
        with chart_tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                # Response time distribution
                fig_response = px.histogram(
                    daily_stats,
                    x='api_response_time',
                    title="⚡ API Response Time Distribution",
                    nbins=20,
                    color_discrete_sequence=['#48cab2']
                )
                fig_response.add_vline(
                    x=daily_stats['api_response_time'].mean(),
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Mean"
                )
                st.plotly_chart(fig_response, use_container_width=True)
            
            with col2:
                # System performance correlation
                fig_correlation = px.scatter(
                    daily_stats,
                    x='system_load',
                    y='api_response_time',
                    color='accuracy',
                    size='total_predictions',
                    title="💻 System Load vs Performance",
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_correlation, use_container_width=True)
        
        with chart_tab3:
            # Weekly pattern analysis
            weekly_stats = daily_stats.groupby('day_of_week').agg({
                'total_predictions': 'mean',
                'fraud_detected': 'mean',
                'fraud_rate': 'mean',
                'api_response_time': 'mean'
            }).round(2)
            
            # Reorder days of week
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekly_stats = weekly_stats.reindex(day_order)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_weekly_volume = px.bar(
                    weekly_stats.reset_index(),
                    x='day_of_week',
                    y='total_predictions',
                    title="📅 Average Predictions by Day of Week",
                    color='total_predictions',
                    color_continuous_scale='Blues'
                )
                fig_weekly_volume.update_xaxes(categoryorder='array', categoryarray=day_order)
                st.plotly_chart(fig_weekly_volume, use_container_width=True)
            
            with col2:
                fig_weekly_fraud = px.line(
                    weekly_stats.reset_index(),
                    x='day_of_week',
                    y='fraud_rate',
                    title="🚨 Fraud Rate by Day of Week",
                    markers=True
                )
                fig_weekly_fraud.update_xaxes(categoryorder='array', categoryarray=day_order)
                st.plotly_chart(fig_weekly_fraud, use_container_width=True)
        
        with chart_tab4:
            # System health indicators
            col1, col2 = st.columns(2)
            
            with col1:
                # System uptime simulation
                uptime_data = pd.DataFrame({
                    'Component': ['API Server', 'ML Model', 'Database', 'Cache', 'Queue'],
                    'Uptime': [99.9, 99.7, 99.8, 99.95, 99.85],
                    'Status': ['Healthy', 'Healthy', 'Healthy', 'Healthy', 'Healthy']
                })
                
                fig_uptime = px.bar(
                    uptime_data,
                    x='Component',
                    y='Uptime',
                    title="🛡️ System Component Uptime",
                    color='Uptime',
                    color_continuous_scale='Greens'
                )
                fig_uptime.update_layout(yaxis_range=[99, 100])
                st.plotly_chart(fig_uptime, use_container_width=True)
            
            with col2:
                # Resource utilization gauge
                current_load = daily_stats['system_load'].iloc[-1]
                fig_gauge_load = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = current_load * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Current System Load"},
                    delta = {'reference': avg_system_load * 100},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgreen"},
                            {'range': [50, 75], 'color': "yellow"},
                            {'range': [75, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 80
                        }
                    }
                ))
                fig_gauge_load.update_layout(height=400)
                st.plotly_chart(fig_gauge_load, use_container_width=True)
        
        # Enhanced summary table
        st.markdown("#### 📋 Detailed Analytics Summary")
        
        summary_tab1, summary_tab2, summary_tab3 = st.tabs(["📊 Daily Stats", "🔍 Insights", "📈 Forecasts"])
        
        with summary_tab1:
            # Show recent daily statistics
            display_stats = daily_stats.tail(10).copy()
            display_stats['date'] = display_stats['date'].dt.strftime('%Y-%m-%d')
            display_stats = display_stats.round(3)
            
            st.dataframe(
                display_stats,
                use_container_width=True,
                column_config={
                    "fraud_rate": st.column_config.ProgressColumn(
                        "Fraud Rate",
                        min_value=0,
                        max_value=display_stats['fraud_rate'].max()
                    ),
                    "system_load": st.column_config.ProgressColumn(
                        "System Load",
                        min_value=0,
                        max_value=1
                    ),
                    "accuracy": st.column_config.ProgressColumn(
                        "Accuracy",
                        min_value=0,
                        max_value=1
                    )
                }
            )
        
        with summary_tab2:
            # Key insights
            st.markdown("##### 🔍 Key Insights")
            
            insights = []
            
            # Peak day analysis
            peak_day = daily_stats.loc[daily_stats['total_predictions'].idxmax(), 'date'].strftime('%Y-%m-%d')
            peak_predictions = daily_stats['total_predictions'].max()
            insights.append(f"📈 **Peak Activity**: {peak_predictions:,} predictions on {peak_day}")
            
            # Fraud rate analysis
            avg_fraud_rate = daily_stats['fraud_rate'].mean()
            recent_fraud_rate = daily_stats.tail(7)['fraud_rate'].mean()
            if recent_fraud_rate > avg_fraud_rate:
                insights.append(f"⚠️ **Fraud Rate Alert**: Recent rate ({recent_fraud_rate:.1%}) above average ({avg_fraud_rate:.1%})")
            else:
                insights.append(f"✅ **Fraud Rate Normal**: Recent rate ({recent_fraud_rate:.1%}) within expected range")
            
            # Performance analysis
            recent_response_time = daily_stats.tail(7)['api_response_time'].mean()
            if recent_response_time > avg_response_time:
                insights.append(f"🐌 **Performance Alert**: Response time increased to {recent_response_time:.3f}s")
            else:
                insights.append(f"⚡ **Good Performance**: Response time stable at {recent_response_time:.3f}s")
            
            # Weekend vs weekday comparison
            weekday_stats = daily_stats[pd.to_datetime(daily_stats['date']).dt.weekday < 5]
            weekend_stats = daily_stats[pd.to_datetime(daily_stats['date']).dt.weekday >= 5]
            weekday_avg = weekday_stats['total_predictions'].mean()
            weekend_avg = weekend_stats['total_predictions'].mean()
            
            if weekend_avg < weekday_avg * 0.8:
                insights.append(f"📅 **Weekend Pattern**: {((weekend_avg/weekday_avg-1)*100):+.0f}% less activity on weekends")
            
            for insight in insights:
                st.markdown(insight)
        
        with summary_tab3:
            # Simple forecasting with error handling
            st.markdown("##### 📈 7-Day Forecast")
            
            try:
                # Try to import scipy for linear regression
                from scipy import stats
                use_scipy = True
            except ImportError:
                # Fallback to simple numpy calculations
                use_scipy = False
                st.info("📊 Using simplified forecasting (install scipy for advanced features)")
            
            # Get trend for predictions
            x = np.arange(len(daily_stats))
            y = daily_stats['total_predictions'].values
            
            if use_scipy:
                # Use scipy linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            else:
                # Simple linear regression using numpy
                coefficients = np.polyfit(x, y, 1)
                slope, intercept = coefficients
                # Calculate correlation coefficient manually
                correlation_matrix = np.corrcoef(x, y)
                r_value = correlation_matrix[0, 1]
            
            # Forecast next 7 days
            forecast_days = 7
            future_x = np.arange(len(daily_stats), len(daily_stats) + forecast_days)
            forecast_predictions = slope * future_x + intercept
            
            # Ensure non-negative predictions
            forecast_predictions = np.maximum(forecast_predictions, 0)
            
            # Create forecast dataframe
            future_dates = pd.date_range(
                start=daily_stats['date'].iloc[-1] + timedelta(days=1),
                periods=forecast_days,
                freq='D'
            )
            
            forecast_df = pd.DataFrame({
                'date': future_dates,
                'predicted_volume': forecast_predictions.astype(int),
                'confidence': ['High' if abs(r_value) > 0.7 else 'Medium' for _ in range(forecast_days)]
            })
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Combine historical and forecast data for visualization
                hist_data = daily_stats[['date', 'total_predictions']].copy()
                hist_data['type'] = 'Historical'
                
                forecast_data = forecast_df[['date', 'predicted_volume']].copy()
                forecast_data.columns = ['date', 'total_predictions']
                forecast_data['type'] = 'Forecast'
                
                combined_data = pd.concat([hist_data.tail(14), forecast_data])
                
                fig_forecast = px.line(
                    combined_data,
                    x='date',
                    y='total_predictions',
                    color='type',
                    title="Transaction Volume Forecast",
                    color_discrete_map={'Historical': '#667eea', 'Forecast': '#ff6b6b'}
                )
                fig_forecast.update_traces(
                    line_dash='dash',
                    selector=dict(name='Forecast')
                )
                st.plotly_chart(fig_forecast, use_container_width=True)
            
            with col2:
                st.markdown("**Forecast Summary**")
                st.metric("Avg Daily Volume", f"{forecast_predictions.mean():.0f}")
                st.metric("Trend", f"{slope:+.1f} per day")
                if use_scipy:
                    st.metric("R-squared", f"{r_value**2:.3f}")
                else:
                    st.metric("Correlation", f"{r_value:.3f}")
                
                # Show forecast table - Fixed styling issue
                display_forecast_df = forecast_df.copy()
                display_forecast_df['date'] = display_forecast_df['date'].dt.strftime('%Y-%m-%d')
                display_forecast_df['predicted_volume'] = display_forecast_df['predicted_volume'].apply(lambda x: f"{x:,.0f}")
                
                st.dataframe(
                    display_forecast_df,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "date": "Date",
                        "predicted_volume": "Predicted Volume",
                        "confidence": "Confidence"
                    }
                )

# ================== Tab 4: Enhanced Transaction Builder ==================
with tab4:
    st.markdown("### 📋 Advanced Transaction Builder")
    st.markdown("Create and test custom transaction scenarios with advanced features")
    
    # Enhanced scenario templates
    st.markdown("#### 🎯 Quick Scenario Templates")
    
    template_col1, template_col2, template_col3, template_col4 = st.columns(4)
    
    scenarios = {
        "💳 Normal Transfer": {
            "step": 1,
            "type": "TRANSFER",
            "amount": 5000.0,
            "oldbalanceOrg": 10000.0,
            "newbalanceOrg": 5000.0,
            "oldbalanceDest": 2000.0,
            "newbalanceDest": 7000.0,
            "isFlaggedFraud": 0,
            "description": "Standard transfer between accounts"
        },
        "🚨 Large Suspicious Transfer": {
            "step": 100,
            "type": "TRANSFER",
            "amount": 500000.0,
            "oldbalanceOrg": 500000.0,
            "newbalanceOrg": 0.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "isFlaggedFraud": 0,
            "description": "Large amount with zero destination balance"
        },
        "💰 Cash Out Pattern": {
            "step": 50,
            "type": "CASH_OUT",
            "amount": 100000.0,
            "oldbalanceOrg": 100000.0,
            "newbalanceOrg": 0.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 100000.0,
            "isFlaggedFraud": 0,
            "description": "High-value cash withdrawal"
        },
        "⚡ Micro Transaction": {
            "step": 25,
            "type": "PAYMENT",
            "amount": 1.0,
            "oldbalanceOrg": 1000.0,
            "newbalanceOrg": 999.0,
            "oldbalanceDest": 500.0,
            "newbalanceDest": 501.0,
            "isFlaggedFraud": 0,
            "description": "Very small payment amount"
        }
    }
    
    selected_scenario = None
    
    with template_col1:
        if st.button("💳 Normal Transfer", use_container_width=True, help=scenarios["💳 Normal Transfer"]["description"]):
            selected_scenario = "💳 Normal Transfer"
            
    with template_col2:
        if st.button("🚨 Suspicious Transfer", use_container_width=True, help=scenarios["🚨 Large Suspicious Transfer"]["description"]):
            selected_scenario = "🚨 Large Suspicious Transfer"
            
    with template_col3:
        if st.button("💰 Cash Out", use_container_width=True, help=scenarios["💰 Cash Out Pattern"]["description"]):
            selected_scenario = "💰 Cash Out Pattern"
            
    with template_col4:
        if st.button("⚡ Micro Payment", use_container_width=True, help=scenarios["⚡ Micro Transaction"]["description"]):
            selected_scenario = "⚡ Micro Transaction"
    
    # Enhanced transaction builder
    st.markdown("#### 🔧 Custom Transaction Builder")
    
    # Initialize session state for transaction list
    if 'transaction_list' not in st.session_state:
        st.session_state.transaction_list = []
    
    # Transaction builder form with enhanced UI
    with st.form("enhanced_transaction_builder", clear_on_submit=False):
        
        # Load scenario data if selected
        if selected_scenario:
            scenario_data = scenarios[selected_scenario]
            st.success(f"✅ Loaded scenario: {selected_scenario}")
            st.info(f"📝 {scenario_data['description']}")
        else:
            scenario_data = {}
        
        # Enhanced form layout
        st.markdown("##### 📊 Transaction Information")
        
        trans_col1, trans_col2, trans_col3 = st.columns(3)
        
        with trans_col1:
            step = st.number_input(
                "⏰ Time Step", 
                value=scenario_data.get("step", 1),
                min_value=0,
                help="Sequential step in transaction timeline"
            )
            
        with trans_col2:
            txn_type = st.selectbox(
                "💱 Transaction Type", 
                TXN_TYPES,
                index=TXN_TYPES.index(scenario_data.get("type", "TRANSFER")),
                help="Select the type of financial transaction"
            )
            
        with trans_col3:
            amount = st.number_input(
                "💰 Transaction Amount", 
                value=scenario_data.get("amount", 1000.0),
                min_value=0.0,
                step=100.0,
                format="%.2f",
                help="Amount to be transferred"
            )
        
        st.markdown("##### 👤 Account Details")
        
        account_col1, account_col2 = st.columns(2)
        
        with account_col1:
            st.markdown("**🏦 Origin Account**")
            name_orig = st.text_input(
                "Account ID", 
                value=scenario_data.get("nameOrig", f"C{np.random.randint(1000000, 9999999)}"),
                help="Unique identifier for origin account",
                key="orig_account_id"  # Add unique key
            )
            old_org = st.number_input(
                "Balance Before", 
                value=scenario_data.get("oldbalanceOrg", 1000.0),
                min_value=0.0,
                step=100.0,
                format="%.2f",
                key="orig_balance_before"  # Add unique key
            )
            new_org = st.number_input(
                "Balance After", 
                value=scenario_data.get("newbalanceOrg", 0.0),
                min_value=0.0,
                step=100.0,
                format="%.2f",
                key="orig_balance_after"  # Add unique key
            )
                
        with account_col2:
            st.markdown("**🎯 Destination Account**")
            name_dest = st.text_input(
                "Account ID", 
                value=scenario_data.get("nameDest", f"M{np.random.randint(1000000, 9999999)}"),
                help="Unique identifier for destination account",
                key="dest_account_id"  # Add unique key
            )
            old_dest = st.number_input(
                "Balance Before", 
                value=scenario_data.get("oldbalanceDest", 0.0),
                min_value=0.0,
                step=100.0,
                format="%.2f",
                key="dest_balance_before"  # Add unique key
            )
            new_dest = st.number_input(
                "Balance After", 
                value=scenario_data.get("newbalanceDest", 1000.0),
                min_value=0.0,
                step=100.0,
                format="%.2f",
                key="dest_balance_after"  # Add unique key
            )
        
        st.markdown("##### 🚩 Additional Flags")
        
        flag_col1, flag_col2, flag_col3 = st.columns(3)
        
        with flag_col1:
            is_flagged = st.selectbox(
                "Initially Flagged", 
                [0, 1], 
                index=scenario_data.get("isFlaggedFraud", 0),
                format_func=lambda x: "🚨 Yes" if x else "✅ No",
                help="Basic rule-based flagging"
            )
        
        with flag_col2:
            add_to_batch = st.checkbox("📦 Add to Batch List", value=False)
            
        with flag_col3:
            validate_balance = st.checkbox("🔍 Validate Balances", value=True)
        
        # Balance validation
        if validate_balance:
            balance_check_passed = True
            balance_warnings = []
            
            # Check if balances make sense
            expected_new_org = old_org - amount if txn_type in ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT"] else old_org + amount
            expected_new_dest = old_dest + amount if txn_type in ["TRANSFER", "CASH_IN"] else old_dest
            
            if abs(new_org - expected_new_org) > 0.01:
                balance_warnings.append(f"⚠️ Origin balance: Expected {expected_new_org:.2f}, got {new_org:.2f}")
                balance_check_passed = False
                
            if abs(new_dest - expected_new_dest) > 0.01 and txn_type in ["TRANSFER", "CASH_IN"]:
                balance_warnings.append(f"⚠️ Destination balance: Expected {expected_new_dest:.2f}, got {new_dest:.2f}")
                balance_check_passed = False
            
            if balance_warnings:
                for warning in balance_warnings:
                    st.warning(warning)
            else:
                st.success("✅ Balance validation passed")
        
        # Form submission buttons
        st.markdown("---")
        
        button_col1, button_col2, button_col3, button_col4 = st.columns(4)
        
        with button_col1:
            submit_single = st.form_submit_button("🔍 Analyze Single", use_container_width=True, type="primary")
            
        with button_col2:
            add_to_list = st.form_submit_button("➕ Add to List", use_container_width=True)
            
        with button_col3:
            generate_random = st.form_submit_button("🎲 Generate Random", use_container_width=True)
            
        with button_col4:
            clear_form = st.form_submit_button("🗑️ Clear Form", use_container_width=True)
    
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
            with st.spinner("🔄 Analyzing transaction..."):
                result = predict_single(transaction_data)
            
            # Enhanced single result display
            st.markdown("#### 🎯 Analysis Results")
            
            result_col1, result_col2, result_col3 = st.columns(3)
            
            with result_col1:
                st.markdown("##### 📊 Core Metrics")
                st.metric("Fraud Score", f"{result['score']:.4f}")
                st.metric("Risk Level", result['risk_level'])
                flagged_text = "🚨 FLAGGED" if result['flagged'] else "✅ CLEAR"
                st.metric("Status", flagged_text)
                
            with result_col2:
                st.markdown("##### 🎯 Model Details")
                st.metric("Threshold Used", f"{result['threshold']:.3f}")
                confidence = abs(result['score'] - result['threshold']) * 100
                st.metric("Confidence", f"{confidence:.1f}%")
                
            with result_col3:
                st.markdown("##### 💡 Risk Assessment")
                risk_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
                risk_emoji = risk_color.get(result['risk_level'], "⚪")
                
                st.markdown(f"**{risk_emoji} {result['risk_level']} RISK**")
                
                if result['flagged']:
                    if result['risk_level'] == 'HIGH':
                        st.error("Immediate review required")
                    elif result['risk_level'] == 'MEDIUM':
                        st.warning("Manual verification needed")
                    else:
                        st.info("Standard fraud workflow")
                else:
                    st.success("Transaction approved")
            
            # Enhanced gauge visualization
            fig_single_gauge = create_enhanced_gauge(result['score'], result['threshold'], "Fraud Risk Assessment")
            st.plotly_chart(fig_single_gauge, use_container_width=True)
            
            # Transaction details
            with st.expander("📄 Transaction Details", expanded=False):
                details_col1, details_col2 = st.columns(2)
                
                with details_col1:
                    st.json({
                        "Basic Info": {
                            "Step": transaction_data["step"],
                            "Type": transaction_data["type"],
                            "Amount": f"${transaction_data['amount']:,.2f}"
                        },
                        "Origin Account": {
                            "ID": transaction_data["nameOrig"],
                            "Old Balance": f"${transaction_data['oldbalanceOrg']:,.2f}",
                            "New Balance": f"${transaction_data['newbalanceOrg']:,.2f}"
                        }
                    })
                    
                with details_col2:
                    st.json({
                        "Destination Account": {
                            "ID": transaction_data["nameDest"],
                            "Old Balance": f"${transaction_data['oldbalanceDest']:,.2f}",
                            "New Balance": f"${transaction_data['newbalanceDest']:,.2f}"
                        },
                        "Flags": {
                            "Initially Flagged": bool(transaction_data["isFlaggedFraud"]),
                            "AI Flagged": result['flagged']
                        }
                    })
                    
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            with st.expander("🔍 Error Details"):
                st.code(str(e))
    
    if add_to_list:
        st.session_state.transaction_list.append(transaction_data.copy())
        st.success(f"✅ Added to batch list! Total transactions: {len(st.session_state.transaction_list)}")
        
        if add_to_batch:
            st.balloons()
    
    if generate_random:
        st.info("🎲 Generated random transaction values!")
        st.rerun()
    
    if clear_form:
        st.success("🗑️ Form cleared!")
        st.rerun()
    
    # Enhanced transaction list management
    if st.session_state.transaction_list:
        st.markdown("#### 📦 Batch Transaction List")
        
        list_col1, list_col2, list_col3 = st.columns([2, 1, 1])
        
        with list_col1:
            st.info(f"📊 {len(st.session_state.transaction_list)} transactions ready for batch processing")
            
        with list_col2:
            if st.button("🚀 Process Batch", use_container_width=True, type="primary"):
                try:
                    transactions_df = pd.DataFrame(st.session_state.transaction_list)
                    
                    with st.spinner(f"🔄 Processing {len(st.session_state.transaction_list)} transactions..."):
                        result = predict_batch(transactions_df)
                    
                    st.success("✅ Batch processing complete!")
                    
                    # Enhanced batch results
                    summary = result['summary']
                    results_list = result['results']
                    
                    # Batch metrics
                    batch_col1, batch_col2, batch_col3, batch_col4 = st.columns(4)
                    
                    with batch_col1:
                        st.metric("Total Processed", summary['total_transactions'])
                    with batch_col2:
                        st.metric("Flagged", summary['flagged_count'])
                    with batch_col3:
                        st.metric("Fraud Rate", f"{summary['flagged_percentage']:.1f}%")
                    with batch_col4:
                        st.metric("Avg Score", f"{summary['avg_score']:.3f}")
                    
                    # Enhanced results table
                    results_df = transactions_df.copy()
                    results_df['fraud_score'] = [r['score'] for r in results_list]
                    results_df['fraud_flagged'] = [r['flagged'] for r in results_list]
                    results_df['risk_level'] = [r['risk_level'] for r in results_list]
                    
                    st.dataframe(
                        results_df,
                        use_container_width=True,
                        column_config={
                            "fraud_score": st.column_config.ProgressColumn(
                                "Fraud Score",
                                min_value=0,
                                max_value=1,
                            ),
                            "fraud_flagged": st.column_config.CheckboxColumn("Flagged"),
                            "risk_level": st.column_config.SelectboxColumn(
                                "Risk",
                                options=['LOW', 'MEDIUM', 'HIGH']
                            ),
                            "amount": st.column_config.NumberColumn(
                                "Amount",
                                format="$%.2f"
                            )
                        }
                    )
                    
                    # Batch visualization
                    scores = [r['score'] for r in results_list]
                    risk_levels = [r['risk_level'] for r in results_list]
                    
                    viz_col1, viz_col2 = st.columns(2)
                    
                    with viz_col1:
                        fig_batch_dist = create_score_distribution_plot(scores)
                        st.plotly_chart(fig_batch_dist, use_container_width=True)
                        
                    with viz_col2:
                        fig_batch_risk = create_risk_level_pie_chart(risk_levels)
                        st.plotly_chart(fig_batch_risk, use_container_width=True)
                    
                    # Download batch results
                    csv_batch = results_df.to_csv(index=False)
                    st.download_button(
                        "💾 Download Batch Results",
                        csv_batch,
                        f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        use_container_width=True
                    )
                    
                except Exception as e:
                    st.error(f"❌ Batch processing failed: {str(e)}")
        
        with list_col3:
            if st.button("🗑️ Clear List", use_container_width=True, type="secondary"):
                st.session_state.transaction_list = []
                st.success("✅ Transaction list cleared!")
                st.rerun()
        
        # Show transaction list
        if len(st.session_state.transaction_list) <= 10:
            st.dataframe(
                pd.DataFrame(st.session_state.transaction_list),
                use_container_width=True
            )
        else:
            st.info(f"Showing first 10 of {len(st.session_state.transaction_list)} transactions")
            st.dataframe(
                pd.DataFrame(st.session_state.transaction_list).head(10),
                use_container_width=True
            )

# ================== Tab 5: Real-time Monitor ==================
with tab5:
    st.markdown("### ⚡ Real-time System Monitor")
    st.markdown("Live monitoring of fraud detection system performance")
    
    # Real-time controls
    monitor_col1, monitor_col2, monitor_col3 = st.columns([2, 1, 1])
    
    with monitor_col1:
        st.markdown("#### 📡 Live System Status")
        
    with monitor_col2:
        auto_refresh = st.checkbox("🔄 Auto Refresh", value=False)
        
    with monitor_col3:
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.rerun()
    
    # Simulate real-time data
    current_time = datetime.now()
    
    # Generate real-time metrics
    np.random.seed(int(current_time.timestamp()) % 1000)
    
    # Current system metrics
    current_metrics = {
        "api_requests_per_second": np.random.poisson(25),
        "active_connections": np.random.randint(150, 300),
        "cpu_usage": np.random.uniform(0.3, 0.8),
        "memory_usage": np.random.uniform(0.4, 0.7),
        "queue_length": np.random.randint(0, 50),
        "error_rate": np.random.uniform(0, 0.02),
        "avg_response_time": np.random.uniform(0.1, 0.5),
        "fraud_alerts_last_hour": np.random.randint(5, 25),
        "model_accuracy": np.random.uniform(0.94, 0.98)
    }
    
    # System status indicators
    status_col1, status_col2, status_col3, status_col4 = st.columns(4)
    
    with status_col1:
        # API Status
        if current_metrics["error_rate"] < 0.01:
            st.success("🟢 API: Healthy")
        else:
            st.warning("🟡 API: Degraded")
        
        st.metric("Requests/sec", f"{current_metrics['api_requests_per_second']}")
        st.metric("Error Rate", f"{current_metrics['error_rate']:.2%}")
    
    with status_col2:
        # System Resources
        if current_metrics["cpu_usage"] < 0.7 and current_metrics["memory_usage"] < 0.8:
            st.success("🟢 Resources: Normal")
        else:
            st.warning("🟡 Resources: High")
            
        st.metric("CPU Usage", f"{current_metrics['cpu_usage']:.1%}")
        st.metric("Memory Usage", f"{current_metrics['memory_usage']:.1%}")
    
    with status_col3:
        # Performance
        if current_metrics["avg_response_time"] < 0.3:
            st.success("🟢 Performance: Good")
        else:
            st.warning("🟡 Performance: Slow")
            
        st.metric("Response Time", f"{current_metrics['avg_response_time']:.3f}s")
        st.metric("Queue Length", f"{current_metrics['queue_length']}")
    
    with status_col4:
        # Fraud Detection
        st.success("🟢 ML Model: Active")
        st.metric("Alerts/Hour", f"{current_metrics['fraud_alerts_last_hour']}")
        st.metric("Model Accuracy", f"{current_metrics['model_accuracy']:.1%}")
    
    # Real-time charts
    st.markdown("#### 📊 Live Performance Charts")
    
    # Generate time series data for last hour
    time_points = pd.date_range(
        end=current_time, 
        periods=60, 
        freq='1min'
    )
    
    # Simulate realistic time series patterns
    base_requests = 25
    time_factor = np.sin(np.linspace(0, 2*np.pi, 60)) * 5  # Cyclical pattern
    noise = np.random.normal(0, 3, 60)
    requests_data = base_requests + time_factor + noise
    requests_data = np.maximum(requests_data, 0)  # Ensure non-negative
    
    response_time_data = 0.2 + np.random.gamma(2, 0.05, 60)  # Gamma distribution for response times
    cpu_data = 0.5 + np.sin(np.linspace(0, 4*np.pi, 60)) * 0.1 + np.random.normal(0, 0.05, 60)
    cpu_data = np.clip(cpu_data, 0, 1)
    
    realtime_df = pd.DataFrame({
        'timestamp': time_points,
        'requests_per_second': requests_data,
        'response_time': response_time_data,
        'cpu_usage': cpu_data * 100,
        'fraud_score_avg': np.random.beta(2, 8, 60)  # Typical fraud score distribution
    })
    
    chart_tab1, chart_tab2, chart_tab3 = st.tabs(["📈 Traffic", "⚡ Performance", "🛡️ Security"])
    
    with chart_tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Request rate over time
            fig_requests = px.line(
                realtime_df,
                x='timestamp',
                y='requests_per_second',
                title="🔄 API Requests per Second (Last Hour)",
                line_shape='spline'
            )
            fig_requests.update_layout(
                height=350,
                showlegend=False,
                xaxis_title="Time",
                yaxis_title="Requests/sec"
            )
            fig_requests.update_traces(line_color='#667eea')
            st.plotly_chart(fig_requests, use_container_width=True)
        
        with col2:
            # Active connections gauge
            fig_connections = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = current_metrics['active_connections'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Active Connections"},
                gauge = {
                    'axis': {'range': [None, 500]},
                    'bar': {'color': "#667eea"},
                    'steps': [
                        {'range': [0, 200], 'color': "lightgray"},
                        {'range': [200, 350], 'color': "yellow"},
                        {'range': [350, 500], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 400
                    }
                }
            ))
            fig_connections.update_layout(height=350)
            st.plotly_chart(fig_connections, use_container_width=True)
    
    with chart_tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Response time trend
            fig_response = px.line(
                realtime_df,
                x='timestamp',
                y='response_time',
                title="⚡ Average Response Time (Last Hour)"
            )
            fig_response.add_hline(y=0.3, line_dash="dash", line_color="red", annotation_text="SLA Threshold")
            fig_response.update_layout(height=350)
            fig_response.update_traces(line_color='#ff6b6b')
            st.plotly_chart(fig_response, use_container_width=True)
        
        with col2:
            # CPU usage trend
            fig_cpu = px.line(
                realtime_df,
                x='timestamp',
                y='cpu_usage',
                title="💻 CPU Usage % (Last Hour)"
            )
            fig_cpu.add_hline(y=80, line_dash="dash", line_color="orange", annotation_text="Warning Level")
            fig_cpu.update_layout(height=350, yaxis_range=[0, 100])
            fig_cpu.update_traces(line_color='#feca57')
            st.plotly_chart(fig_cpu, use_container_width=True)
    
    with chart_tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            # Average fraud scores over time
            fig_fraud_trend = px.line(
                realtime_df,
                x='timestamp',
                y='fraud_score_avg',
                title="🛡️ Average Fraud Scores (Last Hour)"
            )
            fig_fraud_trend.update_layout(height=350)
            fig_fraud_trend.update_traces(line_color='#48cab2')
            st.plotly_chart(fig_fraud_trend, use_container_width=True)
        
        with col2:
            # Security events
            security_events = pd.DataFrame({
                'Event Type': ['High Risk Transaction', 'Suspicious Pattern', 'Account Alert', 'Failed Login'],
                'Count (Last Hour)': [
                    np.random.poisson(8),
                    np.random.poisson(3),
                    np.random.poisson(5),
                    np.random.poisson(15)
                ]
            })
            
            fig_security = px.bar(
                security_events,
                x='Event Type',
                y='Count (Last Hour)',
                title="🚨 Security Events (Last Hour)",
                color='Count (Last Hour)',
                color_continuous_scale='Reds'
            )
            fig_security.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_security, use_container_width=True)
    
    # Real-time alerts and notifications
    st.markdown("#### 🚨 Live Alerts & Notifications")
    
    # Generate sample alerts
    alert_types = [
        ("🔴 HIGH", "Suspicious transaction pattern detected", "Account C123456", "2 minutes ago"),
        ("🟡 MEDIUM", "Unusual transaction volume", "System Monitor", "5 minutes ago"),
        ("🟢 LOW", "Model accuracy check passed", "ML System", "8 minutes ago"),
        ("🟡 MEDIUM", "API response time elevated", "Performance Monitor", "12 minutes ago"),
        ("🔴 HIGH", "Multiple failed authentication attempts", "Security System", "15 minutes ago")
    ]
    
    st.markdown("##### Recent Alerts")
    
    for priority, message, source, time_ago in alert_types:
        alert_col1, alert_col2, alert_col3, alert_col4 = st.columns([1, 3, 2, 1])
        
        with alert_col1:
            st.markdown(priority)
        with alert_col2:
            st.write(message)
        with alert_col3:
            st.caption(source)
        with alert_col4:
            st.caption(time_ago)
    
    # System health summary
    st.markdown("#### 🏥 System Health Summary")
    
    health_col1, health_col2 = st.columns(2)
    
    with health_col1:
        st.markdown("##### 📊 Component Status")
        
        components = [
            ("API Gateway", "🟢", "Healthy", 99.9),
            ("ML Model Service", "🟢", "Healthy", 99.8),
            ("Database", "🟢", "Healthy", 99.95),
            ("Cache Layer", "🟡", "Warning", 98.5),
            ("Message Queue", "🟢", "Healthy", 99.7)
        ]
        
        component_df = pd.DataFrame(components, columns=['Component', 'Status', 'Health', 'Uptime %'])
        
        st.dataframe(
            component_df,
            use_container_width=True,
            column_config={
                "Uptime %": st.column_config.ProgressColumn(
                    "Uptime %",
                    min_value=95,
                    max_value=100
                )
            },
            hide_index=True
        )
    
    with health_col2:
        st.markdown("##### 🎯 Performance Targets")
        
        targets = [
            ("API Response Time", f"{current_metrics['avg_response_time']:.3f}s", "< 0.300s", current_metrics['avg_response_time'] < 0.3),
            ("Error Rate", f"{current_metrics['error_rate']:.2%}", "< 1.00%", current_metrics['error_rate'] < 0.01),
            ("CPU Usage", f"{current_metrics['cpu_usage']:.1%}", "< 70%", current_metrics['cpu_usage'] < 0.7),
            ("Memory Usage", f"{current_metrics['memory_usage']:.1%}", "< 80%", current_metrics['memory_usage'] < 0.8),
            ("Queue Length", f"{current_metrics['queue_length']}", "< 100", current_metrics['queue_length'] < 100)
        ]
        
        for metric, current, target, is_good in targets:
            status_emoji = "✅" if is_good else "❌"
            st.metric(
                f"{status_emoji} {metric}",
                current,
                delta=f"Target: {target}",
                delta_color="normal" if is_good else "inverse"
            )
    
    # Auto-refresh functionality
    if auto_refresh:
        time.sleep(5)  # Wait 5 seconds
        st.rerun()

# ================== Enhanced Footer ==================
st.markdown("---")
st.markdown(
    """
    <div class="footer">
        <h4>🛡️ Advanced Fraud Detection System v2.0</h4>
        <p><strong>Powered by:</strong> Machine Learning • Real-time Analytics • Advanced UI/UX</p>
        <p><strong>Features:</strong> Single Transaction Analysis • Batch Processing • Live Monitoring • Custom Scenarios</p>
        <p><strong>Built with:</strong> Streamlit • Plotly • Pandas • NumPy</p>
        <br>
        <p><em>🔒 Secure • ⚡ Fast • 📊 Intelligent • 🎯 Accurate</em></p>
        <p style="color: #64748b; font-size: 0.9em;">
            For technical support: <strong>support@frauddetection.ai</strong> | 
            Documentation: <strong>docs.frauddetection.ai</strong>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)