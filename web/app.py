import os
import json
import time
import requests
import pandas as pd
import streamlit as st

# ================== Config ==================
DEFAULT_API_URL = "http://localhost:8000"
API_URL = os.getenv("API_URL", DEFAULT_API_URL)

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("🚨 Fraud Detection — Web Dashboard")

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API URL", value=API_URL, help="FastAPI endpoint, ví dụ http://localhost:8000")
    if st.button("Ping /health"):
        try:
            r = requests.get(f"{api_url}/health", timeout=5)
            r.raise_for_status()
            data = r.json()
            st.success(f"OK | threshold={data.get('threshold')}, model={data.get('model_path')}")
        except Exception as e:
            st.error(f"Health check failed: {e}")

tab1, tab2 = st.tabs(["🔎 Single Transaction", "📦 Batch CSV"])

# ================== Helpers ==================
TXN_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

def predict_single(payload: dict) -> dict:
    url = f"{api_url}/predict"
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()

def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    # API schema yêu cầu đúng tên cột
    required = [
        "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrg",
        "nameDest", "oldbalanceDest", "newbalanceDest", "isFlaggedFraud"
    ]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"Thiếu cột bắt buộc: {miss}")
    records = df[required].to_dict(orient="records")
    url = f"{api_url}/batch_predict"
    r = requests.post(url, json={"records": records}, timeout=120)
    r.raise_for_status()
    out = r.json()
    scores = [x["score"] for x in out["results"]]
    flags  = [x["flagged"] for x in out["results"]]
    res = df.copy()
    res["score"] = scores
    res["flagged"] = flags
    return res

# ================== Tab 1: Single ==================
with tab1:
    st.subheader("Nhập một giao dịch")
    with st.form("single_txn"):
        c1, c2, c3 = st.columns(3)
        step   = c1.number_input("step (int)", min_value=0, value=500, step=1)
        ttype  = c2.selectbox("type", TXN_TYPES, index=4)  # mặc định TRANSFER
        amount = c3.number_input("amount", min_value=0.0, value=200000.0, step=1000.0)

        c4, c5, c6 = st.columns(3)
        old_org = c4.number_input("oldbalanceOrg", min_value=0.0, value=0.0, step=100.0)
        new_org = c5.number_input("newbalanceOrg", min_value=0.0, value=0.0, step=100.0)
        flag    = c6.selectbox("isFlaggedFraud", [0,1], index=0)

        c7, c8, c9 = st.columns(3)
        old_dst = c7.number_input("oldbalanceDest", min_value=0.0, value=0.0, step=100.0)
        new_dst = c8.number_input("newbalanceDest", min_value=0.0, value=0.0, step=100.0)
        name_o  = c9.text_input("nameOrig (optional)", value="C12345")

        name_d = st.text_input("nameDest (optional)", value="M98765")

        submitted = st.form_submit_button("Predict")

    if submitted:
        payload = {
            "step": int(step),
            "type": ttype,
            "amount": float(amount),
            "nameOrig": name_o or None,
            "oldbalanceOrg": float(old_org),
            "newbalanceOrg": float(new_org),
            "nameDest": name_d or None,
            "oldbalanceDest": float(old_dst),
            "newbalanceDest": float(new_dst),
            "isFlaggedFraud": int(flag),
        }
        try:
            res = predict_single(payload)
            colA, colB, colC = st.columns(3)
            colA.metric("Score", f"{res['score']:.4f}")
            colB.metric("Flagged", "True" if res["flagged"] else "False")
            colC.metric("Threshold", f"{res['threshold']:.2f}")
            st.code(json.dumps(payload, indent=2), language="json")
        except Exception as e:
            st.error(f"Lỗi gọi API: {e}")

# ================== Tab 2: Batch ==================
with tab2:
    st.subheader("Upload CSV & batch predict")
    st.caption("Cột bắt buộc: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrg, nameDest, oldbalanceDest, newbalanceDest, isFlaggedFraud")

    up = st.file_uploader("Chọn file CSV", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up)
            st.write("Preview:", df.head())
            if st.button("Run batch predict"):
                with st.spinner("Đang gọi API..."):
                    out = predict_batch(df)
                st.success(f"Xong! {out['flagged'].sum()} / {len(out)} giao dịch bị flag.")
                st.dataframe(out, use_container_width=True)

                # Lọc nhanh
                flagged_only = st.toggle("Chỉ hiển thị flagged", value=False)
                view = out[out["flagged"]] if flagged_only else out
                st.dataframe(view, use_container_width=True)

                # Tải xuống kết quả
                csv_bytes = view.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Tải CSV kết quả",
                    data=csv_bytes,
                    file_name="fraud_predictions.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Lỗi xử lý CSV: {e}")
