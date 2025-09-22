import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Literal, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

# ========= Config =========
load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH", "models/fraud_xgb.pkl")
THRESHOLD  = float(os.getenv("THRESHOLD", "0.80"))

# Các loại giao dịch có trong dataset gốc
TXN_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

# Danh sách features cuối cùng (PHẢI khớp lúc train)
# -> Bạn điều chỉnh nếu lúc train bạn dùng cột khác
EXPECTED_FEATURES = [
    "step", "amount",
    "oldbalanceOrg", "newbalanceOrg",
    "oldbalanceDest", "newbalanceDest",
    "isFlaggedFraud",
    # one-hot type_* (drop_first=True trong notebook => thiếu 1 cột; để an toàn tạo đủ 4-5 cột, nếu thiếu sẽ fill 0)
    "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER",
    # engineered
    "delta_org", "delta_dest",
    "zero_org_bal", "zero_dest_bal",
    "ratio_org", "ratio_dest", "net_mismatch",
]

# ========= Model load =========
OBJ = joblib.load(MODEL_PATH)
MODEL = OBJ.get("model", OBJ)  # chấp nhận cả trường hợp lưu trực tiếp model
# Cho phép override threshold từ file pkl nếu bạn đã lưu kèm
THRESHOLD = float(OBJ.get("threshold", THRESHOLD))

app = FastAPI(title="Fraud Detection API", version="1.0")

# ========= Schemas =========
class Txn(BaseModel):
    step: int
    type: Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    amount: float
    nameOrig: Optional[str] = None
    oldbalanceOrg: float
    newbalanceOrg: float
    nameDest: Optional[str] = None
    oldbalanceDest: float
    newbalanceDest: float
    isFlaggedFraud: int

class Batch(BaseModel):
    records: List[Txn]

# ========= Preprocess (phải giống notebooks) =========
def preprocess_one(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Nhận raw df cột gốc -> trả về df features đúng thứ tự EXPECTED_FEATURES
    """
    df = df_raw.copy()

    # One-hot 'type' (drop_first=True lúc train; ở đây tạo đủ rồi align EXPECTED_FEATURES)
    type_dummies = pd.get_dummies(df["type"], prefix="type")
    for col in ["type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"]:
        if col not in type_dummies.columns:
            type_dummies[col] = 0
    df = pd.concat([df.drop(columns=["type"]), type_dummies], axis=1)

    # Engineered (giống notebooks)
    df["delta_org"]  = df["oldbalanceOrg"]  - df["newbalanceOrg"]
    df["delta_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]
    df["zero_org_bal"]  = (df["oldbalanceOrg"]  == 0).astype(int)
    df["zero_dest_bal"] = (df["oldbalanceDest"] == 0).astype(int)
    df["ratio_org"]  = df["amount"] / (df["oldbalanceOrg"].abs()  + 1)
    df["ratio_dest"] = df["amount"] / (df["oldbalanceDest"].abs() + 1)
    df["net_mismatch"] = ((df["delta_org"] - df["delta_dest"]).abs() > 1e-3).astype(int)

    # Bảo đảm đầy đủ cột EXPECTED_FEATURES (thiếu thì fill 0)
    for col in EXPECTED_FEATURES:
        if col not in df.columns:
            df[col] = 0

    # Sắp xếp đúng thứ tự
    df = df[EXPECTED_FEATURES]
    return df

# ========= Routes =========
@app.get("/health")
def health():
    return {"status": "ok", "model_path": MODEL_PATH, "threshold": THRESHOLD}

@app.post("/predict")
def predict(txn: Txn):
    # Raw -> DataFrame(1)
    raw = pd.DataFrame([txn.model_dump()])
    X = preprocess_one(raw)
    score = float(MODEL.predict_proba(X)[:, 1][0])
    flagged = bool(score >= THRESHOLD)
    return {"score": score, "flagged": flagged, "threshold": THRESHOLD}

@app.post("/batch_predict")
def batch_predict(batch: Batch):
    raw = pd.DataFrame([r.model_dump() for r in batch.records])
    X = preprocess_one(raw)
    scores = MODEL.predict_proba(X)[:, 1]
    flags  = scores >= THRESHOLD
    return {
        "results": [{"score": float(s), "flagged": bool(f)} for s, f in zip(scores, flags)],
        "threshold": THRESHOLD,
    }
