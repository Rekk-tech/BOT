import os
import joblib
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import List, Literal, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv
import uvicorn

# ========= Logging Setup =========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fraud_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ========= Config =========
load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH", "models/fraud_xgb.pkl")
THRESHOLD = float(os.getenv("THRESHOLD", "0.80"))
LOG_PREDICTIONS = os.getenv("LOG_PREDICTIONS", "true").lower() == "true"

# Transaction types from original dataset
TXN_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

# Feature list (must match training features)
EXPECTED_FEATURES = [
    "step", "amount",
    "oldbalanceOrg", "newbalanceOrg",
    "oldbalanceDest", "newbalanceDest",
    "isFlaggedFraud",
    # one-hot encoded transaction types
    "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER",
    # engineered features
    "delta_org", "delta_dest",
    "zero_org_bal", "zero_dest_bal",
    "ratio_org", "ratio_dest", "net_mismatch",
]

# ========= Model Loading =========
try:
    OBJ = joblib.load(MODEL_PATH)
    MODEL = OBJ.get("model", OBJ)
    THRESHOLD = float(OBJ.get("threshold", THRESHOLD))
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
    logger.info(f"Using threshold: {THRESHOLD}")
except Exception as e:
    logger.error(f"Failed to load model from {MODEL_PATH}: {str(e)}")
    raise RuntimeError(f"Model loading failed: {str(e)}")

# ========= FastAPI App Setup =========
app = FastAPI(
    title="Fraud Detection API",
    description="Advanced fraud detection system for financial transactions",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========= Enhanced Schemas =========
class Transaction(BaseModel):
    step: int = Field(..., ge=0, description="Time step of transaction")
    type: Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"] = Field(..., description="Transaction type")
    amount: float = Field(..., ge=0, description="Transaction amount")
    nameOrig: Optional[str] = Field(None, description="Origin account name")
    oldbalanceOrg: float = Field(..., ge=0, description="Origin account balance before transaction")
    newbalanceOrg: float = Field(..., ge=0, description="Origin account balance after transaction")
    nameDest: Optional[str] = Field(None, description="Destination account name")
    oldbalanceDest: float = Field(..., ge=0, description="Destination account balance before transaction")
    newbalanceDest: float = Field(..., ge=0, description="Destination account balance after transaction")
    isFlaggedFraud: int = Field(..., ge=0, le=1, description="Flag indicating if transaction was flagged (0 or 1)")

    @validator('amount', 'oldbalanceOrg', 'newbalanceOrg', 'oldbalanceDest', 'newbalanceDest')
    def validate_positive_amounts(cls, v):
        if v < 0:
            raise ValueError('Financial amounts must be non-negative')
        return v

class BatchRequest(BaseModel):
    transactions: List[Transaction] = Field(..., min_items=1, max_items=1000, description="List of transactions to predict")
    
class PredictionResponse(BaseModel):
    score: float = Field(..., description="Fraud probability score (0-1)")
    flagged: bool = Field(..., description="Whether transaction is flagged as fraud")
    threshold: float = Field(..., description="Threshold used for flagging")
    risk_level: str = Field(..., description="Risk level: LOW, MEDIUM, HIGH")
    timestamp: str = Field(..., description="Prediction timestamp")

class BatchResponse(BaseModel):
    results: List[PredictionResponse]
    summary: Dict[str, Any]
    threshold: float
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_path: str
    threshold: float
    uptime: str
    features_count: int
    version: str

# ========= Enhanced Preprocessing =========
def validate_transaction_logic(df: pd.DataFrame) -> List[str]:
    """Validate business logic of transactions"""
    warnings = []
    
    for idx, row in df.iterrows():
        # Check balance consistency for different transaction types
        if row['type'] in ['CASH_OUT', 'PAYMENT', 'TRANSFER', 'DEBIT']:
            expected_new_balance = row['oldbalanceOrg'] - row['amount']
            if abs(row['newbalanceOrg'] - expected_new_balance) > 0.01:
                warnings.append(f"Row {idx}: Suspicious balance change in origin account")
        
        if row['type'] in ['CASH_IN', 'TRANSFER']:
            expected_new_balance = row['oldbalanceDest'] + row['amount']
            if abs(row['newbalanceDest'] - expected_new_balance) > 0.01:
                warnings.append(f"Row {idx}: Suspicious balance change in destination account")
        
        # Check for impossible scenarios
        if row['amount'] > row['oldbalanceOrg'] and row['type'] in ['CASH_OUT', 'PAYMENT', 'TRANSFER', 'DEBIT']:
            warnings.append(f"Row {idx}: Transaction amount exceeds available balance")
    
    return warnings

def preprocess_transactions(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
    """
    Enhanced preprocessing with validation and feature engineering
    Returns: (processed_df, warnings)
    """
    df = df_raw.copy()
    warnings = []
    
    # Validate business logic
    logic_warnings = validate_transaction_logic(df)
    warnings.extend(logic_warnings)
    
    # One-hot encode transaction types
    type_dummies = pd.get_dummies(df["type"], prefix="type")
    for col in ["type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"]:
        if col not in type_dummies.columns:
            type_dummies[col] = 0
    df = pd.concat([df.drop(columns=["type"]), type_dummies], axis=1)

    # Enhanced feature engineering
    df["delta_org"] = df["oldbalanceOrg"] - df["newbalanceOrg"]
    df["delta_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]
    df["zero_org_bal"] = (df["oldbalanceOrg"] == 0).astype(int)
    df["zero_dest_bal"] = (df["oldbalanceDest"] == 0).astype(int)
    
    # Safe ratio calculations with small epsilon
    epsilon = 1e-6
    df["ratio_org"] = df["amount"] / (df["oldbalanceOrg"].abs() + epsilon)
    df["ratio_dest"] = df["amount"] / (df["oldbalanceDest"].abs() + epsilon)
    
    # Network mismatch detection
    df["net_mismatch"] = ((df["delta_org"] - df["delta_dest"]).abs() > 1e-3).astype(int)
    
    # Additional engineered features
    df["amount_to_org_ratio"] = df["amount"] / (df["oldbalanceOrg"] + epsilon)
    df["high_amount_flag"] = (df["amount"] > df["amount"].quantile(0.95)).astype(int)
    
    # Ensure all expected features exist
    for col in EXPECTED_FEATURES:
        if col not in df.columns:
            df[col] = 0
            warnings.append(f"Missing feature '{col}' filled with 0")

    # Select and order features correctly
    df = df[EXPECTED_FEATURES]
    
    return df, warnings

def get_risk_level(score: float) -> str:
    """Determine risk level based on score"""
    if score < 0.3:
        return "LOW"
    elif score < 0.6:
        return "MEDIUM"
    else:
        return "HIGH"

# ========= Prediction Logging =========
def log_prediction(transaction_data: dict, prediction_result: dict):
    """Log predictions for monitoring and audit"""
    if LOG_PREDICTIONS:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "transaction": transaction_data,
            "prediction": prediction_result
        }
        logger.info(f"PREDICTION_LOG: {log_entry}")

# ========= API Routes =========
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Fraud Detection API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check with system information"""
    try:
        # Test model prediction with dummy data
        dummy_data = pd.DataFrame([{
            'step': 1, 'amount': 100, 'oldbalanceOrg': 1000, 'newbalanceOrg': 900,
            'oldbalanceDest': 0, 'newbalanceDest': 100, 'isFlaggedFraud': 0,
            'type_CASH_OUT': 0, 'type_DEBIT': 0, 'type_PAYMENT': 1, 'type_TRANSFER': 0,
            'delta_org': 100, 'delta_dest': 100, 'zero_org_bal': 0, 'zero_dest_bal': 1,
            'ratio_org': 0.1, 'ratio_dest': 100, 'net_mismatch': 0
        }])
        
        _ = MODEL.predict_proba(dummy_data)
        
        return HealthResponse(
            status="healthy",
            model_path=MODEL_PATH,
            threshold=THRESHOLD,
            uptime=str(datetime.now()),
            features_count=len(EXPECTED_FEATURES),
            version="2.0.0"
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(transaction: Transaction, background_tasks: BackgroundTasks):
    """Predict fraud for a single transaction with enhanced features"""
    try:
        # Convert to DataFrame
        raw_data = pd.DataFrame([transaction.dict()])
        
        # Preprocess
        X, warnings = preprocess_transactions(raw_data)
        
        # Predict
        probabilities = MODEL.predict_proba(X)
        score = float(probabilities[0, 1])
        flagged = bool(score >= THRESHOLD)
        risk_level = get_risk_level(score)
        timestamp = datetime.now().isoformat()
        
        # Create response
        response = PredictionResponse(
            score=score,
            flagged=flagged,
            threshold=THRESHOLD,
            risk_level=risk_level,
            timestamp=timestamp
        )
        
        # Log prediction asynchronously
        background_tasks.add_task(
            log_prediction, 
            transaction.dict(), 
            response.dict()
        )
        
        # Log warnings if any
        if warnings:
            logger.warning(f"Transaction validation warnings: {warnings}")
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch_predict", response_model=BatchResponse)
async def predict_batch(batch: BatchRequest, background_tasks: BackgroundTasks):
    """Predict fraud for multiple transactions with summary statistics"""
    try:
        # Convert to DataFrame
        raw_data = pd.DataFrame([txn.dict() for txn in batch.transactions])
        
        # Preprocess
        X, warnings = preprocess_transactions(raw_data)
        
        # Predict
        probabilities = MODEL.predict_proba(X)
        scores = probabilities[:, 1]
        flags = scores >= THRESHOLD
        timestamp = datetime.now().isoformat()
        
        # Create individual results
        results = []
        for i, (score, flagged) in enumerate(zip(scores, flags)):
            results.append(PredictionResponse(
                score=float(score),
                flagged=bool(flagged),
                threshold=THRESHOLD,
                risk_level=get_risk_level(float(score)),
                timestamp=timestamp
            ))
        
        # Create summary statistics
        summary = {
            "total_transactions": len(batch.transactions),
            "flagged_count": int(flags.sum()),
            "flagged_percentage": float(flags.mean() * 100),
            "avg_score": float(scores.mean()),
            "max_score": float(scores.max()),
            "min_score": float(scores.min()),
            "high_risk_count": sum(1 for s in scores if get_risk_level(s) == "HIGH"),
            "medium_risk_count": sum(1 for s in scores if get_risk_level(s) == "MEDIUM"),
            "low_risk_count": sum(1 for s in scores if get_risk_level(s) == "LOW"),
            "warnings": warnings
        }
        
        response = BatchResponse(
            results=results,
            summary=summary,
            threshold=THRESHOLD,
            timestamp=timestamp
        )
        
        # Log batch prediction summary
        background_tasks.add_task(
            logger.info,
            f"BATCH_PREDICTION: {len(batch.transactions)} transactions, "
            f"{int(flags.sum())} flagged ({float(flags.mean() * 100):.1f}%)"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model_info")
async def get_model_info():
    """Get detailed model information"""
    try:
        return {
            "model_path": MODEL_PATH,
            "threshold": THRESHOLD,
            "expected_features": EXPECTED_FEATURES,
            "feature_count": len(EXPECTED_FEATURES),
            "supported_transaction_types": TXN_TYPES,
            "model_type": str(type(MODEL).__name__),
            "logging_enabled": LOG_PREDICTIONS
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@app.post("/update_threshold")
async def update_threshold(new_threshold: float = Field(..., ge=0.0, le=1.0)):
    """Update the fraud detection threshold"""
    global THRESHOLD
    old_threshold = THRESHOLD
    THRESHOLD = new_threshold
    logger.info(f"Threshold updated from {old_threshold} to {new_threshold}")
    return {
        "message": "Threshold updated successfully",
        "old_threshold": old_threshold,
        "new_threshold": THRESHOLD
    }

# ========= Exception Handlers =========
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"detail": f"Invalid input: {str(exc)}"}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# ========= Startup Event =========
@app.on_event("startup")
async def startup_event():
    logger.info("Fraud Detection API started successfully")
    logger.info(f"Model: {MODEL_PATH}")
    logger.info(f"Threshold: {THRESHOLD}")
    logger.info(f"Features: {len(EXPECTED_FEATURES)}")

# ========= Development Server =========
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )