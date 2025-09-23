import os
import joblib
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import List, Literal, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
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
MODEL_PATH = os.getenv("MODEL_PATH", "models\random_forest_model.pkl")
THRESHOLD = float(os.getenv("THRESHOLD", "0.80"))
LOG_PREDICTIONS = os.getenv("LOG_PREDICTIONS", "true").lower() == "true"
API_VERSION = "2.1.0"

# Transaction types from original dataset
TXN_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

# Feature list (must match training features)
EXPECTED_FEATURES = [
    "step", "type_PAYMENT", "type_TRANSFER", "type_DEBIT", "type_CASH_IN",
    "amount", "diff_new_old_origin"
]

# Global variables
MODEL = None
app_start_time = datetime.now()

# ========= Model Loading =========
def load_model():
    """Load the ML model with proper error handling"""
    global MODEL, THRESHOLD
    try:
        # Tải trực tiếp mô hình
        MODEL = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
        logger.info(f"Using default threshold: {THRESHOLD}")
        return True
    except FileNotFoundError:
        logger.error(f"Model file not found at {MODEL_PATH}")
        return False
    except Exception as e:
        logger.error(f"Failed to load model from {MODEL_PATH}: {str(e)}")
        return False
    
# ========= FastAPI App Setup =========
app = FastAPI(
    title="Fraud Detection API",
    description="Advanced fraud detection system for financial transactions",
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "predictions", "description": "Fraud prediction operations"},
        {"name": "health", "description": "Health check and system status"},
        {"name": "admin", "description": "Administrative operations"},
    ]
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

    @field_validator('amount', 'oldbalanceOrg', 'newbalanceOrg', 'oldbalanceDest', 'newbalanceDest')
    @classmethod
    def validate_positive_amounts(cls, v):
        if v < 0:
            raise ValueError('Financial amounts must be non-negative')
        return v

class BatchRequest(BaseModel):
    transactions: List[Transaction] = Field(..., min_length=1, max_length=1000, description="List of transactions to predict")
    
class PredictionResponse(BaseModel):
    score: float = Field(..., description="Fraud probability score (0-1)")
    flagged: bool = Field(..., description="Whether transaction is flagged as fraud")
    threshold: float = Field(..., description="Threshold used for flagging")
    risk_level: str = Field(..., description="Risk level: LOW, MEDIUM, HIGH")
    confidence: str = Field(..., description="Prediction confidence level")
    timestamp: str = Field(..., description="Prediction timestamp")
    warnings: Optional[List[str]] = Field(default=None, description="Any validation warnings")

class BatchResponse(BaseModel):
    results: List[PredictionResponse]
    summary: Dict[str, Any]
    threshold: float
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    threshold: float
    uptime: str
    features_count: int
    version: str
    system_info: Dict[str, Any]

class ThresholdUpdate(BaseModel):
    threshold: float = Field(..., ge=0.0, le=1.0, description="New threshold value between 0 and 1")

# ========= Dependency Injection =========
async def check_model_loaded():
    """Dependency to ensure model is loaded"""
    if MODEL is None:
        if not load_model():
            raise HTTPException(status_code=503, detail="Model not loaded. Service unavailable.")
    return MODEL

# ========= Enhanced Preprocessing =========
def validate_transaction_logic(df: pd.DataFrame) -> List[str]:
    """Validate business logic of transactions"""
    warnings = []
    
    for idx, row in df.iterrows():
        # Check balance consistency for different transaction types
        if row['type'] in ['CASH_OUT', 'PAYMENT', 'TRANSFER', 'DEBIT']:
            expected_new_balance = max(0, row['oldbalanceOrg'] - row['amount'])
            if abs(row['newbalanceOrg'] - expected_new_balance) > 0.01:
                warnings.append(f"Row {idx}: Suspicious balance change in origin account")
        
        if row['type'] in ['CASH_IN', 'TRANSFER']:
            expected_new_balance = row['oldbalanceDest'] + row['amount']
            if abs(row['newbalanceDest'] - expected_new_balance) > 0.01:
                warnings.append(f"Row {idx}: Suspicious balance change in destination account")
        
        # Check for impossible scenarios
        if row['amount'] > row['oldbalanceOrg'] and row['type'] in ['CASH_OUT', 'PAYMENT', 'TRANSFER', 'DEBIT']:
            if row['oldbalanceOrg'] > 0:  # Only warn if there was a balance to begin with
                warnings.append(f"Row {idx}: Transaction amount exceeds available balance")
    
    return warnings

def preprocess_transactions(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
    """
    Preprocess transactions to match model's expected features.
    Returns: (processed_df, warnings)
    """
    df = df_raw.copy()
    warnings = []

    # Tính toán đặc trưng 'diff_new_old_origin'
    df["diff_new_old_origin"] = df["newbalanceOrg"] - df["oldbalanceOrg"]

    # One-hot encode transaction types
    type_dummies = pd.get_dummies(df["type"], prefix="type")
    for col in ["type_PAYMENT", "type_TRANSFER", "type_DEBIT", "type_CASH_IN"]:
        if col not in type_dummies.columns:
            type_dummies[col] = 0
    df = pd.concat([df.drop(columns=["type"]), type_dummies], axis=1)

    # Đảm bảo tất cả các đặc trưng mong đợi đều tồn tại
    for col in EXPECTED_FEATURES:
        if col not in df.columns:
            df[col] = 0
            warnings.append(f"Missing feature '{col}' filled with 0")

    # Chỉ giữ lại các cột trong EXPECTED_FEATURES
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

def get_confidence_level(score: float, threshold: float) -> str:
    """Determine confidence level based on score and threshold"""
    distance_from_threshold = abs(score - threshold)
    if distance_from_threshold > 0.3:
        return "HIGH"
    elif distance_from_threshold > 0.15:
        return "MEDIUM"
    else:
        return "LOW"

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
@app.get("/", response_model=Dict[str, str], tags=["health"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Fraud Detection API",
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health",
        "status": "running"
    }

@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Enhanced health check with system information"""
    model_loaded = MODEL is not None
    uptime = datetime.now() - app_start_time
    
    # System information
    system_info = {
        "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "platform": os.name,
        "log_predictions": LOG_PREDICTIONS,
    }
    
    if model_loaded:
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
            status = "healthy"
            
        except Exception as e:
            logger.error(f"Health check model test failed: {str(e)}")
            status = "degraded"
            system_info["model_error"] = str(e)
    else:
        status = "unhealthy"
        system_info["error"] = "Model not loaded"
    
    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        model_path=MODEL_PATH,
        threshold=THRESHOLD,
        uptime=str(uptime),
        features_count=len(EXPECTED_FEATURES),
        version=API_VERSION,
        system_info=system_info
    )

@app.post("/predict", response_model=PredictionResponse, tags=["predictions"])
async def predict_single(
    transaction: Transaction, 
    background_tasks: BackgroundTasks,
    model = Depends(check_model_loaded)
):
    """Predict fraud for a single transaction with enhanced features"""
    try:
        # Convert to DataFrame
        raw_data = pd.DataFrame([transaction.model_dump()])
        
        # Preprocess
        X, warnings = preprocess_transactions(raw_data)
        
        # Predict
        probabilities = model.predict_proba(X)
        score = float(probabilities[0, 1])
        flagged = bool(score >= THRESHOLD)
        risk_level = get_risk_level(score)
        confidence = get_confidence_level(score, THRESHOLD)
        timestamp = datetime.now().isoformat()
        
        # Create response
        response = PredictionResponse(
            score=score,
            flagged=flagged,
            threshold=THRESHOLD,
            risk_level=risk_level,
            confidence=confidence,
            timestamp=timestamp,
            warnings=warnings if warnings else None
        )
        
        # Log prediction asynchronously
        background_tasks.add_task(
            log_prediction, 
            transaction.model_dump(), 
            response.model_dump()
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch_predict", response_model=BatchResponse, tags=["predictions"])
async def predict_batch(
    batch: BatchRequest, 
    background_tasks: BackgroundTasks,
    model = Depends(check_model_loaded)
):
    """Predict fraud for multiple transactions with summary statistics"""
    try:
        # Convert to DataFrame
        raw_data = pd.DataFrame([txn.model_dump() for txn in batch.transactions])
        
        # Preprocess
        X, warnings = preprocess_transactions(raw_data)
        
        # Predict
        probabilities = model.predict_proba(X)
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
                confidence=get_confidence_level(float(score), THRESHOLD),
                timestamp=timestamp,
                warnings=None
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
            "warnings_count": len(warnings),
            "processing_time_ms": None  # Could add timing if needed
        }
        
        if warnings:
            summary["warnings"] = warnings
        
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

@app.get("/model_info", tags=["admin"])
async def get_model_info():
    """Get detailed model information"""
    try:
        model_info = {
            "model_path": MODEL_PATH,
            "model_loaded": MODEL is not None,
            "threshold": THRESHOLD,
            "expected_features": EXPECTED_FEATURES,
            "feature_count": len(EXPECTED_FEATURES),
            "supported_transaction_types": TXN_TYPES,
            "logging_enabled": LOG_PREDICTIONS,
            "api_version": API_VERSION
        }
        
        if MODEL is not None:
            model_info["model_type"] = str(type(MODEL).__name__)
            # Add more model-specific info if available
            if hasattr(MODEL, 'feature_importances_'):
                model_info["has_feature_importances"] = True
            if hasattr(MODEL, 'n_estimators'):
                model_info["n_estimators"] = MODEL.n_estimators
                
        return model_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@app.post("/update_threshold", tags=["admin"])
async def update_threshold(threshold_update: ThresholdUpdate):
    """Update the fraud detection threshold"""
    global THRESHOLD
    old_threshold = THRESHOLD
    THRESHOLD = threshold_update.threshold
    
    logger.info(f"Threshold updated from {old_threshold} to {THRESHOLD}")
    
    return {
        "message": "Threshold updated successfully",
        "old_threshold": old_threshold,
        "new_threshold": THRESHOLD,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/reload_model", tags=["admin"])
async def reload_model():
    """Reload the model from disk"""
    try:
        success = load_model()
        if success:
            return {
                "message": "Model reloaded successfully",
                "model_path": MODEL_PATH,
                "threshold": THRESHOLD,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to reload model")
    except Exception as e:
        logger.error(f"Model reload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")

# ========= Exception Handlers =========
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={
            "detail": f"Invalid input: {str(exc)}",
            "type": "ValueError",
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "type": "InternalServerError",
            "timestamp": datetime.now().isoformat()
        }
    )

# ========= Startup/Shutdown Events =========
@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info(f"Fraud Detection API {API_VERSION} starting up...")
    
    # Try to load model on startup
    if load_model():
        logger.info("Model loaded successfully on startup")
    else:
        logger.warning("Model failed to load on startup - will try on first request")
    
    logger.info(f"API started successfully on {datetime.now()}")
    logger.info(f"Expected features: {len(EXPECTED_FEATURES)}")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    logger.info("Fraud Detection API shutting down...")

# ========= Development Server =========
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )