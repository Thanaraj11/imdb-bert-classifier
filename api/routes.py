"""API route handlers for sentiment prediction"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List
import time
from api.schemas import PredictRequest, PredictResponse, BatchRequest, BatchResponse
from evaluation.predict import SentimentPredictor
import logging

logger = logging.getLogger(__name__)

# Global predictor instance (initialize in main.py)
predictor = None

def get_predictor():
    """Dependency injection for predictor"""
    global predictor
    if predictor is None:
        predictor = SentimentPredictor()
    return predictor

router = APIRouter(prefix="/api/v1", tags=["sentiment"])

@router.get("/health")
async def health_check(predictor_model=Depends(get_predictor)):
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": predictor_model is not None,
        "device": str(predictor_model.device) if predictor_model else "unloaded"
    }

@router.post("/predict", response_model=PredictResponse)
async def predict_sentiment(
    request: PredictRequest,
    predictor_model=Depends(get_predictor)
):
    """
    Predict sentiment for a single movie review
    
    - **text**: Movie review text (1-5000 characters)
    """
    start_time = time.time()
    
    try:
        result = predictor_model.predict(request.text)
        latency_ms = (time.time() - start_time) * 1000
        
        return PredictResponse(
            label=result["label"],
            confidence=result["confidence"],
            probabilities=result.get("raw_probabilities"),
            latency_ms=round(latency_ms, 2)
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/predict-batch", response_model=BatchResponse)
async def predict_batch_sentiment(
    request: BatchRequest,
    predictor_model=Depends(get_predictor)
):
    """
    Predict sentiment for multiple movie reviews
    
    - **texts**: List of movie review texts (max 100)
    """
    start_time = time.time()
    
    try:
        results = predictor_model.predict_batch(request.texts)
        latency_ms = (time.time() - start_time) * 1000
        
        predictions = [
            PredictResponse(
                label=r["label"],
                confidence=r["confidence"],
                probabilities=r.get("raw_probabilities"),
                latency_ms=round(latency_ms / len(request.texts), 2)
            )
            for r in results
        ]
        
        return BatchResponse(
            predictions=predictions,
            total_time_ms=round(latency_ms, 2)
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@router.get("/metrics")
async def get_metrics(predictor_model=Depends(get_predictor)):
    """Return model performance metrics"""
    return {
        "model_name": "bert-base-uncased",
        "num_labels": 2,
        "max_length": 256,
        "device": str(predictor_model.device) if predictor_model else "unloaded",
        "status": "ready"
    }