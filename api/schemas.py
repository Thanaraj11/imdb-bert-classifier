"""Pydantic schemas for API request/response validation"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict

class PredictRequest(BaseModel):
    """Single prediction request"""
    text: str = Field(
        ..., 
        min_length=1, 
        max_length=5000,
        description="Movie review text to analyze"
    )
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty or whitespace only')
        return v.strip()

class PredictResponse(BaseModel):
    """Single prediction response"""
    label: str = Field(..., description="Sentiment label: positive or negative")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score between 0 and 1")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Raw probability distribution")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")

class BatchRequest(BaseModel):
    """Batch prediction request"""
    texts: List[str] = Field(
        ..., 
    max_items=100,
        description="List of movie reviews (max 100)"
    )
    
    @validator('texts')
    def validate_texts(cls, v):
        if not v:
            raise ValueError('Texts list cannot be empty')
        for i, text in enumerate(v):
            if not text.strip():
                raise ValueError(f'Text at index {i} is empty or whitespace only')
        return [text.strip() for text in v]

class BatchResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictResponse] = Field(..., description="List of predictions")
    total_time_ms: float = Field(..., description="Total processing time in milliseconds")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    device: str
    version: str = "1.0.0"