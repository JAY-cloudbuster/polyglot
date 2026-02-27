"""
FastAPI application for Polyglot Ghost AI Inference Service.
Exposes /predict endpoint for audio deepfake detection.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from inference import load_model, predict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    logger.info("Loading ML model...")
    try:
        load_model()
        logger.info("Model loaded successfully — AI service ready")
    except FileNotFoundError as e:
        logger.error(f"Model loading failed: {e}")
        raise
    yield
    logger.info("AI service shutting down")


app = FastAPI(
    title="Polyglot Ghost AI Service",
    description="Real-time voice deepfake detection inference API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check."""
    return {"service": "polyglot-ghost-ai", "status": "running"}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Predict whether an audio file contains real or deepfake speech.

    Accepts: multipart/form-data with audio file
    Returns: { label, probability, features_used, total_features }
    """
    # Validate file type
    allowed_types = [
        "audio/wav", "audio/x-wav", "audio/wave",
        "audio/mpeg", "audio/mp3", "audio/ogg",
        "audio/webm", "audio/flac",
        "application/octet-stream",  # fallback for browser recordings
    ]

    content_type = file.content_type or "application/octet-stream"
    logger.info(f"Received file: {file.filename}, type: {content_type}")

    # Read audio bytes
    try:
        audio_bytes = await file.read()
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {e}")
        raise HTTPException(status_code=400, detail="Failed to read audio file")

    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")

    if len(audio_bytes) > 10 * 1024 * 1024:  # 10 MB limit
        raise HTTPException(status_code=413, detail="Audio file too large (max 10 MB)")

    # Run inference
    try:
        result = predict(audio_bytes)
        return result
    except ValueError as e:
        logger.error(f"Audio processing error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Internal inference error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
