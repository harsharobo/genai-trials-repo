from fastapi import FastAPI, HTTPException, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Annotated, Union
import base64
import io
import json
import httpx
from PIL import Image
import logging
from datetime import datetime
import os
from pathlib import Path


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Image-to-Image API")

# Get environment (development or production)
# env = os.getenv("ENV", "development")

# CORS middleware - only allow localhost in development
# allowed_origins = ["http://localhost:3000", "http://localhost:5173"] if env == "development" else []

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=allowed_origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

class ImageRequest(BaseModel):
    image1: str  # base64 encoded
    image2: str  # base64 encoded
    prompt: str
    num_inference_steps: Optional[int] = 40
    true_cfg_scale: Optional[float] = 4.0
    guidance_scale: Optional[float] = 1.0


class ImageResponse(BaseModel):
    output_image: str  # base64 encoded
    message: str


class FeedbackRequest(BaseModel):
    feedback_type: str  # "thumbs_up" or "thumbs_down"
    feedback_text: Optional[str] = None
    session_id: Optional[str] = None
    prompt: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str
    message: str


def base64_to_pil(base64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image."""
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]

        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return image
    except Exception as e:
        logger.error(f"Error converting base64 to PIL: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")


def pil_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    try:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        logger.error(f"Error converting PIL to base64: {e}")
        raise HTTPException(status_code=500, detail=f"Error encoding image: {str(e)}")

def get_model_endpoint_url() -> str:
    """Retrieve model endpoint URL from environment variable."""
    host = os.environ.get("DATABRICKS_HOST")
    model_endpoint_url = os.getenv("MODEL_ENDPOINT_URL")
    if not model_endpoint_url or not host:
        logger.error("MODEL_ENDPOINT_URL and DATABRICKS_HOST environment variables not set")
        raise HTTPException(
            status_code=500,
            detail="MODEL_ENDPOINT_URL and DATABRICKS_HOST environment variables not set"
        )
    endpoint_url =f"https://{host}/serving-endpoints/{model_endpoint_url}/invocations"
    return endpoint_url

@app.get("/api")
async def root():
    return {"message": "Image-to-Image API is running"}


@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/api/predict", response_model=ImageResponse)
async def predict(request: ImageRequest,
                  x_forwarded_access_token: Annotated[Union[str, None], Header(alias="X-Forwarded-Access-Token")] = None):
    """
    Process two images and a prompt using model serving endpoint via REST API.
    Returns the generated image as base64 encoded string.
    """
    try:
        logger.info(f"Received prediction request with prompt: {request.prompt}")

        # Extract base64 data (remove data URI prefix if present)
        img1_b64_data = pil_to_base64(base64_to_pil(request.image1))
        img1_b64 = img1_b64_data.split(",")[1] 
        
        img2_b64_data = pil_to_base64(base64_to_pil(request.image2))
        img2_b64 = img2_b64_data.split(",")[1] 

        logger.info(f"Images received successfully")

        # Prepare request payload for the model serving endpoint
        payload = {
            "dataframe_records": [{
                "image1": img1_b64,
                "image2": img2_b64,
                "prompt": request.prompt
            }]
        }

        # Optional: Add authentication headers if required
        headers = {
            "Content-Type": "application/json"
        }

        # Get model endpoint URL from environment variable
        model_endpoint_url = get_model_endpoint_url()
        # Add authentication token if provided
        auth_token = os.getenv("MODEL_ENDPOINT_TOKEN")
        if x_forwarded_access_token:
            headers["Authorization"] = f"Bearer {x_forwarded_access_token}"
        elif auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        logger.info(f"Sending request to model endpoint: {model_endpoint_url}")

        # Make async HTTP request to model serving endpoint
        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                response = await client.post(
                    model_endpoint_url,
                    json=payload,
                    headers=headers
                )
                response.raise_for_status()

                result = response.json()
                logger.info("Received response from model endpoint")

                # Extract output image from response
                # Adjust based on your model endpoint's response format
                predictions = result.get("predictions", [None])[0]
                if not predictions:
                    raise ValueError("No output image found in model response")

                output_image_b64 = predictions.get("output_image", None)
                if not output_image_b64:
                    raise ValueError("No output image found in model response")
                
                output_image = base64_to_pil(output_image_b64)
                output_base64 = pil_to_base64(output_image)
                logger.info("Inference completed successfully")

                return ImageResponse(
                    output_image=output_base64,
                    message="Image generated successfully"
                )

            except httpx.HTTPStatusError as http_err:
                logger.error(f"HTTP error from model endpoint: {http_err}")
                raise HTTPException(
                    status_code=http_err.response.status_code,
                    detail=f"Model endpoint error: {str(http_err)}"
                )
            except httpx.TimeoutException:
                logger.error("Request to model endpoint timed out")
                raise HTTPException(
                    status_code=504,
                    detail="Model endpoint request timed out"
                )
            except Exception as model_error:
                logger.error(f"Model inference error: {model_error}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Model inference failed: {str(model_error)}"
                )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/api/feedback", response_model=FeedbackResponse)
async def submit_feedback(feedback: FeedbackRequest):
    """
    Store user feedback (thumbs up/down with optional text).
    """
    try:
        logger.info(f"Received feedback: {feedback.feedback_type}")

        feedback_data = {
            "timestamp": datetime.now().isoformat(),
            "feedback_type": feedback.feedback_type,
            "feedback_text": feedback.feedback_text,
            "session_id": feedback.session_id,
            "prompt": feedback.prompt
        }

        # Store feedback (implement your storage logic here)
        # Options: Database, file, MLflow tracking, etc.
        with open("feedback_log.json", "a") as f:
            f.write(json.dumps(feedback_data) + "\n")

        logger.info(f"Feedback stored successfully")

        return FeedbackResponse(
            status="success",
            message="Thank you for your feedback!"
        )

    except Exception as e:
        logger.error(f"Feedback submission error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to submit feedback: {str(e)}"
        )

# --- Static Files Setup ---
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)

app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

# --- Catch-all for React Routes ---
@app.get("/{full_path:path}")
async def serve_react(full_path: str):
    index_html = os.path.join(static_dir, "index.html")
    if os.path.exists(index_html):
        logger.info(f"Serving React frontend for path: /{full_path}")
        return FileResponse(index_html)
    logger.error("Frontend not built. index.html missing.")
    raise HTTPException(
        status_code=404,
        detail="Frontend not built. Please run 'npm run build' first."
    )
