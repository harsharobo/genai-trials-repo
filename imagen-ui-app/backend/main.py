from fastapi import FastAPI, HTTPException, Header, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Annotated, Union, Dict
import base64
import io
import json
import httpx
from PIL import Image
import logging
from datetime import datetime
import os
from pathlib import Path
import uuid
import asyncio
import traceback
import mlflow
from mlflow.entities import AssessmentSource


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Image-to-Image API")

# In-memory job storage
jobs: Dict[str, Dict] = {}

#setup mlflow
experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID")
if not experiment_id:
    logger.error("MLFLOW_EXPERIMENT_ID environment variable not set")
    raise AssertionError("MLFLOW_EXPERIMENT_ID environment variable not set")

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(experiment_id=experiment_id)

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
    trace_id: str
    message: str


class FeedbackRequest(BaseModel):
    trace_id: str # trace id from the model serving response
    feedback_type: str  # "thumbs_up" or "thumbs_down"
    feedback_text: Optional[str] = None
    session_id: Optional[str] = None
    prompt: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str
    message: str


class JobStartResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    output_image: Optional[str] = None
    trace_id: Optional[str] = None
    error: Optional[str] = None
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


async def generate_image_from_endpoint(request: ImageRequest) -> tuple[str, str]:
    """
    Common function to generate image from model serving endpoint.
    Returns tuple of (base64 encoded output image, trace_id).
    """
    try:
        # Extract base64 data (remove data URI prefix if present)
        img1_b64_data = pil_to_base64(base64_to_pil(request.image1))
        img1_b64 = img1_b64_data.split(",")[1]

        img2_b64_data = pil_to_base64(base64_to_pil(request.image2))
        img2_b64 = img2_b64_data.split(",")[1]

        logger.info("Images processed successfully")

        # Prepare request payload for the model serving endpoint
        payload = {
            "dataframe_records": [{
                "image1": img1_b64,
                "image2": img2_b64,
                "prompt": request.prompt
            }]
        }

        # Prepare headers
        headers = {
            "Content-Type": "application/json"
        }

        # Get model endpoint URL from environment variable
        model_endpoint_url = get_model_endpoint_url()

        # Add authentication token if provided
        auth_token = os.getenv("MODEL_ENDPOINT_TOKEN")
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        logger.info(f"Sending request to model endpoint: {model_endpoint_url}")

        # Make async HTTP request to model serving endpoint
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                model_endpoint_url,
                json=payload,
                headers=headers
            )
            response.raise_for_status()

            result = response.json()
            logger.info("Received response from model endpoint")

            # Extract output image and trace_id from response
            predictions = result.get("predictions", [None])[0]
            if not predictions:
                raise ValueError("No output image found in model response")

            output_image_b64 = predictions.get("output_image", None)
            if not output_image_b64:
                raise ValueError("No output image found in model response")

            trace_id = predictions.get("trace_id", None)
            if not trace_id:
                raise ValueError("No trace_id found in model response")

            output_image = base64_to_pil(output_image_b64)
            output_base64 = pil_to_base64(output_image)
            logger.info(f"Inference completed successfully with trace_id: {trace_id}")

            return output_base64, trace_id

    except httpx.HTTPStatusError as http_err:
        logger.error(f"HTTP error from model endpoint: {http_err}")
        logger.error(f"Response status: {http_err.response.status_code}")
        logger.error(f"Response body: {http_err.response.text}")
        logger.error(f"Full stack trace:\n{traceback.format_exc()}")
        raise
    except httpx.TimeoutException as timeout_err:
        logger.error(f"Request to model endpoint timed out: {timeout_err}")
        logger.error(f"Full stack trace:\n{traceback.format_exc()}")
        raise
    except Exception as e:
        logger.error(f"Error in generate_image_from_endpoint: {e}")
        logger.error(f"Full stack trace:\n{traceback.format_exc()}")
        raise


async def process_image_job(job_id: str, request: ImageRequest):
    """Background task to process image generation."""
    try:
        logger.info(f"Processing job {job_id} with prompt: {request.prompt}")

        # Update job status to processing
        jobs[job_id]["status"] = "processing"

        # Use common image generation function
        output_base64, trace_id = await generate_image_from_endpoint(request)

        # Update job with result
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["output_image"] = output_base64
        jobs[job_id]["trace_id"] = trace_id
        jobs[job_id]["message"] = "Image generated successfully"
        logger.info(f"Job {job_id} completed successfully with trace_id: {trace_id}")

    except httpx.HTTPStatusError as http_err:
        logger.error(f"HTTP error from model endpoint for job {job_id}: {http_err}")
        logger.error(f"Full stack trace:\n{traceback.format_exc()}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = f"Model endpoint error: {str(http_err)}"
        jobs[job_id]["message"] = "Job failed"
    except httpx.TimeoutException as timeout_err:
        logger.error(f"Request to model endpoint timed out for job {job_id}: {timeout_err}")
        logger.error(f"Full stack trace:\n{traceback.format_exc()}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = "Model endpoint request timed out"
        jobs[job_id]["message"] = "Job failed"
    except Exception as error:
        logger.error(f"Error processing job {job_id}: {error}")
        logger.error(f"Full stack trace:\n{traceback.format_exc()}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(error)
        jobs[job_id]["message"] = "Job failed"

@app.get("/api")
async def root():
    return {"message": "Image-to-Image API is running"}


@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/api/predict/start", response_model=JobStartResponse)
async def start_prediction(request: ImageRequest, background_tasks: BackgroundTasks):
    """
    Start an image generation job and return a job ID.
    Client should poll /api/predict/status/{job_id} to get the result.
    """
    try:
        logger.info(f"Starting new prediction job with prompt: {request.prompt}")

        # Validate inputs
        if not request.image1 or not request.image2 or not request.prompt.strip():
            raise HTTPException(status_code=400, detail="Please provide both images and a prompt")

        # Generate unique job ID
        job_id = str(uuid.uuid4())

        # Initialize job in storage
        jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "output_image": None,
            "trace_id": None,
            "error": None,
            "message": "Job created",
            "created_at": datetime.now().isoformat()
        }

        # Start background task
        background_tasks.add_task(process_image_job, job_id, request)

        logger.info(f"Job {job_id} created and queued for processing")

        return JobStartResponse(
            job_id=job_id,
            status="pending",
            message="Job started successfully. Use job_id to poll for status."
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error starting prediction job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start job: {str(e)}")


@app.get("/api/predict/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Poll the status of a prediction job.
    Returns the job status and result if completed.
    Job is deleted after retrieval if status is 'completed' or 'failed'.
    """
    try:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        job = jobs[job_id]
        job_status = job["status"]

        response = JobStatusResponse(
            job_id=job_id,
            status=job_status,
            output_image=job.get("output_image"),
            trace_id=job.get("trace_id"),
            error=job.get("error"),
            message=job["message"]
        )

        # Delete job from memory if completed or failed
        if job_status in ["completed", "failed"]:
            del jobs[job_id]
            logger.info(f"Job {job_id} deleted after retrieval (status: {job_status})")

        return response

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error retrieving job status: {e}")
        logger.error(f"Full stack trace:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve job status: {str(e)}")


@app.post("/api/predict", response_model=ImageResponse)
async def predict(request: ImageRequest,
                  x_forwarded_access_token: Annotated[Union[str, None], Header(alias="X-Forwarded-Access-Token")] = None):
    """
    Process two images and a prompt using model serving endpoint via REST API.
    Returns the generated image as base64 encoded string.
    """
    try:
        logger.info(f"Received prediction request with prompt: {request.prompt}")

        # Use common image generation function
        output_base64, trace_id = await generate_image_from_endpoint(request)

        return ImageResponse(
            output_image=output_base64,
            trace_id=trace_id,
            message="Image generated successfully"
        )

    except httpx.HTTPStatusError as http_err:
        logger.error(f"HTTP error from model endpoint: {http_err}")
        logger.error(f"Full stack trace:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=http_err.response.status_code,
            detail=f"Model endpoint error: {str(http_err)}"
        )
    except httpx.TimeoutException as timeout_err:
        logger.error(f"Request to model endpoint timed out: {timeout_err}")
        logger.error(f"Full stack trace:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=504,
            detail="Model endpoint request timed out"
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        logger.error(f"Full stack trace:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/api/feedback", response_model=FeedbackResponse)
async def submit_feedback(feedback: FeedbackRequest,
                          x_forwarded_email: Annotated[Union[str, None], Header(alias="X-Forwarded-Email")] = "someone"):
    """
    Store user feedback (thumbs up/down with optional text).
    """
    try:
        logger.info(f"Received feedback: {feedback.feedback_type} for trace_id: {feedback.trace_id} from user: {x_forwarded_email}")

        # feedback_data = {
        #     "timestamp": datetime.now().isoformat(),
        #     "feedback_type": feedback.feedback_type,
        #     "feedback_text": feedback.feedback_text,
        #     "session_id": feedback.session_id,
        #     "prompt": feedback.prompt
        # }
        # Store feedback (implement your storage logic here)
        # Options: Database, file, MLflow tracking, etc.
        # with open("feedback_log.json", "a") as f:
        #     f.write(json.dumps(feedback_data) + "\n")
        mlflow.log_feedback(
            trace_id=feedback.trace_id,
            name="user_feedback",
            value=feedback.feedback_type == "thumbs_up",
            rationale=feedback.feedback_text,
            source=AssessmentSource(source_type="HUMAN",
                                    source_id=x_forwarded_email)   
        )

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
