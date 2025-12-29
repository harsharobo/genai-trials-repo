# GenAI Trials - Multimodal AI on Databricks

Experimental repository for deploying multimodal AI models (image, text, audio) on Databricks with interactive web interfaces.

## Main Components

### 1. **Imagen UI App** (`/imagen-ui-app`)
Full-stack web application for AI-powered image editing using Qwen Image Edit Plus model.

- **Frontend** (`/frontend`): React + TypeScript app with image upload, prompt input, and result display
- **Backend** (`/backend`): FastAPI server with async job processing, connects to Databricks model serving endpoints
- **Deployment**: Databricks Apps with asset bundles

### 2. **Multimodal Deployment** (`/mulitmodal-deployment`)
ML model infrastructure and deployment pipelines.

- **Image-to-Image** (`/image-image`): Qwen Image Edit model (MLflow wrapper, registration & deployment notebooks)
- **Vision-Language** (`/image-text-to-text`): VLM model implementation
- **Audio** (`/audio-text`): Faster Whisper transcription model
- **Resources**: Databricks job definitions for automated model deployment

## Tech Stack

**Frontend**: React, TypeScript, Vite, Axios
**Backend**: FastAPI, Uvicorn, Pillow, httpx
**ML**: MLflow, HuggingFace Diffusers, PyTorch
**Platform**: Databricks (GPU clusters, model serving, Apps)

## Key Features

- Async job processing for long-running model inference
- Base64 image encoding/decoding
- User feedback collection and logging
- MLflow experiment tracking
- Infrastructure as code with Databricks bundles
