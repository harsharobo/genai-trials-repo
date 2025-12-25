# Image-to-Image Editor Application

A full-stack application for image-to-image editing using the Qwen Image Edit model, with React frontend and FastAPI backend integrated with MLflow model serving.

## Features

- **Dual Image Upload**: Upload two images via drag-and-drop or file selection
- **Text Prompt Input**: Describe how you want the images to be combined
- **AI-Powered Generation**: Uses Qwen Image Edit Plus model for image generation
- **Image Preview & Download**: View and download the generated images
- **Feedback System**: Provide thumbs up/down feedback with optional comments
- **Base64 Image Encoding**: Efficient image transfer between frontend and backend

## Architecture

```
├── main.py                 # FastAPI backend - API endpoints and MLflow integration
├── requirements.txt        # Python dependencies
└── frontend/               # React frontend
    ├── src/
    │   ├── components/     # React components
    │   │   ├── ImageUploader.jsx
    │   │   ├── OutputDisplay.jsx
    │   │   └── FeedbackModal.jsx
    │   ├── App.jsx         # Main application component
    │   └── main.jsx        # Application entry point
    └── package.json        # Node dependencies
```

## Prerequisites

- Python 3.8+ (Python 3.10+ recommended)
- Node.js 16+
- pip 21.0+
- CUDA-capable GPU (recommended) or CPU
- MLflow (for model deployment)

## Dependencies & Versioning

### Core Dependencies

- **FastAPI** `>=0.104.0,<0.105.0` - Web framework
- **Uvicorn** `>=0.24.0,<0.25.0` - ASGI server
- **Pydantic** `>=2.5.0,<3.0.0` - Data validation
- **Pillow** `>=10.1.0,<11.0.0` - Image processing
- **MLflow** `>=2.9.0,<3.0.0` - Model serving integration

See [requirements.txt](requirements.txt) for the complete dependency list.

## Setup Instructions

### Development Mode

In development mode, the frontend runs on port 3000 (Vite dev server) and proxies API calls to the backend on port 8000.

#### Backend Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. **Configure Model Endpoint** (Important):
   - Set the `MODEL_ENDPOINT_URL` environment variable to point to your model serving endpoint
   - Optionally set `MODEL_ENDPOINT_TOKEN` for authentication

```bash
export MODEL_ENDPOINT_URL="http://your-model-endpoint:port/invocations"
export MODEL_ENDPOINT_TOKEN="your-token-here"  # optional
```

4. Run the backend server:
```bash
python main.py
```

The backend will start on `http://localhost:8000`

#### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The frontend will start on `http://localhost:3000`

### Production Mode

In production mode, the backend serves both the API and the static frontend files from a single server on port 8000.

#### Build and Deploy

1. Build the frontend:
```bash
cd frontend
npm run build
```

This will create optimized static files in `static/`

2. Set environment variables for production:
```bash
export ENV=production
export MODEL_ENDPOINT_URL="http://your-model-endpoint:port/invocations"
export MODEL_ENDPOINT_TOKEN="your-token-here"  # optional
```

3. Run the backend server:
```bash
python main.py
```

4. Access the application at `http://localhost:8000`

The backend will serve:
- Static files (React app) at `/`
- API endpoints at `/api/*`

## Usage

1. **Upload Images**:
   - Click or drag-and-drop two images into the upload areas
   - Supported formats: PNG, JPG, JPEG

2. **Enter Prompt**:
   - Describe how you want the images to be combined
   - Example: "The magician emperor bear is standing in front of castle with a diamond topped scepter in his hand"

3. **Generate Image**:
   - Click "Generate Image" button
   - Wait for the model to process (may take 30-60 seconds depending on GPU)

4. **View Results**:
   - Generated image will be displayed
   - Download the image using the "Download" button

5. **Provide Feedback**:
   - Click "Provide Feedback" button
   - Select thumbs up or thumbs down
   - Optionally add comments
   - Submit feedback

## API Endpoints

### Backend API

All API endpoints are prefixed with `/api` in production mode.

#### `GET /api`
Health check endpoint

#### `GET /api/health`
Server health status

#### `POST /api/predict`
Generate image from two input images and a prompt

**Request Body:**
```json
{
  "image1": "base64_encoded_image",
  "image2": "base64_encoded_image",
  "prompt": "text description",
  "num_inference_steps": 40,
  "true_cfg_scale": 4.0,
  "guidance_scale": 1.0
}
```

**Response:**
```json
{
  "output_image": "base64_encoded_output_image",
  "message": "Image generated successfully"
}
```

#### `POST /api/feedback`
Submit user feedback

**Request Body:**
```json
{
  "feedback_type": "thumbs_up" | "thumbs_down",
  "feedback_text": "optional comment",
  "session_id": "unique_session_id",
  "prompt": "original_prompt"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Thank you for your feedback!"
}
```

## MLflow Integration

### Current Implementation

The backend currently loads the model directly for demonstration purposes. To integrate with MLflow serving:

1. **Register Your Model** in MLflow:
```python
import mlflow

# Log and register your model
with mlflow.start_run():
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=YourModelWrapper(),
        registered_model_name="image-edit-model"
    )
```

2. **Deploy to MLflow Serving**:
   - Use Databricks Model Serving
   - Or deploy using `mlflow models serve`

3. **Update Backend Code**:
   The backend in [main.py](main.py) is already configured to use a model serving endpoint via the `MODEL_ENDPOINT_URL` environment variable:

```python
# Set environment variable
export MODEL_ENDPOINT_URL="http://your-mlflow-endpoint:port/invocations"
export MODEL_ENDPOINT_TOKEN="your-token-here"  # if authentication is required
```

## Configuration

### Backend Configuration

- **Port**: Default 8000 (change in [main.py](main.py))
- **CORS**:
  - Development: Allows localhost:3000 and localhost:5173
  - Production: No CORS needed (same origin)
- **Model Parameters**: Configurable in request body
- **Environment Variables**:
  - `ENV`: Set to `production` for production mode
  - `MODEL_ENDPOINT_URL`: Model serving endpoint URL (required)
  - `MODEL_ENDPOINT_TOKEN`: Authentication token for model endpoint (optional)

### Frontend Configuration

- **Port**: Default 3000 for dev server (change in [frontend/vite.config.js](frontend/vite.config.js))
- **API Proxy**: Development mode proxies `/api` to `http://localhost:8000`
- **Build Output**: `static/` directory

## Feedback Storage

Feedback is currently stored in `feedback_log.json` in the root directory. For production:

- Integrate with a database (PostgreSQL, MongoDB, etc.)
- Use MLflow tracking to log feedback
- Implement analytics dashboard

## Troubleshooting

### Backend Issues

1. **CUDA Out of Memory**:
   - Reduce `num_inference_steps`
   - Use CPU instead of GPU (change device to "cpu")
   - Process smaller images

2. **Model Loading Fails**:
   - Check internet connection for model download
   - Verify Hugging Face cache directory permissions
   - Ensure sufficient disk space

### Frontend Issues

1. **CORS Errors (Development)**:
   - Verify backend is running on port 8000
   - Check CORS configuration in [main.py](main.py)
   - Ensure Vite proxy is configured in [frontend/vite.config.js](frontend/vite.config.js)

2. **Image Upload Fails**:
   - Check file size (max 10MB recommended)
   - Verify image format (PNG, JPG)

3. **404 Errors on Page Refresh (Production)**:
   - Ensure static files are built (`npm run build`)
   - Verify `static/` directory exists
   - Check that StaticFiles is mounted with `html=True`

## Production Deployment

### Single Server Deployment (Recommended)

The application is configured to serve both frontend and backend from a single server in production.

1. Build the frontend:
```bash
cd frontend
npm run build
```

2. Set environment variable:
```bash
export ENV=production
```

3. Run with production ASGI server:
```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000
```

The application will be available at `http://your-server:8000`

### Separate Server Deployment (Alternative)

If you prefer to deploy frontend and backend separately:

#### Frontend
1. Build production bundle:
```bash
npm run build
```

2. Serve with nginx or similar web server from `static/`

3. Configure nginx to proxy `/api/*` requests to backend

#### Backend
1. Use production ASGI server:
```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
```

2. Set up environment variables for sensitive data
3. Implement proper logging and monitoring
4. Use MLflow model serving endpoint

## License

This project is for demonstration purposes.

## Support

For issues or questions, please check the code comments or contact the development team.
