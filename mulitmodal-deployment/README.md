# Multimodal Model Deployment

This repository contains MLflow-based deployment configurations for various multimodal models on Databricks, including image editing, vision-language models, and audio transcription.

## Architecture

```
Databricks Asset Bundles
    ↓
Multimodal Model Deployment
    ├── Image-to-Image (Qwen Image Edit)
    ├── Image-Text-to-Text (Vision Language Models)
    └── Audio-to-Text (Whisper)
    ↓
MLflow Model Registry
    ↓
Databricks Model Serving Endpoints
```

## Project Structure

```
mulitmodal-deployment/
├── databricks.yml                 # Main bundle configuration
├── resources/
│   └── qwen-image-edit-model-deployment.yml  # Job definitions
├── image-image/                   # Image editing models
│   ├── imagen_model.py           # Qwen Image Edit MLflow wrapper
│   ├── image-model-register.ipynb
│   ├── image-model-deployment.ipynb
│   ├── inference_config.yml
│   ├── requirements.txt
│   └── sample_images/
├── image-text-to-text/           # Vision-language models
│   ├── vlm_model.py              # VLM MLflow wrapper (Llama/Gemma)
│   ├── vlm_model_deployment.ipynb
│   └── inference_config.yml
└── audio-text/                   # Audio transcription
    └── faster_whisper_mlflow_model.ipynb
```

## Models Supported

### 1. Image-to-Image (Qwen Image Edit)
- **Model**: Qwen/Qwen-Image-Edit-2509
- **Purpose**: Image editing using dual image input and text prompts
- **Input**: Two base64-encoded images + text prompt
- **Output**: Base64-encoded edited image
- **Hardware**: GPU (A100 recommended)

### 2. Image-Text-to-Text (Vision Language Models)
- **Models Supported**:
  - Meta Llama Vision models (MllamaForConditionalGeneration)
  - Google Gemma 3 Vision models (Gemma3ForConditionalGeneration)
- **Purpose**: Visual question answering, image understanding
- **Input**: Base64-encoded image + text prompt
- **Output**: Text response
- **Hardware**: GPU with optional quantization (4-bit/8-bit)

### 3. Audio-to-Text (Faster Whisper)
- **Model**: OpenAI Whisper variants
- **Purpose**: Audio transcription and speech recognition
- **Input**: Audio file
- **Output**: Transcribed text

## Setup

### Prerequisites

1. **Databricks Workspace**:
   - Azure Databricks workspace
   - Unity Catalog enabled
   - GPU-enabled clusters (for model deployment)

2. **Required Tools**:
   - Databricks CLI
   - Python 3.8+
   - Git

3. **Databricks CLI Authentication**:
```bash
databricks configure --profile adb-demo
```

### Configuration

1. **Update databricks.yml**:
   - Set your workspace host
   - Update catalog name (default: `uc_sriharsha_jana`)
   - Update schema name (default: `default`)
   - Set notification email

2. **Environment Variables**:
```bash
export DATABRICKS_HOST=<your-workspace-url>
export DATABRICKS_TOKEN=<your-token>
```

## Deployment

### Deploy with Databricks Asset Bundles

1. **Deploy the bundle**:
```bash
databricks bundle deploy --target dev --profile adb-demo
```

2. **Run the deployment job**:
```bash
databricks bundle run qwen-image-edit-deploy-job --target dev --profile adb-demo
```

### Manual Deployment Steps

#### Image-to-Image Model (Qwen)

1. **Register the model**:
   - Open [image-image/image-model-register.ipynb](image-image/image-model-register.ipynb)
   - Run all cells to download and register the model
   - Parameters:
     - `hf_model_id`: Qwen/Qwen-Image-Edit-2509
     - `model_name`: qwen_image_edit_model
     - `model_alias`: staging

2. **Deploy to serving endpoint**:
   - Open [image-image/image-model-deployment.ipynb](image-image/image-model-deployment.ipynb)
   - Run all cells to create serving endpoint
   - Parameters:
     - `endpoint_name`: qwen_image_edit_model
     - Workload size: Small (GPU)

#### Vision Language Model

1. **Deploy VLM model**:
   - Open [image-text-to-text/vlm_model_deployment.ipynb](image-text-to-text/vlm_model_deployment.ipynb)
   - Configure model type in [inference_config.yml](image-text-to-text/inference_config.yml):
     - Set `model_type` to `llama` or `gemma`
     - Enable quantization if needed
   - Run deployment notebook

#### Audio Transcription Model

1. **Deploy Whisper model**:
   - Open [audio-text/faster_whisper_mlflow_model.ipynb](audio-text/faster_whisper_mlflow_model.ipynb)
   - Run all cells to register and deploy

## Model Configuration

### Image Edit Model (inference_config.yml)

```yaml
model_type: qwen_image_edit
use_quantization: false
```

### Vision Language Model (inference_config.yml)

```yaml
model_type: gemma  # or llama
quant_config:
  bnb_4bit_quant_type: nf4
  load_in_4bit: true
  load_in_8bit: false
use_quantization: false  # Set to true to enable quantization
```

## API Usage

### Image Edit Endpoint

**Request Format**:
```json
{
  "dataframe_records": [{
    "image1": "<base64_encoded_image_1>",
    "image2": "<base64_encoded_image_2>",
    "prompt": "Replace the background with a sunset scene"
  }],
  "params": {
    "num_inference_steps": 40,
    "true_cfg_scale": 4.0,
    "guidance_scale": 1.0,
    "negative_prompt": "",
    "num_images_per_prompt": 1,
    "seed": 0
  }
}
```

**Response Format**:
```json
{
  "predictions": [{
    "output_image": "<base64_encoded_result>",
    "trace_id": "<mlflow_trace_id>"
  }]
}
```

### Vision Language Model Endpoint

**Request Format**:
```json
{
  "dataframe_records": [{
    "image": "<base64_encoded_image>",
    "user_prompt": "What objects are in this image?"
  }],
  "params": {
    "max_new_tokens": 200,
    "temperature": 0.1,
    "top_p": 0.9
  }
}
```

**Response Format**:
```json
{
  "predictions": [{
    "output_text": "The image contains a cat, a book, and a lamp."
  }]
}
```

## Model Parameters

### Image Edit Parameters
- `num_inference_steps` (default: 40): Number of denoising steps
- `true_cfg_scale` (default: 4.0): Classifier-free guidance scale
- `guidance_scale` (default: 1.0): Additional guidance scale
- `negative_prompt` (default: " "): Negative prompt for generation
- `num_images_per_prompt` (default: 1): Number of images to generate
- `seed` (default: 0): Random seed for reproducibility

### VLM Parameters
- `max_new_tokens` (default: 200): Maximum tokens to generate
- `temperature` (default: 0.1): Sampling temperature
- `top_p` (default: 0.9): Nucleus sampling parameter

## Hardware Requirements

### Image Edit Model
- **Minimum**: 1x NVIDIA A100 (40GB)
- **Recommended**: 1x NVIDIA A100 (80GB)

### Vision Language Models
- **Without Quantization**: 1-2x NVIDIA A100
- **With 4-bit Quantization**: 1x NVIDIA A10 or better

### Audio Models
- **CPU**: Sufficient for small models
- **GPU**: Recommended for large Whisper models

## Databricks Bundle Targets

### Development (dev)
- Mode: development
- Resources prefixed with `[dev <username>]`
- Jobs paused by default
- Default target

### Production (prod)
- Mode: production
- Strict verification enabled
- Fixed root path
- Runs as specified service principal/user

## MLflow Tracing

All models support MLflow tracing for observability:
- Request/response logging
- Performance metrics
- Token usage tracking (for text models)
- Image processing steps (for vision models)

Access traces in the Databricks MLflow UI under the experiment specified in the deployment.

## Monitoring and Management

### Check Endpoint Status
```bash
databricks serving-endpoints get <endpoint-name>
```

### View Logs
```bash
databricks serving-endpoints logs <endpoint-name>
```

### Update Endpoint
Re-run the deployment notebook or use:
```bash
databricks serving-endpoints update-config <endpoint-name> --config-file config.json
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**:
   - Enable quantization in inference_config.yml
   - Use larger GPU instances
   - Reduce batch size

2. **Slow Inference**:
   - Increase `num_inference_steps` gradually
   - Check GPU utilization
   - Consider model optimization

3. **Model Loading Failures**:
   - Verify Hugging Face model ID
   - Check network connectivity
   - Ensure sufficient disk space

4. **Endpoint Deployment Failures**:
   - Verify cluster configuration
   - Check Unity Catalog permissions
   - Review MLflow experiment settings

## Dependencies

### Python Libraries
- `mlflow[databricks]==3.8.0`
- `transformers`
- `diffusers`
- `accelerate`
- `torch`
- `pandas`
- `pillow`
- `bitsandbytes` (for quantization)

## Security

- Model endpoint tokens stored as Databricks secrets
- Unity Catalog for governance
- Service principal authentication for production
- Network isolation via Databricks workspace

## Additional Resources

- [Databricks Asset Bundles Documentation](https://docs.databricks.com/dev-tools/bundles/)
- [MLflow Model Serving](https://docs.databricks.com/machine-learning/model-serving/)
- [Unity Catalog](https://docs.databricks.com/data-governance/unity-catalog/)
- [Qwen Image Edit Model](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)

## License

Refer to individual model licenses on Hugging Face.

## Support

For issues or questions:
- Email: ${var.notification_email}
- File issues in your repository's issue tracker
