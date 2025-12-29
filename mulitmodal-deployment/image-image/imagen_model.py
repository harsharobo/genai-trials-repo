import logging
import base64
import io
import pandas as pd
from PIL import Image
import torch

import mlflow
from mlflow.pyfunc import PythonModel

from diffusers import QwenImageEditPlusPipeline

# Set up logging
# logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

@mlflow.trace()
def base64_string_to_pillow_image(base64_str):
    return Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))

def pillow_image_to_base64_string(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


class ImageEditModel(PythonModel):

  def __init__(self, model_config=None):
    if model_config:
      self.model_config = model_config
    else:
      self.model_config = mlflow.models.ModelConfig(development_config="inference_config.yml")

  def load_context(self, context):
    model_id = context.artifacts['model_path']

    device_count = torch.cuda.device_count()
    device = "cuda" if device_count > 0 else "cpu"
    logger.warning("total devices found in system: {} and device: {}".format(device_count, device))

    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype
    )
    self.pipeline.to(device)
    self.pipeline.set_progress_bar_config(disable=True)

    logger.warning("Qwen Image Edit pipeline loaded successfully")

  @mlflow.trace()
  def predict(self, context, model_input, params):
    input_df = model_input.iloc[0].to_dict()

    # Extract images from base64 strings
    image1 = base64_string_to_pillow_image(input_df['image1'])
    image2 = base64_string_to_pillow_image(input_df['image2'])
    prompt = input_df['prompt']

    # Get generation parameters
    num_inference_steps = params.get("num_inference_steps", 40)
    true_cfg_scale = params.get("true_cfg_scale", 4.0)
    guidance_scale = params.get("guidance_scale", 1.0)
    negative_prompt = params.get("negative_prompt", " ")
    num_images_per_prompt = params.get("num_images_per_prompt", 1)
    seed = params.get("seed", 0)

    inputs = {
        "image": [image1, image2],
        "prompt": prompt,
        "generator": torch.manual_seed(seed),
        "true_cfg_scale": true_cfg_scale,
        "negative_prompt": negative_prompt,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "num_images_per_prompt": num_images_per_prompt,
    }

    with torch.inference_mode():
      output = self.pipeline(**inputs)
      output_image = output.images[0]

    # Convert output image to base64
    output_image_base64 = pillow_image_to_base64_string(output_image)

    torch.cuda.empty_cache()

    return pd.DataFrame().from_dict({"output_image": [output_image_base64]})

mlflow.models.set_model(ImageEditModel())
