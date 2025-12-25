import logging
import base64
import io
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch

import mlflow
from mlflow.pyfunc import PythonModel

from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, pipeline, BitsAndBytesConfig
from transformers import Gemma3ForConditionalGeneration, MllamaForConditionalGeneration

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
  
def base64_string_to_pillow_image(base64_str):
    return Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))
  
class ModelFactory:

  @staticmethod
  def load_model(model_type, model_id, device_map="auto", torch_dtype="auto", quantization_config=None):
    if model_type == "llama":
      model = MllamaForConditionalGeneration.from_pretrained(model_id, 
                                                            trust_remote_code=True,
                                                            device_map=device_map,
                                                            torch_dtype=torch_dtype,
                                                            quantization_config=quantization_config)
    elif model_type == "gemma":
      model = Gemma3ForConditionalGeneration.from_pretrained(model_id, 
                                                            trust_remote_code=True,
                                                            device_map=device_map,
                                                            torch_dtype=torch_dtype,
                                                            quantization_config=quantization_config)
    else:
      raise ValueError("Invalid model type: {}, supported types are: llama, gemma".format(model_type))
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


class MultimodalCustomModel(PythonModel):

  def __init__(self, model_config=None):
    if model_config:
      self.model_config = model_config
    else:
      self.model_config = mlflow.models.ModelConfig(development_config="inference_config.yaml")
      
  def load_context(self, context):
    model_id = context.artifacts['model_path']
    
    device_count = torch.cuda.device_count()  
    device_map = "auto" if device_count > 1 else "cuda:0" #to avoid CPU offloading
    logger.warn("total devices found in system: {} and device_map: {}".format(device_count, device_map))
    
    quant_enabled = os.getenv('QUANT_ENABLED', 'false').lower()
    quant_model_config = None
    logger.warn("is quant enabled? {}".format(str(quant_enabled)))
    if quant_enabled:
      quant_model_config = self.model_config.get("quant_config") 
    
    model_type = self.model_config.get("model_type")
    self.model, self.processor = ModelFactory.load_model(model_type, 
                                                         model_id,
                                                         device_map, 
                                                         torch_dtype=torch.bfloat16, 
                                                         quantization_config=quant_model_config)

  def predict(self, context, model_input, params):
    input_df = model_input.iloc[0].to_dict()
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": input_df['user_prompt']}
    ]}]
    image_input = base64_string_to_pillow_image(input_df['image'])

    input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = self.processor(image_input, input_text, add_special_tokens=False, return_tensors="pt").to(self.model.device)
    input_len = inputs["input_ids"].shape[-1]

    max_new_tokens = params.get("max_new_tokens", 200)
    temperature = params.get("temperature", 0.1)
    top_p = params.get("top_p", 0.9)
    with torch.inference_mode():
      outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
      outputs = outputs[0][input_len:]

    raw_return_string = self.processor.decode(outputs)
    cleaned_return_string = raw_return_string.replace('<end_of_turn>', '')
    cleaned_return_string = raw_return_string.split("<|end_header_id|>")[-1].replace('<|eot_id|>', '')
    torch.cuda.empty_cache()

    return pd.DataFrame().from_dict({"output_text": [cleaned_return_string]})

mlflow.models.set_model(MultimodalCustomModel())
