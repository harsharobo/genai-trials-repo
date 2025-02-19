import base64
import io
import numpy as np
import pandas as pd
from PIL import Image

import mlflow
from mlflow.pyfunc import PythonModel

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoProcessor, MllamaForConditionalGeneration

  
def base64_string_to_pillow_image(base64_str):
    return Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))
  
class LlamaVLMCustomModel(PythonModel):

  def load_context(self, context):
    model_id = context.artifacts['model_path']
    device_count = torch.cuda.device_count()

    self.tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # ref https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/vision_language.py#L337
    self.llm_engine = LLM(model=model_id,
                          max_model_len=2048,
                          max_num_seqs=16,
                          disable_mm_preprocessor_cache=False,
                          tensor_parallel_size=device_count)

  def predict(self, context, model_input, params):
    #process params
    sampling_params = SamplingParams(temperature=params.get("temperature", 0.1),
                                  max_tokens=params.get("max_new_tokens", 200),
                                  top_p=params.get("top_p", 0.95),
                                  stop_token_ids=None)
    #process input df
    inputs=[]
    for input_df in model_input.to_dict("records"):
        #create text prompt
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": input_df['user_prompt']}
        ]}]
        prompt = self.tokenizer.apply_chat_template(messages,
                                           add_generation_prompt=True,
                                           tokenize=False)
        #process the image
        image_input = base64_string_to_pillow_image(input_df['image'])
        inputs.append({"prompt": prompt,"multi_modal_data": {"image": image_input}})

    outputs = self.llm_engine.generate(inputs, sampling_params=sampling_params)
    output_text=[]
    for o in outputs:
        output_text.append(o.outputs[0].text)
    torch.cuda.empty_cache()

    return pd.DataFrame().from_dict({"output_text": output_text})

mlflow.models.set_model(LlamaVLMCustomModel())
