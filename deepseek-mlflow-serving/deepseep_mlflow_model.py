import json
import torch
import transformers
import pandas as pd
from typing import List, Optional, Dict

from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from vllm import LLM, SamplingParams
from mlflow.types.llm import ChatCompletionResponse, ChatCompletionRequest, ChatMessage, ChatParams, ChatChoice, TokenUsageStats
from mlflow.pyfunc import ChatModel, PythonModel
import mlflow

class DeepseekCustomModel(PythonModel):

    def __init__(self):
        self.model_config = mlflow.models.ModelConfig(development_config="deepseek_vllm_config.yaml")

    def load_context(self, context):
        self.model_path = context.artifacts['model_path']
        
        max_model_len = self.model_config.get("max_model_len")
        tensor_parallel = self.model_config.get("tensor_parallel")
        self.vllm_engine = LLM(model=self.model_path, 
                               tensor_parallel_size=tensor_parallel,
                               trust_remote_code=True,
                               max_model_len=max_model_len)

    def _request_handler(self, messages: List[ChatMessage]):
        temp_messages = []
        for message in messages:
            if hasattr(message, 'tool_call_id') and message.tool_call_id:
                message_block = {"role": "tool",
                                "content": message.content,
                                "tool_call_id": message.tool_call_id}
            else:
                message_block = {"role": message.role,
                                  "content": message.content}
            temp_messages.append(message_block)
        return temp_messages

    def _response_handler(self, model_output):
        vllm_request_output = model_output[0]
        
        input_token_len = len(vllm_request_output.prompt_token_ids)
        if vllm_request_output.outputs[0]:
            output_token_len = len(vllm_request_output.outputs[0].token_ids)
            total_token_len = input_token_len + output_token_len
            
            str_output = vllm_request_output.outputs[0].text.strip()
            try:
                tools_dict_list = json.loads(str_output)
                if len(tools_dict_list)>0 and tools_dict_list[0].get('type') == 'function':
                    chat_message_resp = ChatMessage(role="assistant", content="", tool_calls=tools_dict_list)
                else:
                    chat_message_resp = ChatMessage(role="assistant", content=str_output)
            except json.JSONDecodeError:
                chat_message_resp = ChatMessage(role="assistant", content=str_output)
            
            output = ChatCompletionResponse(
                    choices=[ChatChoice(index=0, message=chat_message_resp)],
                    usage=TokenUsageStats(prompt_tokens=input_token_len, 
                                          completion_tokens=output_token_len, 
                                          total_tokens=total_token_len))
            return output.to_dict()
        else:
            raise Exception("No output found")

    def predict(self, context, model_input):
        model_input_dict = model_input.to_dict("records")
        chat_request = ChatCompletionRequest.from_dict(model_input_dict[0])
        messages = chat_request.messages

        params = dict()
        params['temperature'] = chat_request.temperature
        params['max_tokens'] = chat_request.max_tokens
        params['top_p'] = chat_request.top_p

        sampling_params = SamplingParams(temperature=params.get('temperature', 0.1), 
                                        top_p=params.get('top_p', 0.95),
                                        max_tokens=params.get('max_tokens', 128))
        
        if hasattr(chat_request, 'tools') and chat_request.tools:
            tools = [each_tool.function.to_dict() for each_tool in chat_request.tools]
            input_messages = self._request_handler(messages)
            outputs = self.vllm_engine.chat(input_messages, sampling_params=sampling_params, tools=tools)
        else:
            input_messages = self._request_handler(messages)
            outputs = self.vllm_engine.chat(input_messages, sampling_params=sampling_params)
        
        return self._response_handler(outputs)
      
mlflow.models.set_model(DeepseekCustomModel())