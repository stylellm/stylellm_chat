import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from openai import OpenAI


def unpack(sent):
    sent = sent.strip()
    result = sent

    pattern = r'^.{2,10}[：:] {0,1}(.*)$'
    results = re.findall(pattern, sent, re.DOTALL)
    if results:
        result = results[0]
    result = result.strip().strip('“”"')
    return result


class ChatGPTModel(object):
    def __init__(self, scene=None, character=None,
                 model="gpt-3.5-turbo", **kwargs):
        self.scene = scene
        self.character = character
        self.model = model
        self.client = OpenAI(**kwargs)

    def get_headers(self):
        if self.scene and self.character:
            headers = [
                {"role": "user", "content": f"请扮演{self.scene}中的{self.character}"},
                {"role": "assistant", "content": f"好的"}
            ]
        else:
            headers = []
        return headers

    def generate(self, messages, **kwargs):
        conversation = self.get_headers() + messages

        generate_configs = {"max_tokens": 300}
        generate_configs.update(kwargs)

        responses = self.client.chat.completions.create(
            model=self.model,
            messages=conversation,
            **generate_configs
        )
        return responses.choices[0].message.content


class QwenChatModel(object):
    def __init__(self, scene=None, character=None,
                 model_name_or_path="Qwen/Qwen1.5-7B-Chat-AWQ",
                 device="cuda"):
        self.device = device
        self.scene = scene
        self.model_name_or_path = model_name_or_path
        self.character = character
        self.tokenizer, self.model = self.prepare_model(model_name_or_path)

    def prepare_model(self, model_name_or_path):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.device.startswith("cuda"):
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, device_map=self.device, torch_dtype=torch.float16).eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, device_map=self.device).eval()
        return tokenizer, model

    def get_headers(self):
        if self.scene and self.character:
            headers = [
                {"role": "system", "content": f"请扮演{self.scene}中的{self.character}。用{self.character}：...格式进行回复"},
            ]
        else:
            headers = []
        return headers

    def generate(self, messages, **kwargs):
        generate_configs = {"max_new_tokens": 300}
        generate_configs.update(kwargs)

        conversation = self.get_headers() + messages
        input_ids = self.tokenizer.apply_chat_template(
            conversation=conversation, tokenize=True, add_generation_prompt=True, return_tensors='pt').to(self.device)
        output_ids = self.model.generate(input_ids, **generate_configs)
        response = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

        if self.character is not None:
            return unpack(response)
        else:
            return response
