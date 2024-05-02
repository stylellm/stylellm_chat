import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def unpack(sent):
    sent = sent.strip()
    result = sent

    pattern = r'^.{2,10}[：:] {0,1}(.*)$'
    results = re.findall(pattern, sent, re.DOTALL)
    if results:
        result = results[0]
    result = result.strip().strip('“”"')
    return result


class StyleModel(object):
    default_model_map = {
        "三国演义": "stylellm/SanGuoYanYi-6b",
        "西游记": "stylellm/XiYouji-6b",
        "水浒传": "stylellm/ShuiHuZhuan-6b",
        "红楼梦": "stylellm/HongLouMeng-6b",
    }

    def __init__(self, scene=None, character=None,
                 model_name_or_path=None,
                 device="cuda"):
        self.character = character
        self.device = device
        self.model_name_or_path = model_name_or_path or self.default_model_map.get(scene)
        self.tokenizer, self.model = self.prepare_model(self.model_name_or_path)

    def prepare_model(self, model_name_or_path):
        if model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            if self.device.startswith("cuda"):
                model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path, device_map=self.device, torch_dtype=torch.float16).eval()
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path, device_map=self.device).eval()
            return tokenizer, model
        else:
            print("no style is applied")
            return None, None

    def prepare_input(self, prompt):
        if self.character is not None:
            prompt = f"{self.character}说：“{prompt}”"
        return [{"role": "user", "content": prompt}]

    def generate(self, prompt, **kwargs):
        if self.model is None:
            return prompt

        generate_configs = {"do_sample": False, "repetition_penalty": 1.2}
        generate_configs.update(kwargs)

        prompt = prompt.replace("\n", " ")
        messages = self.prepare_input(prompt)
        input_ids = self.tokenizer.apply_chat_template(
            conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt').to(self.device)
        output_ids = self.model.generate(input_ids, **generate_configs)
        response = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

        response = response.replace("\n", " ")
        if self.character is not None:
            return unpack(response)
        else:
            return response
