from transformers import AutoModelForCausalLM, AutoTokenizer


class StyleModel(object):
    def __init__(self, scene=None, character=None, device="cuda"):
        self.character = character
        self.device = device
        self.tokenizer, self.model = self.prepare_model(scene)

    def prepare_model(self, scene):
        model_map = {
            "三国演义": "stylellm/SanGuoYanYi-6b",
            "西游记": "stylellm/XiYouji-6b",
            "水浒传": "stylellm/ShuiHuZhuan-6b",
            "红楼梦": "stylellm/HongLouMeng-6b",
        }
        model_name = model_map.get(scene)
        if model_name:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device).eval()
            return tokenizer, model
        else:
            print(f"Scene {scene} is not supported, no style is applied")
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

        messages = self.prepare_input(prompt)
        input_ids = self.tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt').to(self.device)
        output_ids = self.model.generate(input_ids, **generate_configs)
        response = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

        begin, end = response.find('“'), response.rfind('”')
        if not (begin > 0 and end > 0):
            begin, end = response.find('"'), response.rfind('"')
        return response[begin + 1: end]
