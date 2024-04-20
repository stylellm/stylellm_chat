from openai import OpenAI


class ChatGPTModel(object):
    def __init__(self, scene=None, character=None,
                 model="gpt-3.5-turbo", **kwargs
        ):
        self.scene = scene
        self.character = character
        self.model = model
        self.client = OpenAI(**kwargs)

    def prepare_input(self, prompt):
        if self.scene and self.character:
            messages = [
                {"role": "user", "content": f"请扮演{self.scene}中的{self.character}"},
                {"role": "assistant", "content": f"好的"}
            ]
        else:
            messages = []
        messages.append({"role": "user", "content": f"{prompt}"})
        return messages

    def generate(self, prompt, **kwargs):
        messages = self.prepare_input(prompt)

        generate_configs = {"max_tokens": 300}
        generate_configs.update(kwargs)

        responses = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **generate_configs
        )
        return responses.choices[0].message.content
