from stylellm_chat.chat_model import ChatGPTModel
from stylellm_chat.style_model import StyleModel


class StyleLLMChat(object):
    def __init__(self, chat_model=None, style_model=None):
        self.chat_model = chat_model
        self.style_model = style_model

    def chat(self, prompt, show_chat_result=False):
        t = self.chat_model.generate(prompt)
        if show_chat_result:
            print("##", t)
        t = self.style_model.generate(t)
        return t


__all__ = [
  'ChatGPTModel',
  'StyleModel',
  'StyleLLMChat'
]
