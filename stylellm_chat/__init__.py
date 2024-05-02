from stylellm_chat.chat_model import ChatGPTModel, QwenChatModel
from stylellm_chat.style_model import StyleModel


class StyleLLMChat(object):
    def __init__(self, chat_model=None, style_model=None, max_ctx_size=1024):
        self.chat_model = chat_model
        self.style_model = style_model
        self.max_ctx_size = max_ctx_size
        self.history = []

    def _truncate_history(self, history):
        ret = []
        ctx_size = 0
        for msg in reversed(history):
            ctx_size += len(msg["role"]) + len(msg["content"])
            if ctx_size > self.max_ctx_size:
                break
            ret.insert(0, msg)
        return ret

    def generate(self, messages, show_chat_result=False, **kwargs):
        _messages = self._truncate_history(messages)
        t = self.chat_model.generate(_messages, **kwargs)
        if show_chat_result:
            print("##", t)
        t = self.style_model.generate(t)
        return t

    def chat(self, prompt, show_chat_result=False, **kwargs):
        self.history.append({"role": "user", "content": prompt})
        t = self.generate(self.history, show_chat_result, **kwargs)
        self.history.append({"role": "assistant", "content": t})
        return t

    def clear_history(self):
        self.history = []


__all__ = [
  'ChatGPTModel',
  'QwenChatModel',
  'StyleModel',
  'StyleLLMChat'
]
