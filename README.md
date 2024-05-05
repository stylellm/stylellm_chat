<div align="center">
<h1>StyleLLM Chat</h1>
</div>

## 项目介绍

> 通用大模型 × 文风大模型 = 多样化风格的🤖💬聊天机器人

StyleLLM Chat利用通用大模型的通用能力（世界知识、逻辑推理、对话问答）和文风大模型（[StyleLLM](https://github.com/stylellm/stylellm_models)） 的语言风格转化能力，探索实现多样化风格的聊天机器人。

利用StyleLLM模型对ChatGPT等通用大模型的输出进行风格改写，可改变对话风格单一、AI味过重的状况。


## 功能
- 集成OpenAI API，填写api_key后可以调用ChatGPT对话能力。
- 集成Qwen模型，支持调用Qwen本地化模型对话能力。 
- 集成stylellm_models，支持对ChatGPT或Qwen生成结果进行风格润色。
- 支持通过场景（scene）和角色（character）参数，控制具体风格类型。
- 支持多轮对话。


## 使用方法
### 在Colab中快速使用 
[打开Colab](https://colab.research.google.com/github/stylellm/stylellm_chat/blob/main/colab_web_demo_gradio.ipynb) ，运行notebook内容后，可进入如下Gradio界面进行对话：<br/>
![demo1](https://github.com/stylellm/stylellm_chat/blob/main/%20assets/gradio_chat_1.png)
![demo2](https://github.com/stylellm/stylellm_chat/blob/main/%20assets/gradio_chat_2.png)

### 本地安装
```shell
git clone https://github.com/stylellm/stylellm_chat.git
cd stylellm_chat
pip install -r requirements.txt
```

### 运行（调用Qwen方式）
```python
from stylellm_chat import StyleLLMChat, QwenChatModel, StyleModel

sc = StyleLLMChat(
    chat_model=QwenChatModel(
        model_name_or_path="Qwen/Qwen1.5-7B-Chat", 
        device="cuda",
        scene=None, 
        character=None
    ),
    style_model=StyleModel(
        model_name_or_path="stylellm/HongLouMeng-6b", 
        device="cuda",
        scene="红楼梦", 
        character="黛玉"
    )
)

response = sc.chat("今天天气真不错")
print(response)
```

### 运行（调用ChatGPT方式）
```python
from stylellm_chat import StyleLLMChat, ChatGPTModel, StyleModel

sc = StyleLLMChat(
    chat_model=ChatGPTModel(
        scene="红楼梦", 
        character="黛玉", 
        api_key="#replace with your api key#"
    ),
    style_model=StyleModel(
        scene="红楼梦", 
        character="黛玉", 
        device="cuda"
    )
)

response = sc.chat("今天天气真不错")
print(response)
```
**注意：**
> 运行上述代码需要替换**api_key**参数的内容，具体获取方式参考：https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key
> 


### 参数说明
> - **QwenChatModel**
>   - **`model_name_or_path`**: 对话模型名称或本地路径。可选值为Qwen1.5系列模型[[查看列表](https://huggingface.co/Qwen)]，如"Qwen/Qwen1.5-7B-Chat"
>   - **`device`**: 对话模型使用的硬件。可选值为"cpu", "cuda"(或用"cuda:0", "cuda:1"指定不同显卡)
>   - **`scene`**, **`character`** ：对话模型人设。设置后对话模型将扮演"scene中的character"。默认为None，即不使用角色扮演，保持对话模型原有的AI助理人设。
>
> - **ChatGPTModel**
>   - **`model`**: 对话模型名称。可选值为GPT系列模型，如"gpt-3.5-turbo", "gpt-4-turbo"
>   - **`api_key`**: openai api key
>   - **`scene`**, **`character`** ：对话模型人设。设置后对话模型将扮演"scene中的character"。默认为None，即不使用角色扮演，保持对话模型原有的AI助理人设。
>
> - **StyleModel**
>   - **`model_name_or_path`**: 风格模型名称或本地路径。可选值为stylellm系列模型[[查看列表](https://github.com/stylellm/stylellm_models?tab=readme-ov-file#%E6%A8%A1%E5%9E%8B%E5%88%97%E8%A1%A8)]，如"stylellm/HongLouMeng-6b"
>   - **`device`**: 风格模型使用的硬件。可选值为"cpu", "cuda"(或用"cuda:0", "cuda:1"指定不同显卡)
>   - **`scene`**, **`character`** ：风格类型。设置后风格模型将以"scene中的character"说话风格输出
>


## 测试案例
### 无角色风格
```python
from stylellm_chat import StyleLLMChat, ChatGPTModel, StyleModel

sc = StyleLLMChat(
    chat_model=ChatGPTModel(api_key=api_key),
    style_model=StyleModel()
)

print(sc.chat("今天天气真不错"))
# 是的，好天气可以让人心情愉快。你有什么计划要在这样美好的天气里做吗？

print(sc.chat("你们天天一起玩，我来了也不理我"))
# 可能是因为他们已经形成了群体，你的加入打破了原有的平衡，需要时间去接纳你。试着主动和他们交流，参与他们的活动，慢慢地就会融入他们的圈子中。也可以尝试和他们私下聊聊，表达你的感受和需求，看看能否得到理解和支持。如果他们还是不理你，那可能他们不是真正的朋友，可以考虑寻找新的朋友圈子。
```

### 三国孔明
```python
from stylellm_chat import StyleLLMChat, ChatGPTModel, StyleModel

sc = StyleLLMChat(
    chat_model=ChatGPTModel(scene="三国演义", character="孔明", api_key=api_key),
    style_model=StyleModel(scene="三国演义", character="孔明", device="cuda")
)

print(sc.chat("今天天气真不错"))
# 然也！今日天色甚佳。吾等游于清风之阳，其乐无穷。愿公享此良辰美景，更当奋勉图功。

print(sc.chat("你们天天一起玩，我来了也不理我"))
# 某乃儒生，不能常侍左右，倘有差遣，乞赐驱使。
```

### 西游八戒
```python
from stylellm_chat import StyleLLMChat, ChatGPTModel, StyleModel

sc = StyleLLMChat(
    chat_model=ChatGPTModel(scene="西游记", character="八戒", api_key=api_key),
    style_model=StyleModel(scene="西游记", character="八戒", device="cuda")
)

print(sc.chat("今天天气真不错"))
# 正是，师父，今日天色甚好。晴云霁日，清风徐来，正宜游耍哩！可也同去么？

print(sc.chat("你们天天一起玩，我来了也不理我"))
# 师父呀！你看那厮又来瞌睡哩。我也没奈何，你又不叫他去，我怎敢近前？却怎么处？
```

### 水浒鲁智深
```python
from stylellm_chat import StyleLLMChat, ChatGPTModel, StyleModel

sc = StyleLLMChat(
    chat_model=ChatGPTModel(scene="水浒传", character="鲁智深", api_key=api_key),
    style_model=StyleModel(scene="水浒传", character="鲁智深", device="cuda")
)

print(sc.chat("今天天气真不错"))
# 好！今朝日色明净，正宜练武。俺便与你使个手到斫头势！

print(sc.chat("你们天天一起玩，我来了也不理我"))
# 呸！你这撮鸟来迟也！休要烦恼，俺这里谁敢怠慢你？且说近日做甚麽勾当？
```

### 红楼黛玉
```python
from stylellm_chat import StyleLLMChat, ChatGPTModel, StyleModel

sc = StyleLLMChat(
    chat_model=ChatGPTModel(scene="红楼梦", character="黛玉", api_key=api_key),
    style_model=StyleModel(scene="红楼梦", character="黛玉", device="cuda")
)

print(sc.chat("今天天气真不错"))
# 可不是么？今日这日头儿倒好。又晴明，又和暖，风也轻柔。不知你还作何兴致？

print(sc.chat("你们天天一起玩，我来了也不理我"))
# 嗳哟！我那里知道你来呢？你要同咱们一处作耍儿，只管来罢。
```

