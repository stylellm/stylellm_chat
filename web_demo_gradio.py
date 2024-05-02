"""A simple web interactive chat demo based on gradio."""
import gradio as gr
from stylellm_chat import StyleLLMChat, QwenChatModel, StyleModel


def parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f"<br></code></pre>"
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def run_app(sc: StyleLLMChat):
    sc.clear_history()
    with gr.Blocks() as demo:
        gr.Markdown(
            '<center><font size=6><a href="https://github.com/stylellm/stylellm_chat">StyleLLM-Chat</a></center>'
        )
        gr.Markdown(
            '<center><font size=2>对话模型：%s &nbsp; 角色：%s</center>' % (
                sc.chat_model.model_name_or_path, sc.chat_model.character)
        )
        gr.Markdown(
            '<center><font size=2>风格模型：%s &nbsp; 角色：%s</center>' % (
                sc.style_model.model_name_or_path, sc.style_model.character)
        )

        chatbot = gr.Chatbot()
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Column(scale=12):
                    user_input = gr.Textbox(
                        show_label=False,
                        placeholder="Input...",
                        lines=10,
                        container=False,
                    )
                with gr.Column(min_width=32, scale=1):
                    submit_btn = gr.Button("🚀 Submit")

            with gr.Column(scale=1):
                empty_btn = gr.Button("🧹 Clear History")
                max_new_tokens = gr.Slider(
                    0,
                    32768,
                    value=1024,
                    step=1.0,
                    label="Maximum length",
                    interactive=True,
                )
                top_p = gr.Slider(
                    0, 1, value=0.8, step=0.01, label="Top P", interactive=True
                )
                temperature = gr.Slider(
                    0.01, 1, value=0.6, step=0.01, label="Temperature", interactive=True
                )

        def user(query, dialog):
            return "", dialog + [[parse_text(query), ""]]

        def predict(history, _max_new_tokens, _top_p, _temperature):
            messages = []
            for idx, (user_msg, model_msg) in enumerate(history):
                if idx == len(history) - 1 and not model_msg:
                    messages.append({"role": "user", "content": user_msg})
                    break
                if user_msg:
                    messages.append({"role": "user", "content": user_msg})
                if model_msg:
                    messages.append({"role": "assistant", "content": model_msg})

            out = sc.generate(messages, max_new_tokens=_max_new_tokens, top_p=_top_p, temperature=_temperature)

            history[-1][-1] = out
            return history

        submit_btn.click(
            user, [user_input, chatbot], [user_input, chatbot], queue=False
        ).then(predict, [chatbot, max_new_tokens, top_p, temperature], chatbot)

        user_input.submit(
            user, [user_input, chatbot], [user_input, chatbot], queue=False
        ).then(predict, [chatbot, max_new_tokens, top_p, temperature], chatbot)

        empty_btn.click(lambda: None, inputs=None, outputs=chatbot, queue=False)

    demo.queue()
    demo.launch(share=True)
    return demo


def main():
    sc = StyleLLMChat(
        chat_model=QwenChatModel(model_name_or_path="Qwen/Qwen1.5-7B-Chat-AWQ", device="cuda"),
        style_model=StyleModel(model_name_or_path="stylellm/ShuiHuZhuan-6b-AWQ", device="cuda",
                               scene="水浒传", character="鲁智深")
    )
    run_app(sc)


if __name__ == '__main__':
    main()
