import gradio as gr

from implementation_2_1.answer import answer_question



def format_context(context):
    result = "<h2 style='color: #ff7800;'>Relevant Context</h2>\n\n"
    for doc in context:
        result += f"<span style='color: #ff7800;'>Source: {doc.metadata['source']}</span>\n\n"
        result += doc.page_content + "\n\n"
    return result


def extract_text(content):
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    elif isinstance(content, dict):
        return content.get("text", "")
    return str(content)


def chat(history):
    clean_history = [
        {"role": msg["role"], "content": extract_text(msg["content"])}
        for msg in history
    ]
    last_message = clean_history[-1]["content"]
    prior = clean_history[:-1]
    answer, context = answer_question(last_message, prior)
    clean_history.append({"role": "assistant", "content": answer})
    return clean_history, format_context(context)


def main():
    def put_message_in_chatbot(message, history):
        return "", history + [{"role": "user", "content": message}]

    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="RAG Apple IPhone", theme=theme) as ui:
        gr.Markdown("# RAG Apple IPhone ")

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="💬 Conversation", height=600, buttons=["copy"]
                )
                message = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask anything about Apple iPhone...",
                    show_label=False,
                )

            with gr.Column(scale=1):
                context_markdown = gr.Markdown(
                    label="📚 Retrieved Context",
                    value="*Retrieved context will appear here*",
                    container=True,
                    height=600,
                )

        message.submit(
            put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]
        ).then(chat, inputs=chatbot, outputs=[chatbot, context_markdown])

    ui.launch(inbrowser=True)


if __name__ == "__main__":
    main()
