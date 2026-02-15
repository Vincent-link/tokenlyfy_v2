"""
加密投研助手 MVP - Gradio 产品演示

使用 Gradio Blocks + 自定义会话列表，演示完整 MVP 功能：
- 个性化分析助手（先结论再四部分）
- 行情/技术/资金面与情绪工具调用
- 记忆系统（session 匿名 ID 持久化）
- RAG 知识库 + Memory Recall + 用户画像
- 流式输出（报告阶段逐字输出）
- 左侧历史对话列表（Gradio 4.x 的 save_history 在 4.44 中未实现或不可见，故自定义实现）

运行：uv run python examples/web_demo/gradio_demo.py
依赖：uv sync 后需安装 evaluation 组，如 uv pip install -e ".[evaluation]"
"""

import threading
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from hello_agents.assistants import create_crypto_assistant  # noqa: E402

try:
    import gradio as gr
except ImportError:
    raise ImportError(
        "Gradio 未安装。请运行: uv pip install -e \".[evaluation]\" 或 pip install gradio"
    )

_agent = None
# 是否流式输出（报告阶段逐 token 展示，需 demo.queue() 支持）
STREAM_RESPONSE = False
EXAMPLES = ["分析 BTC 短线", "ETH 1h 技术面怎么看", "SUI 能抄底吗", "当前恐惧贪婪指数"]


def _get_agent():
    """懒加载 Agent，避免启动时加载 heavy 依赖导致 500。页面加载后后台预加载。"""
    global _agent
    if _agent is None:
        _agent = create_crypto_assistant(
            persist_session=True,
            max_steps=5,
            use_rag=True,
        )
    return _agent


def _make_title(msg: str, max_len: int = 20) -> str:
    return (msg[:max_len] + "…") if len(msg) > max_len else msg


def _run_chat(message: str):
    """执行 Agent 推理，返回回复。"""
    if not message or not message.strip():
        return "请输入您的问题"
    try:
        agent = _get_agent()
        if STREAM_RESPONSE and getattr(agent, "run_stream", None):
            partial = ""
            for chunk in agent.run_stream(message.strip()):
                partial += chunk
                yield partial
        else:
            yield agent.run(message.strip())
    except Exception as e:
        yield f"❌ 错误: {str(e)}"


def submit_msg(message, history, sessions, next_id, current_id):
    """用户发送消息：调用 Agent，追加到当前对话，必要时新建会话。"""
    if not message or not message.strip():
        return history, "", sessions, next_id, current_id, gr.Dropdown(choices=[])
    gen = _run_chat(message.strip())
    response = ""
    for chunk in gen:
        response = chunk
    new_history = history + [(message.strip(), response)]
    title = _make_title(message.strip())
    now = datetime.now().strftime("%H:%M")
    if current_id is not None:
        # 更新当前会话
        updated = []
        for s in sessions:
            if s["id"] == current_id:
                updated.append({"id": s["id"], "title": f"{title} ({now})", "history": new_history})
            else:
                updated.append(s)
        updated_sessions = updated
        sid = current_id
    else:
        # 新建会话
        sid = str(next_id)
        new_session = {"id": sid, "title": f"{title} ({now})", "history": new_history}
        updated_sessions = sessions + [new_session]
        next_id = next_id + 1
    choices = [(s["title"], s["id"]) for s in updated_sessions]
    return new_history, "", updated_sessions, next_id, sid, gr.Dropdown(choices=choices, value=sid)


def new_chat(history, sessions, next_id):
    """新建对话：若有内容则保存到列表，并清空当前区域。清空后 current_id=None。"""
    if history:
        title = _make_title(history[0][0])
        now = datetime.now().strftime("%H:%M")
        sid = str(next_id)
        new_session = {"id": sid, "title": f"{title} ({now})", "history": history}
        updated_sessions = sessions + [new_session]
        choices = [(s["title"], s["id"]) for s in updated_sessions]
        return [], updated_sessions, next_id + 1, gr.Dropdown(choices=choices), None
    choices = [(s["title"], s["id"]) for s in sessions]
    return [], sessions, next_id, gr.Dropdown(choices=choices), None


def load_session(choice, sessions):
    """选择历史对话：加载对应 history 到聊天区域，并设为当前会话。"""
    if not choice or not sessions:
        return [], None
    for s in sessions:
        if s["id"] == choice:
            return s["history"], choice
    return [], None


def build_demo():
    with gr.Blocks(
        title="加密投研助手 MVP",
        theme=gr.themes.Soft(),
        css="""
        .session-list { max-height: 60vh; overflow-y: auto; }
        """,
    ) as demo:
        gr.Markdown("## 加密投研助手 MVP")
        gr.Markdown("分析加密货币行情、技术面、资金面与情绪。支持 BTC、ETH、SOL、SUI 等。回答需 30 秒～2 分钟，请耐心等待。")

        sessions = gr.State([])
        next_id = gr.State(0)
        current_id = gr.State(None)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 历史对话")
                new_btn = gr.Button("➕ 新建对话", variant="secondary")
                session_dropdown = gr.Dropdown(
                    choices=[],
                    label="",
                    value=None,
                    allow_custom_value=False,
                    interactive=True,
                )

            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="", height=500)
                msg = gr.Textbox(placeholder="输入您的问题…", label="", show_label=False, container=False)
                with gr.Row():
                    submit_btn = gr.Button("发送", variant="primary")

        gr.Examples(examples=EXAMPLES, inputs=msg, label="示例问题")

        # 发送消息
        submit_btn.click(
            submit_msg,
            inputs=[msg, chatbot, sessions, next_id, current_id],
            outputs=[chatbot, msg, sessions, next_id, current_id, session_dropdown],
        )
        msg.submit(
            submit_msg,
            inputs=[msg, chatbot, sessions, next_id, current_id],
            outputs=[chatbot, msg, sessions, next_id, current_id, session_dropdown],
        )

        # 新建对话
        new_btn.click(
            new_chat,
            inputs=[chatbot, sessions, next_id],
            outputs=[chatbot, sessions, next_id, session_dropdown, current_id],
        )

        # 切换历史对话
        session_dropdown.change(
            load_session,
            inputs=[session_dropdown, sessions],
            outputs=[chatbot, current_id],
        )

    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.queue()
    threading.Thread(target=_get_agent, daemon=True).start()
    demo.launch(server_name="127.0.0.1", server_port=7861, share=False)
