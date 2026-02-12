"""
加密投研助手 MVP - Gradio 产品演示

使用 Gradio ChatInterface 快速搭建 Web 聊天界面，演示完整 MVP 功能：
- 个性化分析助手（先结论再四部分）
- 行情/技术/资金/情绪工具调用
- 记忆系统（session 匿名 ID 持久化）

运行：uv run python examples/web_demo/gradio_demo.py
依赖：uv sync 后需安装 evaluation 组，如 uv pip install -e ".[evaluation]"
"""

import threading
from dotenv import load_dotenv

load_dotenv()

try:
    import gradio as gr
except ImportError:
    raise ImportError(
        "Gradio 未安装。请运行: uv pip install -e \".[evaluation]\" 或 pip install gradio"
    )

from hello_agents import (
    HelloAgentsLLM,
    ReActAgent,
    PERSONALIZED_ANALYSIS_REACT_PROMPT,
    ToolRegistry,
    search,
    calculate,
)
from hello_agents.core import get_anonymous_user_id
from hello_agents.tools import MemoryTool
from hello_agents.tools.builtin.crypto_tools import (
    get_crypto_price,
    get_fear_greed,
    get_technical,
    get_futures_data,
    get_crypto_analysis,
)

_agent = None


def _get_agent():
    """懒加载 Agent，避免启动时加载 heavy 依赖导致 500。页面加载后后台预加载。"""
    global _agent
    if _agent is None:
        tool_registry = ToolRegistry()
        tool_registry.register_function("search", "网页搜索工具，搜索技术分析或新闻资讯", search)
        tool_registry.register_function("calculate", "数学计算工具", calculate)
        tool_registry.register_function(
            "crypto_analysis",
            "【首选】一次并行获取价格+技术+恐惧贪婪+合约数据，如 crypto_analysis[BTC 1h]",
            get_crypto_analysis,
        )
        tool_registry.register_function(
            "crypto_price",
            "查询加密货币实时价格、市值、24h涨跌幅（如 BTC,ETH）",
            get_crypto_price,
        )
        tool_registry.register_function(
            "fear_greed",
            "查询加密货币恐惧与贪婪指数（输入天数，如 7）",
            get_fear_greed,
        )
        tool_registry.register_function(
            "technical",
            "查询加密货币技术指标RSI/MACD/布林带/EMA/支撑阻力（如 BTC 1h、ETH 4h）",
            get_technical,
        )
        tool_registry.register_function(
            "futures_data",
            "查询合约市场数据：资金费率、持仓量OI、多空比（如 BTC）",
            get_futures_data,
        )
        memory_tool = MemoryTool(
            user_id=get_anonymous_user_id(persist=True),
            memory_types=["working", "episodic", "semantic", "perceptual"],
        )
        tool_registry.register_tool(memory_tool)
        llm = HelloAgentsLLM()
        _agent = ReActAgent(
            name="加密投研助手",
            llm=llm,
            tool_registry=tool_registry,
            max_steps=5,
            custom_prompt=PERSONALIZED_ANALYSIS_REACT_PROMPT,
        )
    return _agent


def chat(message: str, history: list) -> str:
    """ChatInterface 回调：将用户消息交给 Agent，返回回复"""
    if not message or not message.strip():
        return "请输入您的问题"
    try:
        agent = _get_agent()
        response = agent.run(message.strip())
        return response
    except Exception as e:
        return f"❌ 错误: {str(e)}"


demo = gr.ChatInterface(
    fn=chat,
    title="加密投研助手 MVP",
    description="分析加密货币行情、技术面、资金面与情绪。支持 BTC、ETH、SOL、SUI 等主流币种。",
    examples=[
        "分析 BTC 短线",
        "ETH 1h 技术面怎么看",
        "SUI 能抄底吗",
        "当前恐惧贪婪指数",
    ],
    save_history=True,  # 浏览器 localStorage 持久化，关闭后重开可看到历史对话
)

if __name__ == "__main__":
    # 启动时后台预加载 Agent，用户打开页面到发第一条消息通常已加载完成
    threading.Thread(target=_get_agent, daemon=True).start()
    demo.launch(server_name="127.0.0.1", server_port=7861, share=False)
