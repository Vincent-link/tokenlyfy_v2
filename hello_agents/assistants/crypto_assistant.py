"""加密投研助手工厂

提供 create_crypto_assistant() 一行创建完整加密投研助手，
统一工具注册、记忆、RAG、报告生成等配置。
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Any

from .. import (
    HelloAgentsLLM,
    ReActAgent,
    PERSONALIZED_ANALYSIS_REACT_PROMPT,
    MARKET_ANALYSIS_REACT_PROMPT,
    ToolRegistry,
    search,
    calculate,
)
from ..core import get_anonymous_user_id
from ..tools import MemoryTool
from ..tools.builtin.crypto_tools import (
    get_crypto_price,
    get_fear_greed,
    get_technical,
    get_futures_data,
    get_crypto_analysis,
)
from .report_generator import ReportGenerator
from .user_profile import UserProfileStore

logger = logging.getLogger(__name__)


@dataclass
class CryptoAssistantConfig:
    """加密投研助手配置，可用于 create_crypto_assistant(config=...) 或从 YAML/环境加载"""

    persist_session: bool = True
    memory_types: List[str] = field(default_factory=lambda: ["working", "episodic", "semantic", "perceptual"])
    max_steps: int = 5
    prompt_type: str = "personalized"  # "personalized" | "market"
    use_rag: bool = True
    response_cache_ttl_seconds: Optional[int] = 120

    @classmethod
    def from_env(cls) -> "CryptoAssistantConfig":
        """从环境变量加载（可选），未设则用默认值"""
        import os
        return cls(
            persist_session=os.getenv("CRYPTO_ASSISTANT_PERSIST_SESSION", "true").lower() in ("1", "true", "yes"),
            max_steps=int(os.getenv("CRYPTO_ASSISTANT_MAX_STEPS", "5")),
            prompt_type=os.getenv("CRYPTO_ASSISTANT_PROMPT_TYPE", "personalized"),
            use_rag=os.getenv("CRYPTO_ASSISTANT_USE_RAG", "true").lower() in ("1", "true", "yes"),
            response_cache_ttl_seconds=int(os.getenv("CRYPTO_ASSISTANT_CACHE_TTL", "120")) or None,
        )


# 加密知识库路径（用于 RAG 灌入）
_CRYPTO_KNOWLEDGE_NAMESPACE = "crypto_knowledge"
_CRYPTO_KNOWLEDGE_FILES = ["crypto_analysis.md", "crypto_history_cases.md"]
_crypto_rag_ingested = False


def _ensure_crypto_knowledge_ingested(rag_tool) -> bool:
    """确保加密知识库已灌入 RAG，仅首次执行"""
    global _crypto_rag_ingested
    if _crypto_rag_ingested:
        return True
    try:
        knowledge_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "knowledge"
        )
        for filename in _CRYPTO_KNOWLEDGE_FILES:
            path = os.path.join(knowledge_dir, filename)
            if os.path.exists(path):
                rag_tool.run({
                    "action": "add_document",
                    "file_path": path,
                    "namespace": _CRYPTO_KNOWLEDGE_NAMESPACE,
                })
        _crypto_rag_ingested = True
        logger.info("✅ 加密知识库已灌入 RAG")
        return True
    except Exception as e:
        logger.warning(f"RAG 灌入失败，报告将回退静态加载: {e}")
        return False


def create_crypto_assistant(
    persist_session: bool = True,
    memory_types: Optional[List[str]] = None,
    max_steps: int = 5,
    prompt_template=None,
    use_rag: bool = True,
    response_cache_ttl_seconds: Optional[int] = 120,
    config: Optional[CryptoAssistantConfig] = None,
    **overrides: Any,
) -> ReActAgent:
    """创建完整加密投研助手

    Args:
        persist_session: 是否持久化 session（同一设备保留记忆）
        memory_types: 记忆类型列表，默认 working/episodic/semantic/perceptual
        max_steps: ReAct 最大步数
        prompt_template: 搜索阶段 prompt，默认 PERSONALIZED_ANALYSIS_REACT_PROMPT
        use_rag: 是否使用 RAG 检索知识（否则静态加载，无 Qdrant 依赖）
        response_cache_ttl_seconds: 短时同问缓存时长（秒），120=2 分钟内同问复用答案，None 或 0 关闭
        config: 可选配置对象，若提供则以其为准，再被 overrides 覆盖
        **overrides: 覆盖 config 或默认值的参数（persist_session, max_steps, use_rag 等）

    Returns:
        配置好的 ReActAgent 实例
    """
    if config is not None:
        persist_session = overrides.get("persist_session", config.persist_session)
        memory_types = overrides.get("memory_types", config.memory_types)
        max_steps = overrides.get("max_steps", config.max_steps)
        use_rag = overrides.get("use_rag", config.use_rag)
        response_cache_ttl_seconds = overrides.get("response_cache_ttl_seconds", config.response_cache_ttl_seconds)
        if prompt_template is None:
            prompt_template = MARKET_ANALYSIS_REACT_PROMPT if config.prompt_type == "market" else PERSONALIZED_ANALYSIS_REACT_PROMPT
    else:
        persist_session = overrides.get("persist_session", persist_session)
        memory_types = overrides.get("memory_types", memory_types)
        max_steps = overrides.get("max_steps", max_steps)
        use_rag = overrides.get("use_rag", use_rag)
        response_cache_ttl_seconds = overrides.get("response_cache_ttl_seconds", response_cache_ttl_seconds)
    if prompt_template is None:
        prompt_template = PERSONALIZED_ANALYSIS_REACT_PROMPT

    tool_registry = ToolRegistry()

    # 数据工具
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

    # 用户画像存储（报告注入 + 存记忆时同步更新）
    profile_store = UserProfileStore()
    user_id = get_anonymous_user_id(persist=persist_session)
    memory_tool = MemoryTool(
        user_id=user_id,
        memory_types=memory_types or ["working", "episodic", "semantic", "perceptual"],
        profile_store=profile_store,
    )
    tool_registry.register_tool(memory_tool)

    # RAG 工具（可选）
    rag_tool = None
    if use_rag:
        try:
            from ..tools.builtin.rag_tool import RAGTool
            rag_tool = RAGTool(
                rag_namespace=_CRYPTO_KNOWLEDGE_NAMESPACE,
                collection_name="hello_agents_rag_vectors",  # 与记忆存储分离
            )
            _ensure_crypto_knowledge_ingested(rag_tool)
        except Exception as e:
            logger.warning(f"RAG 初始化失败，报告将使用静态知识库: {e}")
            use_rag = False

    # 报告生成器（含 Memory Recall + 用户画像 + 可选 RAG）
    llm = HelloAgentsLLM()
    report_generator = ReportGenerator(
        llm,
        use_rag=use_rag,
        rag_tool=rag_tool,
        memory_tool=memory_tool,
        user_id=user_id,
        user_profile_store=profile_store,
    )

    # 创建 ReActAgent（短时同问复用：2 分钟内相同/相似问题直接返回缓存）
    agent = ReActAgent(
        name="加密投研助手",
        llm=llm,
        tool_registry=tool_registry,
        max_steps=max_steps,
        custom_prompt=prompt_template or PERSONALIZED_ANALYSIS_REACT_PROMPT,
        report_generator=report_generator,
        response_cache_ttl_seconds=response_cache_ttl_seconds,
    )

    return agent
