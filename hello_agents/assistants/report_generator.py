"""报告生成器 - 基于搜索观察生成加密分析报告

从 ReActAgent 中抽离的报告生成逻辑，支持：
- 技术分析方法论知识库（静态/RAG）
- 历史行情案例参考
- 前次预测回顾
- 可选：Memory Recall 注入用户上下文
"""

from typing import Optional, List, Any, Iterator
import os
import logging

logger = logging.getLogger(__name__)


def _get_knowledge_dir() -> str:
    """获取 hello_agents/knowledge 目录路径"""
    return os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "knowledge"
    )


def _load_static_file(filename: str, max_len: int = 2800) -> str:
    """静态加载文件内容，超长则截断"""
    path = os.path.join(_get_knowledge_dir(), filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        if len(content) > max_len:
            content = content[:max_len] + "\n... (更多内容已省略)"
        return content
    except FileNotFoundError:
        logger.debug(f"知识文件未找到: {path}")
        return ""


def _get_previous_prediction_from_history(history: List[Any], max_content_len: int = 600) -> str:
    """从对话历史中提取最近一次分析预测"""
    if not history:
        return ""
    markers = ("置信度", "偏向", "结论", "预测", "短线", "中线", "建议", "抄底", "减仓")
    for m in reversed(history):
        role = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else None)
        content = getattr(m, "content", None) or (m.get("content", "") if isinstance(m, dict) else "")
        if role != "assistant" or not content:
            continue
        content = (content or "").strip()
        if any(kw in content for kw in markers):
            if len(content) > max_content_len:
                content = content[:max_content_len] + "…"
            return content
    return ""


class ReportGenerator:
    """加密分析报告生成器

    接收搜索阶段的 observations，结合知识库、历史案例、用户记忆生成完整报告。
    """

    # 通用分析原则
    ANALYSIS_RULES = """## 分析原则（必须遵守）
1. **数据交叉验证**：不只罗列数据，要分析不同指标之间的关系。例如：RSI 超卖 + Fear & Greed 极度恐惧 + 价格触及布林带下轨 = 强超卖信号。
2. **多空论据对比**：分别列出看多和看空的论据，不要一边倒。
3. **置信度评估**：在结论中给出判断的置信度（如「偏向震荡反弹，置信度 65%」），让用户了解确定性。
4. **引用具体数值**：必须写出查到的指标具体数值（如 RSI=28.5 而非"RSI 偏低"），让结论有据可查。
5. **来源标注**：在关键数据后标注来源，格式：[来源](url)。
6. **历史对比（P2）**：若提供了「历史类似案例」，请根据当前条件挑选最相近案例简要引用；若有「前次预测回顾」，可加一句提醒用户对照验证。"""

    def __init__(
        self,
        llm: Any,
        *,
        use_rag: bool = False,
        rag_tool: Optional[Any] = None,
        memory_tool: Optional[Any] = None,
        user_id: Optional[str] = None,
        user_profile_store: Optional[Any] = None,
    ):
        """
        Args:
            llm: LLM 实例
            use_rag: 是否使用 RAG 检索知识（否则静态加载）
            rag_tool: RAG 工具实例（use_rag=True 时必需）
            memory_tool: 记忆工具，报告前可做 recall 注入
            user_id: 用户 ID，用于 memory recall 与用户画像
            user_profile_store: 用户画像存储，用于注入报告
        """
        self.llm = llm
        self.use_rag = use_rag
        self.rag_tool = rag_tool
        self.memory_tool = memory_tool
        self.user_id = user_id
        self.user_profile_store = user_profile_store

    def _load_knowledge(self, query: str) -> str:
        """加载技术分析方法论：RAG 检索或静态文件"""
        if self.use_rag and self.rag_tool:
            try:
                # 偏向技术分析内容的检索
                search_query = f"{query} 技术分析 RSI MACD 布林带 指标解读"
                result = self.rag_tool.run({
                    "action": "search",
                    "query": search_query,
                    "namespace": "crypto_knowledge",
                    "limit": 5,
                    "max_chars": 2800,
                })
                if result and not result.startswith("❌"):
                    return result
            except Exception as e:
                logger.warning(f"RAG 检索失败，回退静态加载: {e}")
        return _load_static_file("crypto_analysis.md", max_len=2800)

    def _load_history_cases(self, query: str) -> str:
        """加载历史行情案例：RAG 检索或静态文件"""
        if self.use_rag and self.rag_tool:
            try:
                # 偏向历史复盘案例的检索
                search_query = f"{query} 历史案例 复盘 恐惧贪婪 RSI 走势"
                result = self.rag_tool.run({
                    "action": "search",
                    "query": search_query,
                    "namespace": "crypto_knowledge",
                    "limit": 3,
                    "max_chars": 2200,
                })
                if result and not result.startswith("❌"):
                    return result
            except Exception as e:
                logger.warning(f"RAG 历史案例检索失败，回退静态加载: {e}")
        return _load_static_file("crypto_history_cases.md", max_len=2200)

    def _recall_memory(self, question: str, observations_summary: str) -> str:
        """从记忆中召回用户相关上下文"""
        if not self.memory_tool:
            return ""
        try:
            query = f"用户偏好 币种 风险偏好 历史分析 {question[:80]}"
            result = self.memory_tool.run({
                "action": "search",
                "query": query,
                "limit": 5,
            })
            if result and not result.startswith("❌") and len(result.strip()) > 20:
                return result.strip()
        except Exception as e:
            logger.debug(f"Memory recall 失败: {e}")
        return ""

    def generate(
        self,
        question: str,
        observations: str,
        recent_dialogue: str,
        current_date: str,
        conversation_history: Optional[List[Any]] = None,
        is_fixed_template: bool = False,
        **kwargs
    ) -> str:
        """生成分析报告

        Args:
            question: 用户问题
            observations: 搜索阶段收集的数据（history_str）
            recent_dialogue: 最近对话摘要
            current_date: 当前日期时间
            conversation_history: 对话历史（用于前次预测回顾）
            is_fixed_template: 是否使用固定四部分报告结构
            **kwargs: 传给 LLM 的参数
        """
        # 构建检索 query
        query_for_rag = f"{question} {observations[:500]}" if observations else question

        # 知识库
        knowledge = self._load_knowledge(query_for_rag)
        knowledge_section = ""
        if knowledge:
            knowledge_section = f"""## 技术分析方法论参考（请依据此框架解读指标）
{knowledge}
"""

        # 历史案例
        history_cases = self._load_history_cases(query_for_rag)
        history_section = ""
        if history_cases:
            history_section = f"""
## 历史类似案例参考（P2：上次类似情况怎么走的）
请根据当前数据（恐惧贪婪、RSI、资金费率等）挑选最相近的 1～2 个案例作参考，在报告中简要提及，增强结论的可比性。不要机械套用，仅作参考。
{history_cases}
"""

        # 前次预测回顾
        prev_pred = _get_previous_prediction_from_history(conversation_history or [])
        prev_section = ""
        if prev_pred:
            prev_section = f"""
## 前次预测回顾
上次分析中我们的结论/建议摘要如下。可在报告中简要提及，并提醒用户对照近期走势自行验证，提升可信度。
---
{prev_pred}
---
"""

        # Memory Recall 用户上下文
        memory_section = ""
        memory_context = self._recall_memory(question, observations[:500])
        if memory_context:
            memory_section = f"""
## 用户上下文（来自记忆）
{memory_context}
"""

        # 用户画像（结构化偏好）
        profile_section = ""
        if self.user_profile_store and self.user_id:
            try:
                profile = self.user_profile_store.get(self.user_id)
                if profile and getattr(profile, "to_summary", None):
                    summary = profile.to_summary()
                    if summary:
                        profile_section = f"""
## 用户画像（投研偏好）
{summary}
"""
            except Exception as e:
                logger.debug(f"加载用户画像失败: {e}")

        if is_fixed_template:
            report_prompt = f"""你是一个专业的加密货币分析师。根据以下收集到的数据，写出一份完整的分析报告。

{self.ANALYSIS_RULES}

## 报告结构
1. **结论**：1～2 句话概括走势判断 + 置信度（如「短线偏向震荡反弹，置信度 60%」）
2. **1. 价格位置**：当前报价、多空情况；引用 crypto_price 工具的具体数据
3. **2. 技术面**：引用 technical 工具的 RSI/MACD/布林带/EMA/支撑阻力具体数值，给出技术判断
4. **3. 市场情绪与资金面**：引用 fear_greed 工具的指数数值，结合搜索到的资金面信息
5. **4. 多空博弈**：
   | 方向 | 论据 | 权重 |
   分别列出看多和看空的 2～3 条论据及权重
6. **5. 操作提示**：用表格（策略 | 关键价位 | 策略说明），含短线/中长线建议
7. 结尾一句与用户互动的提问

## 最近对话（供理解上下文）
{recent_dialogue}

## 基本信息
- 当前日期：{current_date}
- 用户问题：{question}

{knowledge_section}
{history_section}
{prev_section}
{memory_section}
{profile_section}
## 收集到的数据
{observations}

请直接输出完整报告（只输出报告，不要写 Thought/Action/Finish）："""
        else:
            report_prompt = f"""你是一个专业的加密货币分析师。根据以下收集到的数据和用户问题，写出一份**紧扣问题**的分析回答。

{self.ANALYSIS_RULES}

## 回答方式
1. 先给**结论或总述**（1～2 句话 + 置信度）
2. 根据用户问题设计 **2～4 个小标题**（可用问句或要点）
3. 每个小标题下引用具体数值展开分析
4. 必须包含一段**多空论据对比**（可以是单独小节或融入内容）
5. 结尾一句与用户互动的提问

## 最近对话（供理解上下文，当前问题可能是追问）
{recent_dialogue}

## 基本信息
- 当前日期：{current_date}
- 用户问题：{question}

{knowledge_section}
{history_section}
{prev_section}
{memory_section}
{profile_section}
## 收集到的数据
{observations}

请直接输出完整回答（只输出回答，不要写 Thought/Action/Finish）："""

        report = self.llm.invoke([{"role": "user", "content": report_prompt}], **kwargs)
        return (report or "").strip() or "抱歉，报告生成失败，请重试。"

    def generate_stream(
        self,
        question: str,
        observations: str,
        recent_dialogue: str,
        current_date: str,
        conversation_history: Optional[List[Any]] = None,
        is_fixed_template: bool = False,
        **kwargs
    ) -> Iterator[str]:
        """流式生成报告，逐 token 产出（供 Gradio 等流式展示）"""
        # 与 generate 相同的 prompt 构建逻辑，复用一次
        query_for_rag = f"{question} {observations[:500]}" if observations else question
        knowledge = self._load_knowledge(query_for_rag)
        knowledge_section = f"\n## 技术分析方法论参考（请依据此框架解读指标）\n{knowledge}\n" if knowledge else ""
        history_cases = self._load_history_cases(query_for_rag)
        history_section = f"\n## 历史类似案例参考\n{history_cases}\n" if history_cases else ""
        prev_pred = _get_previous_prediction_from_history(conversation_history or [])
        prev_section = f"\n## 前次预测回顾\n---\n{prev_pred}\n---\n" if prev_pred else ""
        memory_section = ""
        mc = self._recall_memory(question, observations[:500])
        if mc:
            memory_section = f"\n## 用户上下文（来自记忆）\n{mc}\n"
        profile_section = ""
        if self.user_profile_store and self.user_id:
            try:
                profile = self.user_profile_store.get(self.user_id)
                if profile and getattr(profile, "to_summary", None):
                    s = profile.to_summary()
                    if s:
                        profile_section = f"\n## 用户画像\n{s}\n"
            except Exception:
                pass
        report_prompt = f"""你是一个专业的加密货币分析师。根据以下收集到的数据写出一份**紧扣问题**的分析回答。

{self.ANALYSIS_RULES}

## 回答方式：先结论再分点分析，含多空对比与操作建议。

## 最近对话
{recent_dialogue}

## 基本信息
- 当前日期：{current_date}
- 用户问题：{question}
{knowledge_section}{history_section}{prev_section}{memory_section}{profile_section}

## 收集到的数据
{observations}

请直接输出完整回答（只输出回答）："""
        stream = getattr(self.llm, "stream_invoke", None) or getattr(self.llm, "think", None)
        if not stream:
            full = self.llm.invoke([{"role": "user", "content": report_prompt}], **kwargs)
            yield full or ""
            return
        for chunk in stream([{"role": "user", "content": report_prompt}], **kwargs):
            if chunk:
                yield chunk
