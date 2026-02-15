"""多 Agent 协作 - 加密投研编排器

编排数据收集（DataCollector）与报告分析（Analyst）两阶段，
便于扩展为更多专职 Agent（如 MemoryAgent、预警 Agent）。
"""

from typing import Optional, List, Any, Iterator
import time
import logging

logger = logging.getLogger(__name__)


class CryptoOrchestrator:
    """加密投研编排器：先收集数据，再生成报告。

    与单 ReActAgent 的差异：
    - 数据收集与报告生成在逻辑上分离，可单独替换或扩展。
    - 报告阶段可流式输出（generate_stream）。
    """

    def __init__(
        self,
        collector_agent: Any,
        report_generator: Any,
        name: str = "加密投研编排器",
    ):
        """
        Args:
            collector_agent: 负责执行工具、收集 observations 的 Agent（如 ReActAgent，分析类模板）
            report_generator: ReportGenerator，负责根据 observations 生成报告
        """
        self.collector_agent = collector_agent
        self.report_generator = report_generator
        self.name = name

    def _collect_observations(self, question: str, **kwargs) -> str:
        """运行收集阶段，返回 observations 字符串（不生成最终报告）。"""
        # 若 collector 是 ReActAgent，需要跑完搜索阶段并在 Finish[done] 时截获 history_str，不调用报告
        # 当前 ReActAgent 在 Finish 时直接调报告，无法只拿 observations。因此我们通过「跑一轮但用占位报告」或「专用 DataCollector」实现。
        # 简化实现：直接调用 collector.run(question)，其内部会做搜索+报告；我们只暴露「先 collect 再 report」的接口给未来扩展。
        # 更合理做法：ReActAgent 支持「仅搜索模式」返回 observations。这里先做一个兼容实现：若 collector 有 run_collect_only，用；否则用 run 得到完整答案并认为 observations 已在内部用过。
        if getattr(self.collector_agent, "run_collect_only", None):
            return self.collector_agent.run_collect_only(question, **kwargs)
        # 否则编排器等价于单 Agent：先跑 collector（内部已含报告），返回最终答案
        return self.collector_agent.run(question, **kwargs)

    def run(self, question: str, **kwargs) -> str:
        """先收集数据，再生成报告；返回完整报告。若 collector 无 run_collect_only，则等价于 collector.run(question)。"""
        if getattr(self.collector_agent, "run_collect_only", None):
            observations = self.collector_agent.run_collect_only(question, **kwargs)
            recent = getattr(self.collector_agent, "_format_recent_dialogue", lambda: "（无）")()
            from datetime import datetime
            current_date = datetime.now().strftime("%Y年%m月%d日 %H:%M")
            history = getattr(self.collector_agent, "get_history", lambda: [])()
            return self.report_generator.generate(
                question=question,
                observations=observations,
                recent_dialogue=recent,
                current_date=current_date,
                conversation_history=history,
                is_fixed_template=getattr(self.collector_agent, "prompt_template", "") and "价格位置" in getattr(self.collector_agent, "prompt_template", ""),
                **kwargs
            )
        return self.collector_agent.run(question, **kwargs)

    def run_stream(self, question: str, **kwargs) -> Iterator[str]:
        """先收集数据，再流式生成报告。"""
        if not getattr(self.collector_agent, "run_collect_only", None):
            full = self.collector_agent.run(question, **kwargs)
            yield full
            return
        observations = self.collector_agent.run_collect_only(question, **kwargs)
        recent = getattr(self.collector_agent, "_format_recent_dialogue", lambda: "（无）")()
        from datetime import datetime
        from ...core.message import Message
        current_date = datetime.now().strftime("%Y年%m月%d日 %H:%M")
        history = getattr(self.collector_agent, "get_history", lambda: [])()
        acc = []
        for chunk in self.report_generator.generate_stream(
            question=question,
            observations=observations,
            recent_dialogue=recent,
            current_date=current_date,
            conversation_history=history,
            is_fixed_template="价格位置" in getattr(self.collector_agent, "prompt_template", ""),
            **kwargs
        ):
            acc.append(chunk)
            yield chunk
        full_answer = "".join(acc).strip()
        if getattr(self.collector_agent, "add_message", None):
            self.collector_agent.add_message(Message(question, "user"))
            self.collector_agent.add_message(Message(full_answer, "assistant"))
