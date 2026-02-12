"""Plan and Solve Agent实现 - 分解规划与逐步执行的智能体"""

import ast
import re
from typing import Optional, List, Dict, TYPE_CHECKING
from ..core.agent import Agent
from ..core.llm import HelloAgentsLLM
from ..core.config import Config
from ..core.message import Message

if TYPE_CHECKING:
    from ..tools.registry import ToolRegistry

# 默认规划器提示词模板
DEFAULT_PLANNER_PROMPT = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""

# 默认执行器提示词模板
DEFAULT_EXECUTOR_PROMPT = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决"当前步骤"。若配置了工具，需要数据时可使用 [TOOL_CALL:工具名:参数] 调用；否则直接输出答案。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请输出针对"当前步骤"的回答（可直接输出答案，或先调用工具获取数据）:
"""

class Planner:
    """规划器 - 负责将复杂问题分解为简单步骤"""

    def __init__(self, llm_client: HelloAgentsLLM, prompt_template: Optional[str] = None):
        self.llm_client = llm_client
        self.prompt_template = prompt_template if prompt_template else DEFAULT_PLANNER_PROMPT

    def plan(self, question: str, **kwargs) -> List[str]:
        """
        生成执行计划

        Args:
            question: 要解决的问题
            **kwargs: LLM调用参数

        Returns:
            步骤列表
        """
        prompt = self.prompt_template.format(question=question)
        messages = [{"role": "user", "content": prompt}]

        print("--- 正在生成计划 ---")
        response_text = self.llm_client.invoke(messages, **kwargs) or ""
        print(f"✅ 计划已生成:\n{response_text}")

        try:
            # 提取Python代码块中的列表
            plan_str = response_text.split("```python")[1].split("```")[0].strip()
            plan = ast.literal_eval(plan_str)
            return plan if isinstance(plan, list) else []
        except (ValueError, SyntaxError, IndexError) as e:
            print(f"❌ 解析计划时出错: {e}")
            print(f"原始响应: {response_text}")
            return []
        except Exception as e:
            print(f"❌ 解析计划时发生未知错误: {e}")
            return []

class Executor:
    """执行器 - 负责按计划逐步执行，支持工具调用"""

    _TOOL_CALL_PATTERN = re.compile(r'\[TOOL_CALL:([^:]+):([^\]]+)\]')

    def __init__(
        self,
        llm_client: HelloAgentsLLM,
        prompt_template: Optional[str] = None,
        tool_registry: Optional["ToolRegistry"] = None,
    ):
        self.llm_client = llm_client
        self.prompt_template = prompt_template if prompt_template else DEFAULT_EXECUTOR_PROMPT
        self.tool_registry = tool_registry

    def _parse_tool_calls(self, text: str) -> List[Dict[str, str]]:
        """解析文本中的工具调用 [TOOL_CALL:name:params]"""
        out = []
        for m in self._TOOL_CALL_PATTERN.finditer(text):
            out.append({
                "tool_name": m.group(1).strip(),
                "parameters": m.group(2).strip(),
                "original": m.group(0),
            })
        return out

    def _execute_tool(self, tool_name: str, parameters: str) -> str:
        """执行单个工具调用"""
        if not self.tool_registry:
            return "❌ 未配置工具注册表"
        try:
            result = self.tool_registry.execute_tool(tool_name, parameters)
            return f"🔧 工具 {tool_name} 执行结果：\n{result}"
        except Exception as e:
            return f"❌ 工具 {tool_name} 调用失败: {e}"

    def _run_step_with_tools(
        self, question: str, plan: List[str], history: str, step: str, step_idx: int, **kwargs
    ) -> str:
        """执行单步，支持多轮工具调用"""
        base_prompt = self.prompt_template.format(
            question=question,
            plan=plan,
            history=history if history else "无",
            current_step=step,
        )
        if self.tool_registry:
            tools_desc = self.tool_registry.get_tools_description()
            if tools_desc and tools_desc != "暂无可用工具":
                base_prompt = (
                    f"# 可用工具（需要数据时请调用）\n{tools_desc}\n\n"
                    "调用格式: [TOOL_CALL:工具名:参数]，例如 [TOOL_CALL:crypto_price:BTC,ETH]\n\n"
                    + base_prompt
                )

        messages = [{"role": "user", "content": base_prompt}]
        max_tool_iters = 5
        iters = 0

        while iters < max_tool_iters:
            response_text = self.llm_client.invoke(messages, **kwargs) or ""
            tool_calls = self._parse_tool_calls(response_text)

            if not tool_calls:
                return response_text.strip()

            results = []
            for call in tool_calls:
                result = self._execute_tool(call["tool_name"], call["parameters"])
                results.append(result)
                params_preview = call["parameters"][:40] + "..." if len(call["parameters"]) > 40 else call["parameters"]
                print(f"  🔧 调用 {call['tool_name']}[{params_preview}] -> 成功")
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": "Observation:\n" + "\n\n".join(results) + "\n\n请基于以上工具结果，继续完成当前步骤并输出最终答案。"})
            iters += 1

        return response_text.strip()

    def execute(self, question: str, plan: List[str], **kwargs) -> str:
        """
        按计划执行任务，每步可调用工具获取数据

        Args:
            question: 原始问题
            plan: 执行计划
            **kwargs: LLM调用参数

        Returns:
            最终答案
        """
        history = ""
        final_answer = ""

        print("\n--- 正在执行计划 ---")
        for i, step in enumerate(plan, 1):
            print(f"\n-> 正在执行步骤 {i}/{len(plan)}: {step}")
            response_text = self._run_step_with_tools(
                question, plan, history, step, i, **kwargs
            )
            history += f"步骤 {i}: {step}\n结果: {response_text}\n\n"
            final_answer = response_text
            print(f"✅ 步骤 {i} 已完成")

        return final_answer

class PlanAndSolveAgent(Agent):
    """
    Plan and Solve Agent - 分解规划与逐步执行的智能体
    
    这个Agent能够：
    1. 将复杂问题分解为简单步骤
    2. 按照计划逐步执行
    3. 维护执行历史和上下文
    4. 得出最终答案
    
    特别适合多步骤推理、数学问题、复杂分析等任务。
    """
    
    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        custom_prompts: Optional[Dict[str, str]] = None,
        tool_registry: Optional["ToolRegistry"] = None,
    ):
        """
        初始化PlanAndSolveAgent

        Args:
            name: Agent名称
            llm: LLM实例
            system_prompt: 系统提示词
            config: 配置对象
            custom_prompts: 自定义提示词模板 {"planner": "", "executor": ""}
            tool_registry: 工具注册表（可选），提供后执行器可调用工具获取数据
        """
        super().__init__(name, llm, system_prompt, config)

        # 设置提示词模板：用户自定义优先，否则使用默认模板
        if custom_prompts:
            planner_prompt = custom_prompts.get("planner")
            executor_prompt = custom_prompts.get("executor")
        else:
            planner_prompt = None
            executor_prompt = None

        self.tool_registry = tool_registry
        self.planner = Planner(self.llm, planner_prompt)
        self.executor = Executor(self.llm, executor_prompt, tool_registry=tool_registry)
    
    def run(self, input_text: str, **kwargs) -> str:
        """
        运行Plan and Solve Agent
        
        Args:
            input_text: 要解决的问题
            **kwargs: 其他参数
            
        Returns:
            最终答案
        """
        print(f"\n🤖 {self.name} 开始处理问题: {input_text}")
        
        # 1. 生成计划
        plan = self.planner.plan(input_text, **kwargs)
        if not plan:
            final_answer = "无法生成有效的行动计划，任务终止。"
            print(f"\n--- 任务终止 ---\n{final_answer}")
            
            # 保存到历史记录
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(final_answer, "assistant"))
            
            return final_answer
        
        # 2. 执行计划
        final_answer = self.executor.execute(input_text, plan, **kwargs)
        print(f"\n--- 任务完成 ---\n最终答案: {final_answer}")
        
        # 保存到历史记录
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))
        
        return final_answer
