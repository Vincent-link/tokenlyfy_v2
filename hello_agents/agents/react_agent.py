"""ReAct Agent实现 - 推理与行动结合的智能体"""

import re
from datetime import datetime
from typing import Optional, List, Tuple
from ..core.agent import Agent
from ..core.llm import HelloAgentsLLM
from ..core.config import Config
from ..core.message import Message
from ..tools.registry import ToolRegistry

# 默认ReAct提示词模板
DEFAULT_REACT_PROMPT = """你是一个具备推理和行动能力的AI助手。你可以通过思考分析问题，然后调用合适的工具来获取信息，最终给出准确的答案。

## 可用工具
{tools}

## 工作流程
请严格按照以下格式进行回应，每次只能执行一个步骤：

Thought: 分析问题，确定需要什么信息，制定研究策略。
Action: 选择合适的工具获取信息，格式为：
- `{{tool_name}}[{{tool_input}}]`：调用工具获取信息。
- `Finish[你的直接回答]`：当你有足够信息时，在方括号内写出**对用户问题的直接回答**（具体结论、预测数值或建议），不要只写「我接下来要做什么」或「我已收集了哪些信息」。

## 记忆工具（若有 memory 工具）
- 用户提供个人信息或希望被记住的内容时，用 `memory[store=内容]` 存储。
- 需要回忆之前对话中的信息时，用 `memory[recall=查询关键词]` 检索。
- 需要记忆摘要时，用 `memory[action=summary]` 获取。

## 重要提醒
1. 每次回应必须包含Thought和Action两部分
2. 工具调用的格式必须严格遵循：工具名[参数]
3. 只有当你确信有足够信息回答问题时，才使用Finish
4. 使用Finish时，方括号内必须是**对问题的直接回答**：若用户问预测，则写具体预测结论（如价格区间、趋势）；若问事实，则写事实结论。不要写「需要综合分析」等计划性表述
5. 若用户问的是**具体数据或指标**（如市场情绪、恐惧贪婪指数、资金费率、价格），Finish 中必须**明确写出查到的数值**（如「恐惧贪婪指数 14，极度恐惧」「资金费率 0.01%」），不可只写「需要确认」「需要获取」等，也不可只写思考过程不写答案
6. 如果工具返回的信息不够，继续使用其他工具或相同工具的不同参数
7. 对搜索结果中的日期、价格等时效性信息保持警惕，优先引用与「当前日期」一致、来源明确的数据；若搜索结果中的日期晚于当前日期或明显不合理，应在回答中说明并避免采信

## 最近对话（供理解上下文）
{recent_dialogue}

## 当前任务
**当前日期与时间：** {current_date}（请据此判断搜索结果中的日期是否合理）
**Question:** {question}

## 执行历史
{history}

现在开始你的推理和行动："""

# ============================================================
# 分析类提示词（搜索阶段）：只负责搜索收集信息，Finish[done] 表示搜索完毕
# 报告生成由 run() 中独立的 LLM 调用完成，不在 Finish[] 里写报告
# ============================================================

# 行情/分析类结构化报告（搜索阶段提示词）
MARKET_ANALYSIS_REACT_PROMPT = """你是一个行情分析助手的**信息收集模块**。你的任务是通过搜索工具尽可能多地收集与用户问题相关的数据（价格、技术指标、资金流向、情绪指标等），收集完毕后用 Finish[done] 结束。

## 可用工具
{tools}

## 工作流程
每次只能执行一个步骤，格式如下：

Thought: 分析还缺什么信息，制定下一步搜索策略。
Action: 选择合适的工具获取信息，格式为：
- `{{tool_name}}[{{tool_input}}]`：调用工具搜索信息。
- `Finish[done]`：当你认为已收集到足够信息时（搜索 2～3 次即可），用此结束搜索阶段。

## 搜索策略（优先一次调用，减少等待）
1. **`crypto_analysis`** 【首选】一次并行获取价格+技术+恐惧贪婪+合约数据，如 `crypto_analysis[BTC 1h]` 或 `crypto_analysis[ETH 4h]`，周期缺省默认 1h。**单币分析优先用此，可节省 3～4 次调用**。
2. 若需多币或单工具，再用 `crypto_price`、`technical`、`fear_greed`、`futures_data`。
3. **`search`** 仅当需新闻或外部资讯时补充（0～1 次）。

**注意**：单币分析优先 `crypto_analysis[币种 周期]`，通常 1～2 次工具调用即可完成。

## 重要提醒
1. 每次回应必须包含 Thought 和 Action 两部分。
2. Finish[done] 只表示搜索完毕，**不要在方括号里写报告或分析**——报告会由系统另行生成。
3. 对搜索结果中的日期保持警惕，优先获取与当前日期一致的数据。
4. 当前日期与时间：{current_date}。

## 最近对话（供理解上下文）
{recent_dialogue}

## 当前任务
**Question:** {question}

## 执行历史
{history}

现在开始搜索信息："""

# 个性化分析（搜索阶段提示词）
PERSONALIZED_ANALYSIS_REACT_PROMPT = """你是一个行情分析助手的**信息收集模块**。你的任务是根据用户问题，通过搜索工具收集相关数据，收集完毕后用 Finish[done] 结束。

## 可用工具
{tools}

## 工作流程
每次只能执行一个步骤，格式如下：

Thought: 分析用户问题的关键点，确定还需要搜索什么信息。
Action: 选择合适的工具获取信息，格式为：
- `{{tool_name}}[{{tool_input}}]`：调用工具搜索信息。
- `Finish[done]`：当你认为已收集到足够信息时（2～3 次即可），用此结束搜索阶段。

## 搜索策略（优先一次调用，减少等待）
1. **`crypto_analysis`** 【首选】一次并行获取价格+技术+恐惧贪婪+合约数据，如 `crypto_analysis[BTC 1h]` 或 `crypto_analysis[ETH 4h]`。单币分析优先用此，可节省 3～4 次调用。
2. 多币或单工具时再用 `crypto_price`、`technical`、`fear_greed`、`futures_data`。
3. **`search`** 仅当需新闻或外部资讯时补充（0～1 次）。

**注意**：单币分析优先 `crypto_analysis[币种 周期]`，通常 1～2 次工具调用即可。

## 重要提醒
1. 每次回应必须包含 Thought 和 Action 两部分。
2. Finish[done] 只表示搜索完毕，**不要在方括号里写报告或回答**——回答会由系统另行生成。
3. 若「最近对话」非空，当前问题可能是追问（如「短线」「小时线」），请结合上下文理解要搜什么。
4. 当前日期与时间：{current_date}。

## 最近对话（供理解上下文）
{recent_dialogue}

## 当前任务
**Question:** {question}

## 执行历史
{history}

现在开始搜索信息："""

class ReActAgent(Agent):
    """
    ReAct (Reasoning and Acting) Agent
    
    结合推理和行动的智能体，能够：
    1. 分析问题并制定行动计划
    2. 调用外部工具获取信息
    3. 基于观察结果进行推理
    4. 迭代执行直到得出最终答案
    
    这是一个经典的Agent范式，特别适合需要外部信息的任务。
    """
    
    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        tool_registry: Optional[ToolRegistry] = None,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_steps: int = 5,
        custom_prompt: Optional[str] = None
    ):
        """
        初始化ReActAgent

        Args:
            name: Agent名称
            llm: LLM实例
            tool_registry: 工具注册表（可选，如果不提供则创建空的工具注册表）
            system_prompt: 系统提示词
            config: 配置对象
            max_steps: 最大执行步数
            custom_prompt: 自定义提示词模板
        """
        super().__init__(name, llm, system_prompt, config)

        # 如果没有提供tool_registry，创建一个空的
        if tool_registry is None:
            self.tool_registry = ToolRegistry()
        else:
            self.tool_registry = tool_registry

        self.max_steps = max_steps
        self.current_history: List[str] = []

        # 设置提示词模板：用户自定义优先，否则使用默认模板
        self.prompt_template = custom_prompt if custom_prompt else DEFAULT_REACT_PROMPT

    def add_tool(self, tool):
        """
        添加工具到工具注册表
        支持MCP工具的自动展开

        Args:
            tool: 工具实例(可以是普通Tool或MCPTool)
        """
        # 检查是否是MCP工具
        if hasattr(tool, 'auto_expand') and tool.auto_expand:
            # MCP工具会自动展开为多个工具
            if hasattr(tool, '_available_tools') and tool._available_tools:
                for mcp_tool in tool._available_tools:
                    # 创建包装工具
                    from ..tools.base import Tool
                    wrapped_tool = Tool(
                        name=f"{tool.name}_{mcp_tool['name']}",
                        description=mcp_tool.get('description', ''),
                        func=lambda input_text, t=tool, tn=mcp_tool['name']: t.run({
                            "action": "call_tool",
                            "tool_name": tn,
                            "arguments": {"input": input_text}
                        })
                    )
                    self.tool_registry.register_tool(wrapped_tool)
                print(f"✅ MCP工具 '{tool.name}' 已展开为 {len(tool._available_tools)} 个独立工具")
            else:
                self.tool_registry.register_tool(tool)
        else:
            self.tool_registry.register_tool(tool)

    def _is_analysis_template(self) -> bool:
        """判断当前使用的是否为分析类模板（两阶段模式）"""
        return "信息收集模块" in self.prompt_template

    def _check_crypto_intent(self, question: str, recent_dialogue: str) -> Optional[str]:
        """检查用户问题是否属于加密货币投研领域。
        
        若非加密问题，返回礼貌拒绝文本；若是加密问题，返回 None（放行）。
        使用关键词快速判断，避免额外 LLM 调用。
        """
        text = question.lower().strip()
        # 加密货币相关关键词
        crypto_keywords = (
            "btc", "eth", "sol", "bnb", "xrp", "doge", "ada", "dot", "link",
            "比特币", "以太坊", "加密", "币", "区块链", "链上", "defi", "nft",
            "k线", "kline", "macd", "rsi", "布林", "支撑", "阻力", "均线",
            "合约", "资金费率", "杠杆", "做多", "做空", "多头", "空头",
            "涨", "跌", "行情", "走势", "价格", "市值", "抄底", "追高",
            "牛市", "熊市", "减半", "挖矿", "矿工", "gas", "质押", "staking",
            "恐惧", "贪婪", "fear", "greed", "whale", "巨鲸",
            "交易所", "binance", "coinbase", "okx", "bybit",
            "usdt", "usdc", "稳定币", "token", "代币",
            "短线", "中线", "长线", "日线", "小时线", "周线", "月线",
            "etf", "灰度", "grayscale", "web3", "crypto", "bitcoin", "ethereum",
        )
        # 结合上下文判断：若最近对话是加密话题，则简短追问也放行
        context = text + " " + recent_dialogue.lower()
        if any(kw in context for kw in crypto_keywords):
            return None  # 放行
        
        return (
            "🙏 我是**加密货币投研助手**，专注于加密货币的行情分析、技术指标解读和操作建议。\n\n"
            "您的问题似乎不在加密货币投研范围内。我可以帮您分析：\n"
            "- 📊 某个币种的行情走势（如「BTC 明天怎么走」「ETH 技术面分析」）\n"
            "- 📈 技术指标解读（如「RSI 超卖了吗」「小时线支撑阻力」）\n"
            "- 😱 市场情绪（如「当前恐惧贪婪指数」）\n"
            "- 💡 操作建议（如「BTC 能抄底吗」「短线怎么操作」）\n\n"
            "请换一个加密货币相关的问题试试吧！"
        )

    @staticmethod
    def _load_knowledge() -> str:
        """加载加密货币技术分析方法论知识库"""
        import os
        knowledge_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "knowledge", "crypto_analysis.md"
        )
        try:
            with open(knowledge_path, "r", encoding="utf-8") as f:
                content = f.read()
            # 截取关键部分，避免 prompt 过长
            if len(content) > 2800:
                content = content[:2800] + "\n... (更多内容已省略)"
            return content
        except FileNotFoundError:
            return ""

    @staticmethod
    def _load_history_cases() -> str:
        """加载历史行情复盘案例（P2：历史对比参考）"""
        import os
        path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "knowledge", "crypto_history_cases.md"
        )
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            if len(content) > 2200:
                content = content[:2200] + "\n... (更多案例已省略)"
            return content
        except FileNotFoundError:
            return ""

    def _get_previous_prediction(self, max_content_len: int = 600) -> str:
        """从对话历史中提取最近一次分析预测，供「前次预测回顾」使用（P2）"""
        history = self.get_history()
        if not history:
            return ""
        # 从后往前找最近一条 assistant 消息，且像分析报告（含结论/置信度/偏向等）
        for m in reversed(history):
            if m.role != "assistant" or not m.content:
                continue
            content = (m.content or "").strip()
            markers = ("置信度", "偏向", "结论", "预测", "短线", "中线", "建议", "抄底", "减仓")
            if any(kw in content for kw in markers):
                if len(content) > max_content_len:
                    content = content[:max_content_len] + "…"
                return content
        return ""

    def _generate_report(self, question: str, history_str: str,
                         current_date: str, recent_dialogue: str, **kwargs) -> str:
        """基于搜索阶段收集的观察内容，用独立 LLM 调用生成完整分析报告。
        
        不受 Finish[...] 方括号限制，模型可自由写长文、带链接、带表格。
        自动注入技术分析方法论知识库，供模型参考指标判读规则。
        """
        is_fixed = "价格位置" in self.prompt_template

        # 加载知识库
        knowledge = self._load_knowledge()
        knowledge_section = ""
        if knowledge:
            knowledge_section = f"""## 技术分析方法论参考（请依据此框架解读指标）
{knowledge}
"""

        # P2：历史行情案例（类似市场状况下的走势复盘）
        history_cases = self._load_history_cases()
        history_section = ""
        if history_cases:
            history_section = f"""
## 历史类似案例参考（P2：上次类似情况怎么走的）
请根据当前数据（恐惧贪婪、RSI、资金费率等）挑选最相近的 1～2 个案例作参考，在报告中简要提及，增强结论的可比性。不要机械套用，仅作参考。
{history_cases}
"""

        # P2：前次预测回顾（利用对话历史）
        prev_pred = self._get_previous_prediction()
        prev_section = ""
        if prev_pred:
            prev_section = f"""
## 前次预测回顾
上次分析中我们的结论/建议摘要如下。可在报告中简要提及，并提醒用户对照近期走势自行验证，提升可信度。
---
{prev_pred}
---
"""
        
        # 通用分析要求（多空论据 + 置信度 + 数据交叉验证 + 历史对比）
        analysis_rules = """## 分析原则（必须遵守）
1. **数据交叉验证**：不只罗列数据，要分析不同指标之间的关系。例如：RSI 超卖 + Fear & Greed 极度恐惧 + 价格触及布林带下轨 = 强超卖信号。
2. **多空论据对比**：分别列出看多和看空的论据，不要一边倒。
3. **置信度评估**：在结论中给出判断的置信度（如「偏向震荡反弹，置信度 65%」），让用户了解确定性。
4. **引用具体数值**：必须写出查到的指标具体数值（如 RSI=28.5 而非"RSI 偏低"），让结论有据可查。
5. **来源标注**：在关键数据后标注来源，格式：[来源](url)。
6. **历史对比（P2）**：若提供了「历史类似案例」，请根据当前条件挑选最相近案例简要引用；若有「前次预测回顾」，可加一句提醒用户对照验证。"""

        if is_fixed:
            report_prompt = f"""你是一个专业的加密货币分析师。根据以下收集到的数据，写出一份完整的分析报告。

{analysis_rules}

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
## 收集到的数据
{history_str}

请直接输出完整报告（只输出报告，不要写 Thought/Action/Finish）："""
        else:
            report_prompt = f"""你是一个专业的加密货币分析师。根据以下收集到的数据和用户问题，写出一份**紧扣问题**的分析回答。

{analysis_rules}

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
## 收集到的数据
{history_str}

请直接输出完整回答（只输出回答，不要写 Thought/Action/Finish）："""

        report = self.llm.invoke([{"role": "user", "content": report_prompt}], **kwargs)
        return (report or "").strip() or "抱歉，报告生成失败，请重试。"

    def _format_recent_dialogue(self, max_turns: int = 3, max_content_len: int = 800) -> str:
        """格式化最近对话供注入 prompt，便于模型根据上下文理解当前问题。"""
        history = self.get_history()
        if not history:
            return "（无此前对话）"
        # 取最近 max_turns 轮（每轮 user + assistant）
        recent = history[-(max_turns * 2) :]
        lines = []
        for m in recent:
            role = "用户" if m.role == "user" else "助手"
            content = (m.content or "").strip()
            if len(content) > max_content_len:
                content = content[:max_content_len] + "…"
            lines.append(f"{role}: {content}")
        return "\n".join(lines) if lines else "（无此前对话）"

    def run(self, input_text: str, **kwargs) -> str:
        """
        运行ReAct Agent
        
        Args:
            input_text: 用户问题
            **kwargs: 其他参数
            
        Returns:
            最终答案
        """
        self.current_history = []
        current_step = 0
        recent_dialogue = self._format_recent_dialogue()
        
        # 分析类模板：先做意图检查，非加密问题直接拒绝
        if self._is_analysis_template():
            rejection = self._check_crypto_intent(input_text, recent_dialogue)
            if rejection:
                self.add_message(Message(input_text, "user"))
                self.add_message(Message(rejection, "assistant"))
                return rejection
        
        print(f"\n🤖 {self.name} 开始处理问题: {input_text}")
        
        while current_step < self.max_steps:
            current_step += 1
            print(f"\n--- 第 {current_step} 步 ---")
            
            # 构建提示词（注入当前日期与最近对话，供模型判断时效性和上下文）
            tools_desc = self.tool_registry.get_tools_description()
            history_str = "\n".join(self.current_history)
            current_date = datetime.now().strftime("%Y年%m月%d日 %H:%M")
            prompt = self.prompt_template.format(
                tools=tools_desc,
                question=input_text,
                history=history_str,
                current_date=current_date,
                recent_dialogue=recent_dialogue
            )
            # 最后一步时强制要求给出结论，避免步数用尽仍无 Finish
            if current_step == self.max_steps:
                prompt += "\n\n【重要】你已到达最后一步，请在本轮必须使用 Finish[你的结论] 给出最终回答，即使信息不完整也要基于已有观察总结。"
            
            # 调用LLM
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm.invoke(messages, **kwargs)
            
            if not response_text:
                print("❌ 错误：LLM未能返回有效响应。")
                break
            
            # 解析输出
            thought, action = self._parse_output(response_text)
            
            if thought:
                print(f"🤔 思考: {thought}")
            
            if not action:
                print("⚠️ 警告：未能解析出有效的Action，流程终止。")
                break
            
            # 检查是否完成
            if action.startswith("Finish"):
                # 判断是否为分析类模板（两阶段模式）
                is_analysis_prompt = self._is_analysis_template()
                
                if is_analysis_prompt:
                    # ===== 分析类：搜索阶段结束，进入独立的报告生成阶段 =====
                    print("📝 搜索完毕，正在生成分析报告…")
                    final_answer = self._generate_report(
                        input_text, history_str, current_date, recent_dialogue, **kwargs
                    )
                else:
                    # ===== 普通 ReAct：Finish 里的内容就是答案 =====
                    final_answer = self._parse_action_input(action)
                    if not final_answer and thought:
                        final_answer = thought.strip()
                
                # 保存到历史记录
                self.add_message(Message(input_text, "user"))
                self.add_message(Message(final_answer, "assistant"))
                
                return final_answer
            
            # 执行工具调用
            tool_name, tool_input = self._parse_action(action)
            if not tool_name or tool_input is None:
                self.current_history.append("Observation: 无效的Action格式，请检查。")
                continue
            
            print(f"🎬 行动: {tool_name}[{tool_input}]")
            
            # 调用工具
            observation = self.tool_registry.execute_tool(tool_name, tool_input)
            print(f"👀 观察: {observation}")
            
            # 更新历史
            self.current_history.append(f"Action: {action}")
            self.current_history.append(f"Observation: {observation}")
        
        # 达到最大步数：分析类仍然尝试基于已有观察生成报告
        if self._is_analysis_template() and self.current_history:
            print("⏰ 已达到最大步数，基于已有观察生成报告…")
            history_str = "\n".join(self.current_history)
            current_date = datetime.now().strftime("%Y年%m月%d日 %H:%M")
            final_answer = self._generate_report(
                input_text, history_str, current_date, recent_dialogue, **kwargs
            )
        else:
            print("⏰ 已达到最大步数，流程终止。")
            final_answer = "抱歉，我无法在限定步数内完成这个任务。"
        
        # 保存到历史记录
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))
        
        return final_answer
    
    def _parse_output(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """解析LLM输出，提取思考和行动"""
        thought_match = re.search(r"Thought: (.*)", text)
        action_match = re.search(r"Action: (.*)", text)
        
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        
        return thought, action
    
    def _parse_action(self, action_text: str) -> Tuple[Optional[str], Optional[str]]:
        """解析行动文本，提取工具名称和输入。
        
        使用手动切片而非正则，避免嵌套方括号（如 Markdown 链接 [text](url)）导致截断。
        """
        bracket_pos = action_text.find("[")
        if bracket_pos == -1:
            return None, None
        tool_name = action_text[:bracket_pos].strip()
        # 取第一个 [ 到最后一个 ] 之间的全部内容
        last_bracket = action_text.rfind("]")
        if last_bracket <= bracket_pos:
            return None, None
        tool_input = action_text[bracket_pos + 1 : last_bracket]
        return tool_name, tool_input
    
    def _parse_action_input(self, action_text: str) -> str:
        """解析行动输入（取第一个 [ 到最后一个 ] 之间的内容）"""
        bracket_pos = action_text.find("[")
        if bracket_pos == -1:
            return ""
        last_bracket = action_text.rfind("]")
        if last_bracket <= bracket_pos:
            return ""
        return action_text[bracket_pos + 1 : last_bracket]
