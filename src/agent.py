# src/agent.py
# Agent 核心循环：CoT 思维链 + ReAct + Self-Correction（项目算法亮点）
# CoT: 强制模型在行动前先输出推理过程
# Self-Correction: 工具失败/结果不佳时自动修正重试

from src.schemas import Message, Role, ToolResult, FinalResponse
from src.llm_client import LLMClient
from src.tools import ToolExecutor


# CoT 系统提示词 —— 注入到每次 Agent 对话
COT_SYSTEM_PROMPT = """你是一个具备推理能力的 AI Agent。请遵循以下流程：

## 思考流程
1. **分析问题**：理解用户的问题，明确需要什么信息。
2. **制定计划**：决定是否需要调用工具，如果需要，选择最合适的工具。
3. **执行行动**：调用工具获取信息。
4. **验证结果**：检查工具返回的信息是否充分、正确。
5. **修正错误**：如果结果不正确或不完整，尝试其他方法或工具。
6. **输出答案**：基于所有信息给出最终答案。

## 重要规则
- 每次只调用一个工具，等结果返回后再决定下一步。
- 如果工具调用失败，分析原因并尝试修正（如修改参数、换用其他工具）。
- 最终答案必须基于实际获取的数据，不得猜测或编造。"""


class ReActAgent:
    """
    CoT + ReAct + Self-Correction Agent。
    亮点：
    - CoT：注入结构化思维链 Prompt，让模型"先想再做"
    - Self-Correction：工具失败时自动分析错误并修正
    - 工具执行：通过 ToolExecutor 调度真实工具
    """

    def __init__(self, llm: LLMClient, tools: list[dict] | None = None,
                 tool_executor: ToolExecutor | None = None,
                 max_turns: int = 5):
        """
        Args:
            llm: LLMClient 实例
            tools: OpenAI Function Calling 格式的工具列表
            tool_executor: ToolExecutor 实例（如果为 None 则使用虚方法）
            max_turns: 最大推理轮次（防止死循环）
        """
        self.llm = llm
        self.tools = tools or []
        self.tool_executor = tool_executor or ToolExecutor()
        self.max_turns = max_turns
        self.tool_trace: list[ToolResult] = []

    def run(self, user_query: str) -> FinalResponse:
        """
        执行一次完整的 Agent 对话。
        Args:
            user_query: 用户输入
        Returns:
            FinalResponse（含答案、工具链路、Token 消耗）
        """
        self.tool_trace = []  # 每次 run 前清空 trace
        total_usage = {"prompt": 0, "completion": 0, "total": 0}
        messages: list[dict] = [
            Message(role=Role.SYSTEM, content=COT_SYSTEM_PROMPT).to_openai(),
            Message(role=Role.USER, content=user_query).to_openai(),
        ]

        for turn in range(self.max_turns):
            response = self.llm.chat(messages=messages, tools=self.tools or None)
            msg = response.choices[0].message
            # 累计 token 消耗
            for k in total_usage:
                total_usage[k] += getattr(response.usage, f"{k}_tokens", 0)

            # 模型要调用工具 → 执行 → 自修正检查
            if msg.tool_calls:
                # 1. 先将 assistant 消息（含 tool_calls）加入历史
                #    OpenAI API 要求 tool 消息必须紧跟对应的 assistant(tool_calls) 消息
                messages.append(msg.model_dump())

                # 2. 执行每个工具并将结果加入历史
                for tc in msg.tool_calls:
                    result = self._execute_tool(tc)
                    self.tool_trace.append(result)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result.output,
                    })

                # 3. Self-Correction：追加反思提示，驱动模型自我修正
                correction_prompt = self._build_correction_prompt()
                messages.append({"role": "user", "content": correction_prompt})
                continue

            # 模型直接给出文本回复 → 终止
            answer = msg.content or ""
            return FinalResponse(answer=answer, tool_trace=self.tool_trace, usage=total_usage)

        return FinalResponse(
            answer="Agent 推理轮次已达上限，请检查任务复杂度或工具链。",
            tool_trace=self.tool_trace,
            usage=total_usage,
        )

    # ============================================================
    # 自修正 (Self-Correction) —— 核心算法亮点
    # ============================================================

    def _build_correction_prompt(self) -> str:
        """
        构建自修正提示词。
        检查最近一次工具调用的结果质量，驱动模型自我纠正。
        """
        last_result = self.tool_trace[-1] if self.tool_trace else None
        status = "成功" if (last_result and last_result.success) else "失败"

        prompt = (
            f"【自修正检查 - 第 {len(self.tool_trace)} 步】\n"
            f"上一步工具调用状态：{status}\n"
        )
        if last_result and last_result.error:
            prompt += f"错误信息：{last_result.error}\n"
            prompt += "请分析错误原因，并决定：修改参数重试 / 换用其他工具 / 基于已有信息给出答案。\n"
        else:
            prompt += (
                "请评估当前信息是否足够回答用户问题：\n"
                "1. 信息是否完整？\n"
                "2. 是否需要额外数据？\n"
                "3. 如果已充分，请直接给出最终答案。\n"
            )
        return prompt

    # ============================================================
    # 工具执行（通过 ToolExecutor 调度）
    # ============================================================

    def _execute_tool(self, tool_call) -> ToolResult:
        """
        执行工具调用，优先使用 ToolExecutor。
        Args:
            tool_call: OpenAI tool_call 对象
        """
        import json
        name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            args = {}
        return self.tool_executor.execute(
            call_id=tool_call.id,
            tool_name=name,
            arguments=args,
        )
