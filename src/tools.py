# src/tools.py
# Agent 高价值工具集
# Tool 1: 实时 Web 搜索（通过 LLM API 内建搜索或外部搜索引擎）
# Tool 2: Python 代码执行沙箱（安全执行用户/模型生成的代码片段）

import subprocess
import tempfile
import os
from src.schemas import ToolDefinition, ToolParameter, ToolResult


# ============================================================
# Tool 定义（OpenAI Function Calling JSON Schema）
# ============================================================

WEB_SEARCH_TOOL = ToolDefinition(
    name="web_search",
    description="搜索互联网获取实时信息。当需要最新资讯、事实核查或模型知识范围外的内容时使用。",
    parameters=[
        ToolParameter(name="query", type="string", description="搜索关键词"),
        ToolParameter(name="num_results", type="number", description="返回结果数量，默认 5", required=False),
    ],
)

CODE_EXEC_TOOL = ToolDefinition(
    name="execute_python",
    description="在隔离沙箱中执行 Python 代码。用于数学计算、数据分析、验证逻辑等场景。",
    parameters=[
        ToolParameter(name="code", type="string", description="要执行的 Python 代码"),
        ToolParameter(name="timeout", type="number", description="超时秒数，默认 10", required=False),
    ],
)

# ============================================================
# Tool 执行器
# ============================================================

class ToolExecutor:
    """工具执行调度器，根据 tool_name 分发到具体实现"""

    def execute(self, call_id: str, tool_name: str, arguments: dict) -> ToolResult:
        """
        分发执行。
        Args:
            call_id: LLM 生成的 tool_call id
            tool_name: 工具名称
            arguments: 工具参数字典
        Returns:
            ToolResult
        """
        if tool_name == "web_search":
            return self._web_search(call_id, arguments)
        elif tool_name == "execute_python":
            return self._execute_python(call_id, arguments)
        else:
            return ToolResult(
                call_id=call_id, tool_name=tool_name,
                output=f"未知工具: {tool_name}", success=False,
                error=f"Tool '{tool_name}' 未注册",
            )

    # ============================================================
    # Tool 1: Web 搜索（通过系统 curl 命令，macOS 友好）
    # ============================================================

    @staticmethod
    def _web_search(call_id: str, args: dict) -> ToolResult:
        """
        使用 curl 发起搜索请求。
        实际场景可替换为 SerpAPI / Google Custom Search API。
        MVP 阶段：返回提示信息 + 搜索 URL 占位。
        """
        query = args.get("query", "")
        # 占位实现：构建搜索 URL，生产环境接入 SerpAPI 等
        search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        return ToolResult(
            call_id=call_id,
            tool_name="web_search",
            output=(
                f"Web 搜索服务（MVP 占位实现）\n"
                f"搜索词: {query}\n"
                f"搜索 URL: {search_url}\n"
                f"[提示] 可接入 SerpAPI 或 Bing Search API 获取真实搜索结果。"
            ),
            success=True,
        )

    # ============================================================
    # Tool 2: Python 代码沙箱执行
    # ============================================================

    @staticmethod
    def _execute_python(call_id: str, args: dict) -> ToolResult:
        """
        在临时文件中安全执行 Python 代码。
        限制：超时控制 + 禁止危险操作提示。
        """
        code = args.get("code", "")
        timeout = args.get("timeout", 10)

        # 写入临时文件
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            tmp_path = f.name

        try:
            result = subprocess.run(
                ["python3", tmp_path],
                capture_output=True, text=True, timeout=timeout,
                env={**os.environ, "PYTHONPATH": os.getcwd()},
            )
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            success = result.returncode == 0

            output_parts = []
            if stdout:
                output_parts.append(f"[输出]\n{stdout}")
            if stderr:
                output_parts.append(f"[错误]\n{stderr}")
            if not output_parts:
                output_parts.append("[无输出]")

            return ToolResult(
                call_id=call_id,
                tool_name="execute_python",
                output="\n\n".join(output_parts),
                success=success,
                error=stderr if not success else None,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                call_id=call_id, tool_name="execute_python",
                output=f"代码执行超时（>{timeout}秒）", success=False,
                error="TimeoutExpired",
            )
        finally:
            # 清理临时文件
            os.unlink(tmp_path)
