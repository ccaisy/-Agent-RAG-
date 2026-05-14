# src/schemas.py
# 核心 Schema 定义 —— 统一 Agent Tool Calling 与 RAG 的输入输出协议
# 使用 Python dataclass 作为运行时校验，JSON Schema 用于 API 文档/外部对接

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


# ============================================================
# 1. 通用消息协议
# ============================================================

class Role(str, Enum):
    """消息角色枚举"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """单条消息，兼容 OpenAI Chat Completions 格式"""
    role: Role
    content: str
    name: Optional[str] = None       # 可选发送者名称
    tool_call_id: Optional[str] = None  # Tool 回复时关联的 call id

    def to_openai(self) -> dict:
        """转为 OpenAI API dict 格式（去除 None 字段）"""
        d = {"role": self.role.value, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d


# ============================================================
# 2. Agent Tool Calling 协议
# ============================================================

@dataclass
class ToolParameter:
    """Tool 参数定义"""
    name: str
    type: str            # "string" | "number" | "boolean" | "object" | "array"
    description: str
    required: bool = True
    enum: Optional[list[str]] = None  # 可选：限定枚举值


@dataclass
class ToolDefinition:
    """工具定义 —— Agent 可调用的单个 Tool 描述"""
    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)

    def to_openai_function(self) -> dict:
        """转为 OpenAI Function Calling 兼容格式"""
        props = {}
        required = []
        for p in self.parameters:
            prop_def: dict[str, Any] = {"type": p.type, "description": p.description}
            if p.enum:
                prop_def["enum"] = p.enum
            props[p.name] = prop_def
            if p.required:
                required.append(p.name)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                },
            },
        }


@dataclass
class ToolCall:
    """Agent 发出的单次工具调用"""
    id: str
    tool_name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """工具执行结果"""
    call_id: str
    tool_name: str
    output: str                     # 工具返回的文本内容
    success: bool = True
    error: Optional[str] = None     # 失败时的错误信息


# ============================================================
# 3. RAG 检索与生成协议
# ============================================================

@dataclass
class Document:
    """RAG 文档单元 —— 检索入库 & 召回出库的统一格式"""
    doc_id: str                     # 唯一标识符
    content: str                    # 文档正文
    metadata: dict[str, Any] = field(default_factory=dict)  # 来源、页码、URL 等


@dataclass
class Chunk:
    """文档分块 —— 向量检索的基本粒度"""
    chunk_id: str
    doc_id: str                     # 所属文档 ID
    content: str                    # 分块文本
    chunk_index: int = 0            # 在文档中的序号
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """单条检索结果"""
    chunk: Chunk
    score: float                    # 相似度分数

    def to_prompt_context(self) -> str:
        """将检索结果拼接为 Prompt 可用的上下文片段"""
        return f"[来源: {self.chunk.doc_id} | 相关度: {self.score:.3f}]\n{self.chunk.content}"


@dataclass
class RAGPayload:
    """RAG 一次完整的 Retrieve → Rerank → Generate 请求"""
    query: str                                  # 用户原始查询
    top_k: int = 5                              # 最终返回给 LLM 的文档数
    retrieval_results: list[RetrievalResult] = field(default_factory=list)
    reranked_results: list[RetrievalResult] = field(default_factory=list)  # Rerank 后重排结果


# ============================================================
# 4. 输出协议：统一最终响应
# ============================================================

@dataclass
class FinalResponse:
    """系统最终返回给用户的统一格式"""
    answer: str                                 # 自然语言回答
    sources: list[dict[str, Any]] = field(default_factory=list)  # 引用来源（doc_id, 片段）
    tool_trace: list[ToolResult] = field(default_factory=list)   # Agent 模式的工具调用链路
    usage: dict[str, int] = field(default_factory=dict)          # Token 消耗统计 {prompt, completion, total}
