# src/rag.py
# RAG 核心链路：Retrieve → Rerank → Generation
# 输入用户查询 → 检索相关文档分块 → 重排序筛选 → 拼接上下文生成答案

from src.schemas import RAGPayload, RetrievalResult, FinalResponse, Message, Role
from src.llm_client import LLMClient


class RAGPipeline:
    """
    RAG 基础链路管线。
    流程：用户查询 → 向量检索(Retrieve) → 重排序(Rerank) → LLM 生成(Generate)
    """

    def __init__(self, llm: LLMClient, vector_store, embedder, top_k: int = 5):
        """
        Args:
            llm: LLMClient 实例（用于生成阶段）
            vector_store: VectorStore 实例（用于检索阶段）
            embedder: 嵌入函数 callable(text: str) → np.ndarray
            top_k: 检索返回数
        """
        self.llm = llm
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k

    def run(self, query: str) -> FinalResponse:
        """
        执行一次完整的 RAG 问答。
        Args:
            query: 用户查询
        Returns:
            FinalResponse（含答案、来源、Token 消耗）
        """
        # Step 1: Retrieve —— 向量检索召回候选文档
        query_vec = self.embedder(query)
        raw_results = self.vector_store.search(query_vec, top_k=self.top_k * 2)

        # Step 2: Rerank —— 对召回结果重排序，选出最相关的 top_k 条
        retrieval_results = self._rerank(query, raw_results)

        # Step 3: Generate —— 将检索结果拼入 Prompt，交由 LLM 生成答案
        context = self._build_prompt(query, retrieval_results)
        messages = [
            Message(role=Role.SYSTEM, content="你是一个基于参考资料回答问题的助手。请严格依据提供的上下文回答，不得编造信息。").to_openai(),
            Message(role=Role.USER, content=context).to_openai(),
        ]
        response = self.llm.chat(messages=messages)
        answer = response.choices[0].message.content or ""
        usage = {
            "prompt": response.usage.prompt_tokens,
            "completion": response.usage.completion_tokens,
            "total": response.usage.total_tokens,
        }

        return FinalResponse(
            answer=answer,
            sources=[{"doc_id": r.chunk.doc_id, "chunk_id": r.chunk.chunk_id, "score": r.score}
                     for r in retrieval_results],
            usage=usage,
        )

    # ============================================================
    # Rerank: 使用 LLM 对召回结果做精排（评分 + 筛选）
    # ============================================================

    def _rerank(self, query: str, results: list[tuple], top_k: int | None = None) -> list[RetrievalResult]:
        """
        对粗排结果做 LLM-based Rerank。
        原理：让 LLM 对每条召回结果与 query 的相关性打分（1-10），只保留高分结果。
        """
        if top_k is None:
            top_k = self.top_k
        if not results:
            return []

        scored = []
        for chunk, vector_score in results:
            relevance = self._score_relevance(query, chunk.content)
            # 综合得分 = 向量相似度 * 0.3 + LLM 相关性 * 0.7
            combined = vector_score * 0.3 + relevance * 0.1  # LLM score 1-10 → 归一化为 0.1-1.0
            scored.append((chunk, combined))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [RetrievalResult(chunk=chunk, score=score) for chunk, score in scored[:top_k]]

    def _score_relevance(self, query: str, chunk_content: str) -> float:
        """
        调用 LLM 对单条内容与查询的相关性打分（1-10）。
        注：MVP 阶段使用简化提示词，生产环境可用 Cross-Encoder 模型替代。
        """
        prompt = (
            f"查询: {query}\n"
            f"文档片段: {chunk_content[:500]}\n"
            "请仅返回一个 1-10 的数字，表示该文档与查询的相关程度（10=完全相关，1=完全无关）。\n"
            "只输出数字："
        )
        response = self.llm.chat(messages=[{"role": "user", "content": prompt}], temperature=0)
        try:
            score = float(response.choices[0].message.content.strip())
            return max(1, min(10, score))  # 钳制在 1-10 范围
        except ValueError:
            return 5.0  # 解析失败时使用中性分数

    # ============================================================
    # Prompt 构建
    # ============================================================

    def _build_prompt(self, query: str, results: list[RetrievalResult]) -> str:
        """将检索结果拼接为最终的 Generation Prompt"""
        context = "\n\n---\n\n".join(r.to_prompt_context() for r in results)
        return (
            f"参考资料：\n{context}\n\n"
            f"---\n"
            f"用户问题：{query}\n"
            f"请基于以上参考资料回答。如果资料中没有相关信息，请明确说明。"
        )
