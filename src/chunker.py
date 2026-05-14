# src/chunker.py
# 文档分块模块：将清洗后文本切分为可索引的片段（重叠分块策略）
# 入库：支持 FAISS 向量库（本地 CPU 友好）

import hashlib
import numpy as np
from src.schemas import Document, Chunk


class TextChunker:
    """
    文本分块器 —— 使用重叠滑动窗口策略。
    核心原则：相邻 Chunk 之间共享 overlap 长度的文本，减少语义断裂。
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 128):
        """
        Args:
            chunk_size: 每个分块的最大字符数
            overlap: 相邻分块的重叠字符数（推荐 chunk_size 的 20%-25%）
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, doc: Document) -> list[Chunk]:
        """
        将单个 Document 切分为多个 Chunk。
        分块逻辑：按字符滑动窗口，尽量在句末/段末断开。
        """
        text = doc.content
        chunks = []
        start = 0
        idx = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]

            # 尝试在句末/段末断开（向后找最近的断点）
            if end < len(text):
                break_points = ["\n\n", "\n", "。", "！", "？", ". ", "! ", "? "]
                best_break = end
                for bp in break_points:
                    pos = chunk_text.rfind(bp)
                    if pos > self.chunk_size * 0.6:  # 断开位置不能太靠前
                        best_break = start + pos + len(bp)
                        break
                end = best_break
                chunk_text = text[start:end]

            if chunk_text.strip():
                chunk_id = self._make_chunk_id(doc.doc_id, idx)
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    content=chunk_text.strip(),
                    chunk_index=idx,
                    metadata={**doc.metadata, "chunk_index": idx},
                ))
                idx += 1

            # 下一个窗口起点 = 当前终点 - overlap（不重叠会导致语义断裂）
            start = end - self.overlap if end < len(text) else len(text)

        return chunks

    @staticmethod
    def _make_chunk_id(doc_id: str, index: int) -> str:
        """生成唯一 Chunk ID"""
        raw = f"{doc_id}::{index}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]


# ============================================================
# FAISS 向量库入库（极简封装）
# ============================================================

class VectorStore:
    """
    本地向量库，基于 FAISS + numpy。
    用于 RAG 的文档索引和相似度检索。
    注：此处用 numpy 模拟向量检索，生产环境替换为 FAISS IndexFlatIP/IndexIVFFlat
    """

    def __init__(self, dim: int = 768):
        """
        Args:
            dim: 向量维度（需与 Embedding 模型输出维度一致）
        """
        self.dim = dim
        self.vectors: np.ndarray | None = None   # shape: (n, dim)
        self.chunks: list[Chunk] = []            # 与 vectors 行一一对应

    def add(self, chunks: list[Chunk], embeddings: np.ndarray):
        """
        将 Chunk 向量化后入库。
        Args:
            chunks: 分块列表
            embeddings: 对应的向量矩阵 shape (len(chunks), dim)
        """
        self.chunks.extend(chunks)
        if self.vectors is None:
            self.vectors = embeddings.astype(np.float32)
        else:
            self.vectors = np.concatenate([self.vectors, embeddings.astype(np.float32)], axis=0)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """
        余弦相似度检索（numpy 实现）。
        Args:
            query_embedding: 查询向量 shape (1, dim)
            top_k: 返回最相似的 K 个结果
        Returns:
            [(Chunk, score), ...] 按相似度降序排列
        """
        if self.vectors is None or len(self.vectors) == 0:
            return []

        # 归一化后点积 = 余弦相似度
        query_norm = query_embedding / (np.linalg.norm(query_embedding, axis=-1, keepdims=True) + 1e-8)
        vecs_norm = self.vectors / (np.linalg.norm(self.vectors, axis=-1, keepdims=True) + 1e-8)
        scores = np.dot(vecs_norm, query_norm.T).flatten()

        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(self.chunks[i], float(scores[i])) for i in top_indices if scores[i] > 0]
