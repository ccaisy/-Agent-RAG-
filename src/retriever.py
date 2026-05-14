# src/retriever.py
# RAG 检索质量增强模块
# 亮点 1: Hybrid Search（向量检索 + BM25 关键词检索 融合）
# 亮点 2: Query Transformation（查询扩展与重写，提升召回率）

import re
import math
import numpy as np
from collections import Counter
from src.schemas import Chunk
from src.chunker import VectorStore
from src.llm_client import LLMClient


# ============================================================
# BM25 关键词检索（极简 numpy 实现）
# ============================================================

class BM25Retriever:
    """
    BM25 关键词检索器。
    原理：基于词频(TF)和逆文档频率(IDF)计算查询与文档的相关性，
    弥补向量检索对精确关键词匹配不敏感的问题。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            k1: 词频饱和度参数（通常 1.2-2.0）
            b:  文档长度归一化参数（通常 0.75）
        """
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = []
        self.doc_freq: dict[str, int] = {}     # 每个词出现在几个文档中
        self.doc_lengths: list[int] = []        # 每个文档的长度
        self.avg_doc_len: float = 0.0

    def index(self, chunks: list[Chunk]):
        """构建 BM25 索引"""
        self.chunks = chunks
        self.doc_lengths = []
        self.doc_freq = {}
        tokenized_docs = []

        for chunk in chunks:
            tokens = self._tokenize(chunk.content)
            tokenized_docs.append(tokens)
            self.doc_lengths.append(len(tokens))
            for token in set(tokens):  # 每个词在每个文档中只计数一次（IDF）
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1

        self.avg_doc_len = np.mean(self.doc_lengths) if self.doc_lengths else 1.0
        self._tokenized_docs = tokenized_docs

    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]:
        """BM25 检索，返回 top_k 个最相关文档"""
        if not self.chunks:
            return []
        query_tokens = self._tokenize(query)
        scores = np.zeros(len(self.chunks))
        N = len(self.chunks)

        for token in set(query_tokens):
            df = self.doc_freq.get(token, 0)
            if df == 0:
                continue
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
            for i, doc_tokens in enumerate(self._tokenized_docs):
                tf = doc_tokens.count(token)
                doc_len = self.doc_lengths[i]
                # BM25 核心公式
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                scores[i] += idf * numerator / denominator

        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(self.chunks[i], float(scores[i])) for i in top_indices if scores[i] > 0]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """中文分词简化版：按字符 + 英文词混合切分"""
        # 提取中文字符和英文单词
        tokens = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z]+', text.lower())
        return tokens


# ============================================================
# Hybrid Search: 融合向量检索 + 关键词检索结果
# ============================================================

class HybridRetriever:
    """
    混合检索器：向量检索（语义相似） + BM25（关键词匹配）。
    使用加权 Reciprocal Rank Fusion (RRF) 融合两种排序结果。
    """

    def __init__(self, vector_store: VectorStore, bm25: BM25Retriever, alpha: float = 0.7):
        """
        Args:
            vector_store: 向量检索器
            bm25: 关键词检索器
            alpha: 向量检索权重（0-1），1 = 纯向量，0 = 纯关键词
        """
        self.vector_store = vector_store
        self.bm25 = bm25
        self.alpha = alpha

    def search(self, query: str, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """混合检索：融合两种排序"""
        # 向量检索
        vec_results = self.vector_store.search(query_embedding, top_k=top_k * 2)
        # 关键词检索
        kw_results = self.bm25.search(query, top_k=top_k * 2)

        # RRF 融合：每个结果得分 = ∑ 1/(rank + k)，按权重加权
        k = 60  # RRF 平滑常数
        scores: dict[str, tuple[Chunk, float]] = {}

        for rank, (chunk, vec_score) in enumerate(vec_results):
            rrf = self.alpha / (rank + k)
            scores[chunk.chunk_id] = (chunk, rrf)

        for rank, (chunk, kw_score) in enumerate(kw_results):
            rrf = (1 - self.alpha) / (rank + k)
            if chunk.chunk_id in scores:
                scores[chunk.chunk_id] = (chunk, scores[chunk.chunk_id][1] + rrf)
            else:
                scores[chunk.chunk_id] = (chunk, rrf)

        # 按融合分数降序排列
        merged = sorted(scores.values(), key=lambda x: x[1], reverse=True)
        return merged[:top_k]


# ============================================================
# Query Transformation: 查询扩展与重写
# ============================================================

class QueryTransformer:
    """
    查询转换器 —— 在检索前对用户 Query 做扩展/重写，提升召回。
    策略 1: LLM 生成多个相关子查询（Query Expansion）
    策略 2: 启发式关键词补充
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def expand(self, query: str, num_variants: int = 3) -> list[str]:
        """
        使用 LLM 生成查询变体（同义改写 + 多角度提问）。
        返回原始 query + N 个变体，用于多路召回。
        """
        prompt = (
            f"原始查询：{query}\n\n"
            f"请生成 {num_variants} 个语义等价但表述不同的查询变体，"
            f"用于提升搜索引擎的召回率。每个变体占一行，不要编号，不要其他内容。"
        )
        response = self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        variants = [q.strip() for q in response.choices[0].message.content.strip().split("\n") if q.strip()]
        return [query] + variants[:num_variants]

    @staticmethod
    def add_keywords(query: str, keywords: list[str]) -> str:
        """启发式：向 query 追加关键词（提高 BM25 命中率）"""
        return query + " " + " ".join(keywords)

    def multi_search(self, query: str, retriever, query_embedding: np.ndarray,
                     top_k: int = 5) -> list[tuple[Chunk, float]]:
        """
        多路召回：将原始 query 扩展为多个变体 →
        每个变体分别检索 → 合并去重返回。
        """
        queries = self.expand(query)
        seen: set[str] = set()
        all_results: list[tuple[Chunk, float]] = []

        for q in queries:
            q_emb = query_embedding  # MVP 阶段使用相同 embedding，生产环境需要 embedder(q)
            for chunk, score in retriever.search(q, q_emb, top_k=top_k):
                if chunk.chunk_id not in seen:
                    seen.add(chunk.chunk_id)
                    all_results.append((chunk, score))

        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:top_k]
