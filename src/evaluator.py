# src/evaluator.py
# 评估模块：测试用例集 + LLM-as-a-Judge 自动评分
# 目的：为面试提供量化评估数据，证明项目效果

import json
import time
from dataclasses import dataclass, field
from src.llm_client import LLMClient


@dataclass
class TestCase:
    """单条测试用例"""
    id: str
    query: str                      # 用户输入
    expected_keywords: list[str]    # 期望出现的关键词（快速检查）
    category: str = "general"       # 分类：factual / reasoning / multi_hop / code
    difficulty: str = "medium"      # easy / medium / hard


@dataclass
class JudgeResult:
    """LLM-as-a-Judge 评分结果"""
    case_id: str
    query: str
    answer: str
    accuracy: float         # 准确性 1-10
    relevance: float        # 相关性 1-10
    completeness: float     # 完整性 1-10
    overall: float          # 综合分 = mean of above
    critique: str           # 评语
    latency_ms: float = 0.0


# ============================================================
# 预置测试用例集（20 条，覆盖 4 个分类）
# ============================================================

TEST_CASES: list[TestCase] = [
    # —— factual: 事实类 ——
    TestCase("F01", "什么是大语言模型中的注意力机制？", ["注意力", "Attention", "Query", "Key", "Value"], "factual", "easy"),
    TestCase("F02", "Python 中的 GIL 是什么？有什么影响？", ["GIL", "全局解释器锁", "线程"], "factual", "easy"),
    TestCase("F03", "解释 RAG（检索增强生成）的基本原理", ["检索", "生成", "Retrieve", "Generate", "RAG"], "factual", "easy"),
    TestCase("F04", "什么是 Transformer 架构？", ["Transformer", "编码器", "解码器", "自注意力"], "factual", "easy"),
    TestCase("F05", "对比 BERT 和 GPT 的架构差异", ["BERT", "GPT", "双向", "自回归", "编码器"], "factual", "medium"),

    # —— reasoning: 推理类 ——
    TestCase("R01", "如果温度升高 2°C，海平面可能上升多少米？请给出推理过程。", ["海平面", "上升", "米", "推理"], "reasoning", "medium"),
    TestCase("R02", "为什么模型在长文本输入下推理速度会变慢？请从计算复杂度角度分析。", ["复杂度", "平方", "O(n²)", "注意力"], "reasoning", "hard"),
    TestCase("R03", "分析 Prompt Engineering 为什么能改善模型输出质量。", ["Prompt", "提示", "few-shot", "引导"], "reasoning", "medium"),
    TestCase("R04", "如果有人均算力翻倍，对 AI 行业可能产生哪些影响？", ["算力", "AI", "模型", "训练"], "reasoning", "medium"),
    TestCase("R05", "解释为什么向量检索比关键词检索更适合语义搜索。", ["向量", "语义", "关键词", "嵌入", "embedding"], "reasoning", "medium"),

    # —— multi_hop: 多跳推理类 ——
    TestCase("M01", "Transformer 论文的作者中，谁后来创办了 Character.AI？", ["Noam Shazeer", "Character.AI", "Transformer"], "multi_hop", "hard"),
    TestCase("M02", "PyTorch 和 TensorFlow 分别由哪家公司主导开发？它们最初的设计目标有什么不同？", ["PyTorch", "Meta", "TensorFlow", "Google", "动态图"], "multi_hop", "medium"),
    TestCase("M03", "AlphaGo 使用了哪些技术？其中哪些技术后来被应用到了 NLP 领域？", ["AlphaGo", "蒙特卡洛", "强化学习", "NLP"], "multi_hop", "hard"),
    TestCase("M04", "从 Word2Vec 到 GPT-4，词表示方法经历了哪些关键变革？", ["Word2Vec", "ELMo", "BERT", "GPT", "上下文"], "multi_hop", "hard"),
    TestCase("M05", "CLIP 模型如何连接文本和图像？这种跨模态思路对后续多模态模型有什么影响？", ["CLIP", "对比学习", "跨模态", "多模态"], "multi_hop", "hard"),

    # —— code: 代码/工具类 ——
    TestCase("C01", "写一个 Python 函数计算斐波那契数列的第 n 项。", ["def", "fibonacci", "return"], "code", "easy"),
    TestCase("C02", "用 Python 实现一个简单的 LRU 缓存。", ["LRU", "cache", "get", "put"], "code", "medium"),
    TestCase("C03", "写一个 Python 脚本读取 CSV 文件并计算每列的均值和标准差。", ["csv", "mean", "std", "pandas", "read"], "code", "easy"),
    TestCase("C04", "用 Python 实现文本分块算法，支持重叠窗口。", ["chunk", "overlap", "滑动", "窗口"], "code", "medium"),
    TestCase("C05", "编写一个异步 Python 函数并发调用 3 个 API 并合并结果。", ["async", "await", "asyncio", "gather", "并发"], "code", "medium"),
]


# ============================================================
# LLM-as-a-Judge 评分器
# ============================================================

class LLMJudge:
    """
    使用 LLM 作为裁判，对系统输出进行多维度评分。
    原理：将系统回答 + 期望标准发给另一个 LLM，让其打分并给出评语。
    """

    JUDGE_PROMPT = """你是一个严格的 AI 输出质量评审专家。请对以下问答进行评分。

## 用户问题
{query}

## 参考标准（期望包含的关键信息）
{expected_keywords}

## 系统回答
{answer}

## 评分要求
请从以下三个维度打分（1-10 分，10 分为最佳）：
- accuracy（准确性）：回答是否正确、事实无误
- relevance（相关性）：回答是否切题、不偏题
- completeness（完整性）：回答是否涵盖了问题的全部关键信息

请以 JSON 格式返回评分结果：
```json
{{
    "accuracy": <1-10>,
    "relevance": <1-10>,
    "completeness": <1-10>,
    "critique": "<中文评语，指出优点和不足，100字以内>"
}}
```"""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def evaluate(self, case: TestCase, answer: str, latency_ms: float = 0.0) -> JudgeResult:
        """
        对单条测试结果进行 LLM-as-a-Judge 评分。
        Args:
            case: 测试用例
            answer: 系统生成的回答
            latency_ms: 系统生成回答的耗时
        Returns:
            JudgeResult（含多维度评分 + 评语）
        """
        prompt = self.JUDGE_PROMPT.format(
            query=case.query,
            expected_keywords=", ".join(case.expected_keywords),
            answer=answer[:2000],  # 截断过长回答
        )
        response = self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = response.choices[0].message.content

        # 解析 JSON（容错处理）
        try:
            # 提取 JSON 块（可能在 ```json``` 中）
            import re
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            scores = json.loads(json_match.group(0)) if json_match else {}
        except json.JSONDecodeError:
            scores = {}

        accuracy = float(scores.get("accuracy", 5))
        relevance = float(scores.get("relevance", 5))
        completeness = float(scores.get("completeness", 5))

        return JudgeResult(
            case_id=case.id,
            query=case.query,
            answer=answer[:500],
            accuracy=min(10, max(1, accuracy)),
            relevance=min(10, max(1, relevance)),
            completeness=min(10, max(1, completeness)),
            overall=round((accuracy + relevance + completeness) / 3, 1),
            critique=str(scores.get("critique", "无法解析评语")),
            latency_ms=latency_ms,
        )


# ============================================================
# 批量评测运行器
# ============================================================

class EvalRunner:
    """批量运行测试用例并收集评分"""

    def __init__(self, judge: LLMJudge):
        self.judge = judge
        self.results: list[JudgeResult] = []

    def run(self, cases: list[TestCase], answer_fn) -> list[JudgeResult]:
        """
        逐条运行测试用例。
        Args:
            cases: 测试用例列表
            answer_fn: 回答函数 callable(query: str) -> str
        Returns:
            评分结果列表
        """
        self.results = []
        for case in cases:
            start = time.perf_counter()
            answer = answer_fn(case.query)
            latency = (time.perf_counter() - start) * 1000
            result = self.judge.evaluate(case, answer, latency_ms=latency)
            self.results.append(result)
            print(f"  [{case.id}] {case.category}/{case.difficulty} → 综合分: {result.overall}")
        return self.results

    def report(self) -> dict:
        """生成评估报告汇总"""
        if not self.results:
            return {"message": "暂无评估结果"}

        avg = sum(r.overall for r in self.results) / len(self.results)
        by_category: dict[str, list[float]] = {}
        for r in self.results:
            case = next((c for c in TEST_CASES if c.id == r.case_id), None)
            cat = case.category if case else "unknown"
            by_category.setdefault(cat, []).append(r.overall)

        return {
            "total_cases": len(self.results),
            "avg_overall_score": round(avg, 1),
            "by_category": {k: round(sum(v) / len(v), 1) for k, v in by_category.items()},
            "by_difficulty": self._by_difficulty(),
            "avg_latency_ms": round(sum(r.latency_ms for r in self.results) / len(self.results), 1),
        }

    def _by_difficulty(self) -> dict[str, float]:
        result: dict[str, list[float]] = {}
        for r in self.results:
            case = next((c for c in TEST_CASES if c.id == r.case_id), None)
            diff = case.difficulty if case else "unknown"
            result.setdefault(diff, []).append(r.overall)
        return {k: round(sum(v) / len(v), 1) for k, v in result.items()}
