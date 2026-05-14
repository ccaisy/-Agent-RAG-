# src/metrics.py
# 性能分析模块：追踪 Token 消耗、耗时 (Latency) 和成功率
# 为面试展示提供量化数据支撑

import time
import json
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class CallRecord:
    """单次 LLM 调用的性能记录"""
    call_type: str        # "chat" | "agent" | "rag" | "embed"
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float     # 耗时（毫秒）
    success: bool
    error: str | None = None


class MetricsTracker:
    """
    性能追踪器 —— 记录每次 LLM 调用的 Token / 延迟 / 成功率。
    支持按模块（Agent / RAG / Embedding）分组统计。
    """

    def __init__(self):
        self.records: list[CallRecord] = []

    def record(self, record: CallRecord):
        """记录一次调用"""
        self.records.append(record)

    # ============================================================
    # 统计查询
    # ============================================================

    def summary(self) -> dict:
        """汇总统计：总体 Token 消耗、平均延迟、成功率"""
        if not self.records:
            return {"total_calls": 0, "message": "暂无调用记录"}

        total_prompt = sum(r.prompt_tokens for r in self.records)
        total_completion = sum(r.completion_tokens for r in self.records)
        success_count = sum(1 for r in self.records if r.success)
        latencies = [r.latency_ms for r in self.records]

        return {
            "total_calls": len(self.records),
            "success_count": success_count,
            "success_rate": f"{success_count / len(self.records) * 100:.1f}%",
            "total_tokens": {
                "prompt": total_prompt,
                "completion": total_completion,
                "total": total_prompt + total_completion,
            },
            "avg_latency_ms": f"{sum(latencies) / len(latencies):.1f}",
            "p50_latency_ms": f"{sorted(latencies)[len(latencies) // 2]:.1f}",
            "p95_latency_ms": f"{sorted(latencies)[int(len(latencies) * 0.95)]:.1f}" if len(latencies) >= 20 else "N/A",
        }

    def by_module(self) -> dict[str, dict]:
        """按模块分组统计（Agent / RAG / Embedding）"""
        groups: dict[str, list[CallRecord]] = defaultdict(list)
        for r in self.records:
            groups[r.call_type].append(r)

        result = {}
        for module, recs in groups.items():
            result[module] = {
                "calls": len(recs),
                "avg_latency_ms": sum(r.latency_ms for r in recs) / len(recs),
                "total_tokens": sum(r.prompt_tokens + r.completion_tokens for r in recs),
            }
        return result

    def export_json(self, filepath: str):
        """导出全部记录为 JSON（用于可视化分析）"""
        data = [{
            "call_type": r.call_type,
            "model": r.model,
            "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens,
            "latency_ms": r.latency_ms,
            "success": r.success,
            "error": r.error,
        } for r in self.records]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# ============================================================
# 装饰器：自动追踪函数调用性能
# ============================================================

def track_latency(tracker: MetricsTracker, call_type: str, model: str = "deepseek-chat"):
    """
    装饰器：自动记录被装饰函数的 Token 和延迟。
    被装饰函数需返回 OpenAI API response 对象或 FinalResponse。
    用法: @track_latency(tracker, "rag")
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                latency = (time.perf_counter() - start) * 1000
                usage = getattr(result, "usage", None)
                if usage:
                    tracker.record(CallRecord(
                        call_type=call_type, model=model,
                        prompt_tokens=getattr(usage, "prompt_tokens", 0),
                        completion_tokens=getattr(usage, "completion_tokens", 0),
                        latency_ms=latency, success=True,
                    ))
                else:
                    # FinalResponse 格式
                    u = getattr(result, "usage", {})
                    tracker.record(CallRecord(
                        call_type=call_type, model=model,
                        prompt_tokens=u.get("prompt", 0),
                        completion_tokens=u.get("completion", 0),
                        latency_ms=latency, success=True,
                    ))
                return result
            except Exception as e:
                latency = (time.perf_counter() - start) * 1000
                tracker.record(CallRecord(
                    call_type=call_type, model=model,
                    prompt_tokens=0, completion_tokens=0,
                    latency_ms=latency, success=False, error=str(e),
                ))
                raise
        return wrapper
    return decorator


# ============================================================
# 全局单例
# ============================================================

_global_tracker = MetricsTracker()

def get_tracker() -> MetricsTracker:
    """获取全局 MetricsTracker 单例"""
    return _global_tracker
