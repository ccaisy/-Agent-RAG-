# main.py
# 项目入口：交互式对话演示 + 评测模式
# 用法: python main.py          # 交互式对话
#       python main.py --eval   # 运行评测集

import sys
import argparse
from src.llm_client import LLMClient, quick_chat
from src.agent import ReActAgent
from src.tools import ToolExecutor, WEB_SEARCH_TOOL, CODE_EXEC_TOOL
from src.evaluator import TEST_CASES, LLMJudge, EvalRunner
from src.metrics import get_tracker, track_latency


def interactive_mode():
    """交互式对话模式 —— 支持 Agent 和简单对话两种模式"""
    print("=" * 50)
    print("  My LLM Project - 交互式对话演示")
    print("  输入 'quit' 退出 | 输入 'agent' 切换 Agent 模式")
    print("  输入 'chat' 切换普通对话模式 | 输入 'help' 查看帮助")
    print("=" * 50)

    llm = LLMClient(provider="deepseek")
    agent = ReActAgent(
        llm=llm,
        tools=[WEB_SEARCH_TOOL.to_openai_function(), CODE_EXEC_TOOL.to_openai_function()],
        tool_executor=ToolExecutor(),
    )
    mode = "chat"  # chat | agent

    while True:
        try:
            user_input = input(f"\n[{mode}] 用户: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("再见！")
            break
        elif user_input.lower() == "agent":
            mode = "agent"
            print("[系统] 已切换到 Agent 模式（CoT + ReAct + Self-Correction）")
            continue
        elif user_input.lower() == "chat":
            mode = "chat"
            print("[系统] 已切换到普通对话模式")
            continue
        elif user_input.lower() == "help":
            print("命令说明:")
            print("  chat  - 切换到普通对话模式")
            print("  agent - 切换到 Agent 模式（可调用工具）")
            print("  quit  - 退出程序")
            continue

        # 处理输入
        if mode == "agent":
            result = agent.run(user_input)
            print(f"\n[Agent] {result.answer}")
            if result.tool_trace:
                print(f"[工具链路] 共调用 {len(result.tool_trace)} 次工具:")
                for t in result.tool_trace:
                    status = "✓" if t.success else "✗"
                    print(f"  {status} {t.tool_name}")
        else:
            reply = quick_chat(user_input)
            print(f"\n[模型] {reply}")

    # 打印本次会话统计
    tracker = get_tracker()
    summary = tracker.summary()
    if summary["total_calls"] > 0:
        print(f"\n[会话统计] 调用次数: {summary['total_calls']}, "
              f"成功率: {summary['success_rate']}, "
              f"平均延迟: {summary['avg_latency_ms']}ms")


def eval_mode():
    """批量评测模式 —— 对所有测试用例运行 LLM-as-a-Judge 评分"""
    print("=" * 50)
    print("  My LLM Project - 批量评测模式")
    print(f"  测试用例: {len(TEST_CASES)} 条")
    print("=" * 50)

    llm = LLMClient(provider="deepseek")
    judge = LLMJudge(llm)
    runner = EvalRunner(judge)

    # 使用 quick_chat 作为基准回答函数
    def answer_fn(query: str) -> str:
        return quick_chat(query)

    print("\n开始评测...\n")
    results = runner.run(TEST_CASES, answer_fn)

    # 输出报告
    report = runner.report()
    print(f"\n{'=' * 50}")
    print("  评测报告汇总")
    print(f"{'=' * 50}")
    print(f"  总用例数:   {report['total_cases']}")
    print(f"  平均综合分: {report['avg_overall_score']} / 10")
    print(f"  平均延迟:   {report['avg_latency_ms']}ms")
    print(f"\n  按分类:")
    for cat, score in report.get("by_category", {}).items():
        print(f"    {cat}: {score}")
    print(f"\n  按难度:")
    for diff, score in report.get("by_difficulty", {}).items():
        print(f"    {diff}: {score}")

    # 导出详细结果
    import json
    output = [{
        "case_id": r.case_id,
        "query": r.query,
        "accuracy": r.accuracy,
        "relevance": r.relevance,
        "completeness": r.completeness,
        "overall": r.overall,
        "critique": r.critique,
    } for r in runner.results]
    with open("data/eval_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已导出至 data/eval_results.json")


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="My LLM Project - AI Agent + RAG 系统")
    parser.add_argument("--eval", action="store_true", help="运行批量评测模式")
    args = parser.parse_args()

    try:
        if args.eval:
            eval_mode()
        else:
            interactive_mode()
    except KeyboardInterrupt:
        print("\n程序已退出。")
    except Exception as e:
        print(f"\n[错误] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
