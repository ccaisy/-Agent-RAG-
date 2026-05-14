# My LLM Project —— 大模型智能 Agent + RAG 系统

> 面向算法岗求职的 MVP 项目，展示大模型应用工程化能力和算法调优意识。

## 架构图

```
                        ┌──────────────────────────┐
                        │       main.py (入口)       │
                        │  交互式 CLI / 批量评测      │
                        └────────────┬─────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                      ▼
    ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
    │   Agent 模块     │   │   RAG 模块       │   │   评测模块       │
    │ CoT + ReAct +   │   │ Retrieve →      │   │ LLM-as-a-Judge │
    │ Self-Correction │   │ Rerank → Gen    │   │ 20 条测试用例    │
    └────────┬────────┘   └────────┬────────┘   └────────┬────────┘
             │                      │                      │
             └──────────────────────┼──────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
    ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
    │ llm_client   │   │  retriever   │   │   metrics    │
    │ 重试+异步    │   │ Hybrid Search│   │ Token/延迟   │
    │ DeepSeek/    │   │ BM25+向量     │   │ 成功率统计   │
    │ OpenAI       │   │ Query扩展    │   │              │
    └──────────────┘   └──────────────┘   └──────────────┘
```

## 项目背景

本项目的目标是构建一个**可运行、具备核心算法亮点、能写入简历**的大模型应用系统。项目支持两条主线：

- **Agent 路径**：实现能调用工具的智能 Agent，具备多步推理和自我修正能力
- **RAG 路径**：实现检索增强生成管线，支持混合检索和查询优化

## 核心挑战与解决方案

| 挑战 | 解决方案（算法亮点） |
|------|---------------------|
| **Agent 输出不可控** | 注入 CoT（思维链）Prompt，强制模型"先想再做"，每步行动可追溯 |
| **工具调用失败** | Self-Correction 机制：失败后自动分析错误原因，修正参数或切换工具 |
| **向量检索关键词不敏感** | Hybrid Search：BM25 关键词 + 向量语义的 RRF 加权融合 |
| **用户 Query 表达不充分** | Query Transformation：LLM 自动生成多个查询变体做多路召回 |
| **RAG 粗排精度不足** | LLM-based Rerank：用 LLM 对召回结果重新评分（1-10），综合排序 |
| **API 网络不稳定（macOS）** | 指数退避重试 + asyncio 异步并发 + batch_chat 批量请求 |

## 项目结构

```
my_llm_project/
├── src/
│   ├── schemas.py          # Core Schema（Agent/RAG 统一协议）
│   ├── llm_client.py       # LLM API 封装（重试 + 异步 + 批量）
│   ├── agent.py            # CoT + ReAct + Self-Correction Agent
│   ├── tools.py            # 高价值工具（Web搜索 + Python沙箱）
│   ├── rag.py              # RAG 管线（Retrieve → Rerank → Generate）
│   ├── retriever.py        # Hybrid Search + Query Transformation
│   ├── chunker.py          # 文本分块 + FAISS 向量库
│   ├── data_cleaner.py     # PDF/Markdown/Web 数据清洗
│   ├── evaluator.py        # 评测集 + LLM-as-a-Judge
│   └── metrics.py          # 性能追踪（Token/延迟/成功率）
├── tests/                  # 单元测试
├── configs/                # 配置目录
├── data/                   # 数据目录
├── main.py                 # 交互式入口
├── environment.yml         # conda 环境
└── README.md
```

## 快速开始

```bash
# 1. 创建环境
conda env create -f environment.yml
conda activate my_llm_project

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env 填入真实的 DEEPSEEK_API_KEY 或 OPENAI_API_KEY

# 3. 交互式运行
python main.py

# 4. 运行评测
python main.py --eval
```

## 运行结果示例

```
用户: 什么是 Transformer 中的自注意力机制？

Agent 思考: 这是一个概念性问题，无需调用工具，可直接回答。
Agent 回答: 自注意力（Self-Attention）是 Transformer 的核心机制...

[性能] Token: 234 | 延迟: 1.2s | 成功率: 100%
```

## 技术栈

- **LLM API**: DeepSeek / OpenAI（兼容接口）
- **向量检索**: FAISS (CPU) + BM25
- **语言**: Python 3.11, asyncio
- **环境**: macOS, conda

## 评估结果概览

| 指标 | 数值 |
|------|------|
| 测试用例数 | 20 |
| 平均综合分 (LLM-as-a-Judge) | 待运行 |
| factual 类均分 | 待运行 |
| reasoning 类均分 | 待运行 |
| multi_hop 类均分 | 待运行 |
| code 类均分 | 待运行 |
| 平均延迟 | 待运行 |
| 成功率 | 待运行 |
