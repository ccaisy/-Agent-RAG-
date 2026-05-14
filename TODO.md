# TODO: 大模型算法求职项目开发路径图 (MVP 1.0)

## 阶段一：项目底座与架构规约 (设计态)
> **目标**：建立符合工业界标准的模块化结构，拒绝单文件大杂烩代码。
- [ ] **项目骨架搭建**：按照 `src/`, `configs/`, `data/`, `tests/` 建立标准目录。
- [ ] **技术栈选型与环境隔离**：
    - 使用 `poetry` 或 `conda` 创建环境。
    - 接入 LLM API (如 DeepSeek, OpenAI) 或本地部署 7B/8B 轻量模型。
- [ ] **定义 Core Schema**：明确输入输出协议 (JSON Schema)，为 Agent 的 Tool Calling 或 RAG 的 Payload 建立标准。
- [ ] **亮点预埋**：调研并确定项目的“算法差异化”，如：引入多跳检索 (Multi-hop)、自反思机制 (Self-reflection) 或特定的 Prompt Engineering 策略。

## 阶段二：核心链路打通 (实现态)
> **目标**：实现“输入测试数据，无报错输出结果”的 MVP 闭环。
- [ ] **数据处理模块 (Data Pipeline)**：
    - 编写清理脚本：针对 PDF/Markdown/Web 数据进行清洗。
    - 针对 RAG：实现 Chunking 策略（如重叠分块）并入库 FAISS/Chroma。
- [ ] **核心逻辑模块实现**：
    - **Agent 项目**：实现 ReAct 或 Plan-and-Execute 循环。
    - **RAG 项目**：实现 Retrieve -> Rerank -> Generation 基础链路。
- [ ] **LLM 接口封装**：实现具备重试机制 (Retry) 和异步调用 (Async) 的 API 客户端，处理 macOS 上的网络并发。

## 阶段三：算法调优与亮点注入 (增强态)
> **目标**：将“玩具项目”升级为“求职项目”，解决 1-2 个具体的技术痛点。
- [ ] **检索质量增强 (针对 RAG)**：
    - 引入 **Hybrid Search** (向量检索 + 关键词检索)。
    - 实现 **Query Transformation** (Query 扩展或重写)。
- [ ] **推理逻辑增强 (针对 Agent)**：
    - 注入 **CoT (思维链)** 或 **Self-Correction (自我修正)** 逻辑。
    - 设计至少 2 个高价值 Tool (如实时 Web 搜索、代码执行、私有数据分析)。
- [ ] **性能分析**：记录 Token 消耗、耗时 (Latency) 和成功率。

## 阶段四：评估、文档与简历包装 (交付态)
> **目标**：为面试做准备，确保证据链闭环。
- [ ] **构建评估集 (Evaluation)**：
    - 准备 20-30 条高质量测试 case。
    - 使用 LLM-as-a-judge 方式对输出进行评分。
- [ ] **README.md 撰写 (简历镜像)**：
    - 画出架构图 (Architecture Diagram)。
    - 明确列出：项目背景、核心挑战、解决方案 (亮点)、运行结果示例。
- [ ] **终端演示优化**：编写一个简单的 `main.py` 入口，支持交互式对话演示，确保无报错退出。