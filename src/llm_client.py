# src/llm_client.py
# LLM API 客户端封装，支持 OpenAI / DeepSeek（兼容接口）
# 特性：重试机制（指数退避）+ 异步调用（macOS 网络并发友好）

import os
import time
import asyncio
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()  # 加载 .env 中的环境变量


class LLMClient:
    """
    统一的 LLM 调用客户端（同步 + 异步）。
    特性：自动重试（指数退避）、provider 自适应。
    """

    def __init__(self, provider: str = "deepseek"):
        """
        Args:
            provider: "openai" 或 "deepseek"
        """
        self.provider = provider.lower()
        if self.provider == "openai":
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            api_key = os.getenv("OPENAI_API_KEY", "")
        else:  # 默认 deepseek
            base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
            api_key = os.getenv("DEEPSEEK_API_KEY", "")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    # ============================================================
    # 同步调用（含重试）
    # ============================================================

    def chat(self, messages: list[dict], model: str = "deepseek-chat",
             max_retries: int = 3, retry_delay: float = 1.0, **kwargs):
        """
        发送聊天请求（同步，含指数退避重试）。
        Args:
            messages: 标准 OpenAI 消息格式
            model: 模型名称
            max_retries: 最大重试次数（默认 3 次）
            retry_delay: 初始重试等待秒数（每次翻倍）
            **kwargs: 透传参数（temperature, max_tokens 等）
        Returns:
            API 响应对象
        """
        model = self._auto_model(model)
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return self.client.chat.completions.create(
                    model=model, messages=messages, **kwargs
                )
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    wait = retry_delay * (2 ** attempt)  # 指数退避：1s, 2s, 4s...
                    time.sleep(wait)
        raise last_error  # 重试耗尽，抛出最后异常

    # ============================================================
    # 异步调用（macOS 并发友好）
    # ============================================================

    async def chat_async(self, messages: list[dict], model: str = "deepseek-chat",
                         max_retries: int = 3, retry_delay: float = 1.0, **kwargs):
        """
        发送聊天请求（异步，含指数退避重试）。
        用法: response = await client.chat_async(messages=[...])
        """
        model = self._auto_model(model)
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return await self.async_client.chat.completions.create(
                    model=model, messages=messages, **kwargs
                )
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    wait = retry_delay * (2 ** attempt)
                    await asyncio.sleep(wait)
        raise last_error

    # ============================================================
    # 并发批量调用 —— 同时处理多个请求
    # ============================================================

    async def batch_chat(self, requests: list[dict]) -> list:
        """
        并发执行多个 chat 请求（适合需要同时查询多个 prompt 的场景）。
        Args:
            requests: [{"messages": [...], "model": "...", ...}, ...]
        Returns:
            按原顺序的响应列表
        """
        tasks = [self.chat_async(**req) for req in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)

    # ============================================================
    # 辅助方法
    # ============================================================

    def _auto_model(self, model: str) -> str:
        """provider 与 model 自动适配"""
        if self.provider == "openai" and model == "deepseek-chat":
            return "gpt-4o-mini"
        return model

    @staticmethod
    def extract_reply(response) -> str:
        """从 API 响应中提取第一条回复文本"""
        return response.choices[0].message.content


# ============================================================
# 便捷函数
# ============================================================

def quick_chat(prompt: str, provider: str = "deepseek") -> str:
    """同步单轮对话快捷入口"""
    client = LLMClient(provider=provider)
    response = client.chat(messages=[{"role": "user", "content": prompt}])
    return LLMClient.extract_reply(response)


async def quick_chat_async(prompt: str, provider: str = "deepseek") -> str:
    """异步单轮对话快捷入口"""
    client = LLMClient(provider=provider)
    response = await client.chat_async(messages=[{"role": "user", "content": prompt}])
    return LLMClient.extract_reply(response)
