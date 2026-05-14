# src/llm_client.py
# 极简 LLM API 客户端封装，支持 OpenAI / DeepSeek（兼容接口）

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # 加载 .env 中的环境变量


class LLMClient:
    """
    统一的 LLM 调用客户端。
    当前支持 provider: "openai" 或 "deepseek"（均兼容 OpenAI SDK）。
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

    def chat(self, messages: list[dict], model: str = "deepseek-chat", **kwargs):
        """
        发送聊天请求。
        Args:
            messages: 标准 OpenAI 消息格式 [{"role": "user/system/assistant", "content": "..."}]
            model: 模型名称，deepseek 默认 "deepseek-chat"，openai 默认 "gpt-4o-mini"
            **kwargs: 透传给 API 的额外参数（temperature, max_tokens 等）
        Returns:
            API 响应的完整对象
        """
        if self.provider == "openai" and model == "deepseek-chat":
            model = "gpt-4o-mini"  # 自动切换默认模型
        return self.client.chat.completions.create(model=model, messages=messages, **kwargs)

    def extract_reply(self, response) -> str:
        """从 API 响应中提取第一条回复文本。"""
        return response.choices[0].message.content


# 便捷函数：快速调用一次对话
def quick_chat(prompt: str, provider: str = "deepseek") -> str:
    """单轮对话快捷入口。"""
    client = LLMClient(provider=provider)
    response = client.chat(messages=[{"role": "user", "content": prompt}])
    return client.extract_reply(response)
