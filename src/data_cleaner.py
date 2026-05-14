# src/data_cleaner.py
# 数据清洗模块：针对 PDF / Markdown / Web 文本进行标准化清洗
# 输入：原始文本字符串
# 输出：去噪、格式化后的纯文本

import re
import html


class TextCleaner:
    """统一文本清洗器，处理常见的噪音问题"""

    @staticmethod
    def clean_pdf(text: str) -> str:
        """
        清洗 PDF 提取的文本。
        典型噪音：多余换行、页眉页脚数字、乱码字符
        """
        text = text.strip()
        # 去除 PDF 常见页码（单独一行的纯数字）
        text = re.sub(r'\n\s*\d{1,4}\s*\n', '\n', text)
        # 合并断行（单行非句末的换行视为 PDF 排版断行）
        text = re.sub(r'(?<![。！？\.\!\?\n])\n(?=[^\n])', '', text)
        # 压缩连续空行
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 去除不可见控制字符（保留常见空白）
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
        return text.strip()

    @staticmethod
    def clean_markdown(text: str) -> str:
        """
        清洗 Markdown 文本。
        处理：去除 code block 标记、保留代码内容、去除图片链接
        """
        # 去除图片语法 ![...](...)
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
        # 去除链接语法保留文字 [text](url) → text
        text = re.sub(r'\[([^\]]*)\]\(.*?\)', r'\1', text)
        # 去除 Markdown 格式标记（粗体、斜体、行内代码）
        text = re.sub(r'[*_~`]{1,3}', '', text)
        # 去除 HTML 标签
        text = re.sub(r'<[^>]+>', '', text)
        # 压缩空行
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    @staticmethod
    def clean_web(text: str) -> str:
        """
        清洗网页抓取文本。
        处理：HTML 实体解码、多余空白、JavaScript/CSS 残留
        """
        # HTML 实体解码（&amp; → &, &lt; → < 等）
        text = html.unescape(text)
        # 去除 HTML 注释
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
        # 去除 <script> 和 <style> 块
        text = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', text, flags=re.DOTALL | re.IGNORECASE)
        # 去除所有 HTML 标签
        text = re.sub(r'<[^>]+>', '', text)
        # 去除 URL
        text = re.sub(r'https?://\S+', '', text)
        # 压缩空白和空行
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    @classmethod
    def auto_clean(cls, text: str, source: str = "auto") -> str:
        """
        自动识别来源并清洗。
        Args:
            text: 原始文本
            source: "pdf" | "markdown" | "web" | "auto"
        """
        if source == "pdf":
            return cls.clean_pdf(text)
        elif source == "markdown":
            return cls.clean_markdown(text)
        elif source == "web":
            return cls.clean_web(text)
        else:  # auto: 按顺序尝试所有清洗器
            text = cls.clean_web(text)
            text = cls.clean_markdown(text)
            return text.strip()


# 便捷函数
def clean_text(text: str, source: str = "auto") -> str:
    """输入原始文本 → 返回清洗后文本（单行调用）"""
    return TextCleaner.auto_clean(text, source=source)
