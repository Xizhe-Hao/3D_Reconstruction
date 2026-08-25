"""
共享的解析工具：从模型回复中提取 JSON，支持前后有多余文字（如推理说明）。
"""
import json
import re
from typing import Any, Dict


def extract_json_from_response(raw_text: str) -> Dict[str, Any]:
    """
    从模型返回文本中提取 JSON 对象。
    兼容：纯 JSON、```json ... ``` 包裹、推理文字 + JSON 等格式。
    """
    text = raw_text.strip()

    # 1) 去掉 markdown 代码块
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

    # 2) 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 3) 尝试从文本中提取 {...} 块
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i, c in enumerate(text[start:], start=start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
                    break

    raise ValueError(
        f"模型返回不是合法 JSON，请检查 prompt。\n原始输出:\n{raw_text}"
    )


def extract_reasoning_before_json(raw_text: str) -> str:
    """
    从模型回复中提取 JSON 之前的文本，作为推理内容。
    """
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()
    start = text.find("{")
    if start <= 0:
        return ""
    prefix = text[:start].strip()
    if re.match(r"^(?:json|```)\s*$", prefix, re.I):
        return ""
    return prefix if prefix else ""
