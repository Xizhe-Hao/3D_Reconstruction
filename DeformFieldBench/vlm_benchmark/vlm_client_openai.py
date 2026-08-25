"""
OpenAI API 视频分析客户端。

OpenAI 当前不直接支持视频输入，需将视频抽帧后以图片形式发送给 GPT 视觉模型。

支持两种调用方式（参考 OpenAI Cookbook 与 Reasoning 文档）：
1. Responses API：推荐，支持 reasoning.summary 获取思维过程
2. Chat Completions API：兜底，仅返回最终文本
"""
import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from vlm_benchmark.parse_utils import (
        extract_json_from_response,
        extract_reasoning_before_json,
    )
except ModuleNotFoundError:
    from parse_utils import (
        extract_json_from_response,
        extract_reasoning_before_json,
    )

try:
    import cv2  # type: ignore
except ImportError as exc:
    raise RuntimeError(
        "未安装 opencv-python。请先执行：pip install opencv-python"
    ) from exc


class OpenAIVideoClient:
    """
    使用 OpenAI API 分析视频的客户端。

    流程：
      1. 使用 OpenCV 按指定 fps 从视频中抽取帧
      2. 将帧编码为 base64 JPEG
      3. 通过 chat.completions API 的 vision 能力发送给 GPT 模型
      4. 解析模型返回的 JSON 输出

    环境变量：OPENAI_API_KEY（或 api_key_env 指定的变量名）
    """

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        fps: float = 2.0,
        max_frames: int = 64,
        use_responses_api: bool = True,
        use_reasoning_summary: bool = True,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.fps = float(fps)
        self.max_frames = max(int(max_frames), 1)
        self.use_responses_api = use_responses_api
        self.use_reasoning_summary = use_reasoning_summary

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"环境变量 {api_key_env} 未设置，无法调用 OpenAI API。"
            )
        self.api_key = api_key

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "未安装 openai SDK。请先执行：pip install openai"
            ) from exc

        kwargs: Dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)

    def _extract_frames(self, video_path: str) -> List[str]:
        """
        从视频中按 fps 抽帧，返回 base64 编码的 JPEG 列表。
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total_frames <= 0:
                raise ValueError(f"视频帧数无效: {video_path}")
            indices = {
                int(round(x))
                for x in np.linspace(0, total_frames - 1, min(self.max_frames, total_frames))
            }
            base64_frames: List[str] = []
            frame_idx = 0
            while len(base64_frames) < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx in indices:
                    _, buffer = cv2.imencode(".jpg", frame)
                    b64 = base64.b64encode(buffer).decode("utf-8")
                    base64_frames.append(b64)
                frame_idx += 1

            return base64_frames
        finally:
            cap.release()

    def predict_video(
        self,
        video_path: str,
        system_prompt: str,
        user_prompt: str,
        timeout: Optional[float] = None,
        video_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"找不到视频文件: {video_path}")

        # 1) 抽帧
        base64_frames = self._extract_frames(video_path)
        if not base64_frames:
            raise ValueError(f"视频中未能抽取到任何帧: {video_path}")

        # 2) 合并 prompt
        merged_text = (
            (system_prompt.strip() + "\n\n" + user_prompt.strip()).strip()
        )

        # 3) 构造 content：先文本，再多图
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": merged_text},
        ]
        for b64 in base64_frames:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            )

        # 4) 调用 API：优先 Responses API（可获取 reasoning），失败则回退 Chat Completions
        raw_text = ""
        reasoning_text = ""
        last_exc: Optional[Exception] = None

        if self.use_responses_api:
            input_items: List[Dict[str, Any]] = [
                {"type": "input_text", "text": merged_text},
            ]
            for b64 in base64_frames:
                input_items.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{b64}",
                    }
                )
            api_input = [{"role": "user", "content": input_items}]

            for attempt in range(1, 4):
                try:
                    call_kwargs: Dict[str, Any] = {
                        "model": self.model,
                        "input": api_input,
                        "timeout": timeout or 120.0,
                    }
                    if self.use_reasoning_summary:
                        call_kwargs["reasoning"] = {
                            "effort": "low",
                            "summary": "auto",
                        }
                    resp = self._client.responses.create(**call_kwargs)
                    raw_text, reasoning_text = self._parse_responses_output(resp)
                    break
                except AttributeError:
                    last_exc = AttributeError(
                        "openai SDK 无 responses API，请升级: pip install -U openai"
                    )
                    break
                except Exception as exc:
                    err_lower = str(exc).lower()
                    if self.use_reasoning_summary and (
                        "reasoning" in err_lower or "summary" in err_lower
                    ):
                        try:
                            call_kwargs = {
                                "model": self.model,
                                "input": api_input,
                                "timeout": timeout or 120.0,
                            }
                            resp = self._client.responses.create(**call_kwargs)
                            raw_text, reasoning_text = (
                                self._parse_responses_output(resp)
                            )
                            break
                        except Exception:
                            pass
                    last_exc = exc
                    wait_s = min(2 ** (attempt - 1), 4)
                    print(
                        f"[WARN] OpenAI Responses 调用失败（第 {attempt}/3 次），"
                        f"{wait_s}s 后重试: {exc}"
                    )
                    time.sleep(wait_s)
            else:
                raw_text = ""

        if not raw_text:
            # 使用 Chat Completions 兜底
            messages = [{"role": "user", "content": content}]
            for attempt in range(1, 4):
                try:
                    resp = self._client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        timeout=timeout or 120.0,
                    )
                    choice = resp.choices[0]
                    raw_text = (
                        getattr(choice.message, "content", None) or ""
                    ).strip()
                    break
                except Exception as exc:
                    last_exc = exc
                    wait_s = min(2 ** (attempt - 1), 4)
                    print(
                        f"[WARN] OpenAI Chat 调用失败（第 {attempt}/3 次），"
                        f"{wait_s}s 后重试: {exc}"
                    )
                    time.sleep(wait_s)
            else:
                raise last_exc  # type: ignore[misc]

        if not raw_text:
            raise ValueError("OpenAI 返回了空内容")

        # 5) 解析 JSON（支持 推理+JSON、```json...``` 等格式）
        parsed = extract_json_from_response(raw_text)

        if not isinstance(parsed, dict):
            raise ValueError(f"期望 JSON 对象，但得到: {type(parsed)}")

        parsed["__raw_text__"] = raw_text
        # 若 API 未返回 reasoning（如 gpt-4o 不支持 reasoning.summary），则从回复文本中提取 JSON 前的推理内容
        if not reasoning_text:
            reasoning_text = extract_reasoning_before_json(raw_text)
        if reasoning_text:
            parsed["__reasoning__"] = reasoning_text
        return parsed

    def _parse_responses_output(self, resp: Any) -> tuple[str, str]:
        """
        从 Responses API 的 output 数组中提取 reasoning 与主文本。
        参考：https://platform.openai.com/docs/guides/reasoning#reasoning-summaries
        """
        raw_text = ""
        reasoning_parts: List[str] = []
        output = getattr(resp, "output", None) or []
        for item in output:
            item_type = getattr(item, "type", None)
            if item_type == "reasoning":
                summary = getattr(item, "summary", None)
                if isinstance(summary, list):
                    for s in summary:
                        s_type = getattr(s, "type", None)
                        if s_type == "summary_text":
                            t = getattr(s, "text", "") or ""
                            if t:
                                reasoning_parts.append(t)
            elif item_type == "message":
                content = getattr(item, "content", None) or []
                for c in content:
                    c_type = getattr(c, "type", None)
                    if c_type == "output_text":
                        raw_text = getattr(c, "text", "") or ""
                        break
        if not raw_text:
            raw_text = getattr(resp, "output_text", "") or ""
        reasoning_text = "\n\n".join(reasoning_parts) if reasoning_parts else ""
        return raw_text.strip(), reasoning_text


__all__ = ["OpenAIVideoClient"]
