import json
import os
import tempfile
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


class DashscopeMultiModalClient:
    """
    使用 DashScope 官方 SDK 的 MultiModalConversation 调用多模态模型（含视频）。

    参考文档用法：
      - dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"
      - messages: [{'role':'user','content':[{'video': 'file:///abs/path.mp4', 'fps':2},{'text':'...'}]}]
      - MultiModalConversation.call(api_key=..., model=..., messages=...)
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key_env: str,
        fps: int = 2,
        max_frames: int = 64,
        prefer_image_frames: bool = False,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.fps = int(fps)
        self.max_frames = max(int(max_frames), 1)
        self.prefer_image_frames = bool(prefer_image_frames)

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"环境变量 {api_key_env} 未设置，无法调用 DashScope。")
        self.api_key = api_key

        try:
            import dashscope  # type: ignore
            from dashscope import MultiModalConversation  # type: ignore
            import cv2  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "未安装 dashscope SDK 或 opencv-python-headless。请先执行：pip install dashscope opencv-python-headless"
            ) from exc

        # 设置 DashScope API base url（地域）
        dashscope.base_http_api_url = self.base_url
        self._mm = MultiModalConversation
        self._cv2 = cv2

    def _extract_temp_frames(self, video_path: str, *, max_frames: Optional[int] = None) -> List[Path]:
        target_frames = self.max_frames if max_frames is None else max(int(max_frames), 1)
        cap = self._cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        total_frames = int(cap.get(self._cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            cap.release()
            raise ValueError(f"视频帧数无效: {video_path}")
        indices = {
            int(round(x))
            for x in np.linspace(0, total_frames - 1, min(target_frames, total_frames))
        }
        tmp_dir = Path(tempfile.mkdtemp(prefix="dashscope_frames_"))
        out_paths: List[Path] = []
        frame_idx = 0
        try:
            while len(out_paths) < target_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx in indices:
                    out_path = tmp_dir / f"frame_{len(out_paths):03d}.jpg"
                    self._cv2.imwrite(str(out_path), frame)
                    out_paths.append(out_path)
                frame_idx += 1
            return out_paths
        finally:
            cap.release()

    def _call_with_video(self, *, video_uri: str, merged_text: str):
        messages = [
            {
                "role": "user",
                "content": [
                    {"video": video_uri, "fps": self.fps},
                    {"text": merged_text},
                ],
            }
        ]
        return self._mm.call(
            api_key=self.api_key,
            model=self.model,
            messages=messages,
        )

    def _call_with_images(self, *, frame_paths: List[Path], merged_text: str):
        content: List[Dict[str, object]] = [{"text": merged_text}]
        for frame_path in frame_paths:
            content.append({"image": frame_path.resolve().as_uri()})
        messages = [{"role": "user", "content": content}]
        return self._mm.call(
            api_key=self.api_key,
            model=self.model,
            messages=messages,
        )

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

        # DashScope 官方示例使用 file:// 形式。
        # 重要：路径中可能含 '+'（如科学计数法 e+04），部分 URI 解析会把 '+' 当空格。
        # 使用 Path.as_uri() 做 percent-encoding，确保 '+' -> %2B，避免 “file not exists” 误判。
        video_uri = Path(os.path.abspath(video_path)).resolve().as_uri()

        # 这里仍沿用我们 benchmark 的 prompt：system + user
        # MultiModalConversation 只需要 role=user 也能跑，但我们把 system prompt 合并进 text，避免丢失约束。
        merged_text = (system_prompt.strip() + "\n\n" + user_prompt.strip()).strip()

        # timeout 参数：dashscope SDK 是否支持传入 timeout 取决于版本；
        # 这里不强行传，保持兼容性。
        # OSS 上传阶段偶发 TLS EOF / 网络抖动，做有限次重试
        if self.prefer_image_frames:
            frame_paths = self._extract_temp_frames(video_path)
            print(
                f"[DashScope] 使用图片帧输入：video_id={video_id or Path(video_path).stem}, "
                f"frames={len(frame_paths)}"
            )
            resp = self._call_with_images(frame_paths=frame_paths, merged_text=merged_text)
        else:
            last_exc: Optional[Exception] = None
            for attempt in range(1, 4):
                try:
                    resp = self._call_with_video(video_uri=video_uri, merged_text=merged_text)
                    break
                except Exception as exc:  # dashscope 会封装 requests 异常，统一捕获
                    last_exc = exc
                    wait_s = min(2 ** (attempt - 1), 4)
                    print(f"[WARN] DashScope 调用失败（第 {attempt}/3 次），{wait_s}s 后重试: {exc}")
                    time.sleep(wait_s)
            else:
                try:
                    frame_paths = self._extract_temp_frames(video_path)
                    print(
                        f"[WARN] DashScope 视频上传失败，改为图片帧回退：video_id={video_id or Path(video_path).stem}, "
                        f"frames={len(frame_paths)}"
                    )
                    resp = self._call_with_images(frame_paths=frame_paths, merged_text=merged_text)
                except Exception:
                    raise last_exc  # type: ignore[misc]

        # 尝试取出 reasoning（若 SDK 返回）
        reasoning_out = ""
        try:
            content_items_dbg = resp.output.choices[0].message.content  # type: ignore[attr-defined]
            for it in content_items_dbg:
                if isinstance(it, dict) and ("reasoning_content" in it):
                    reasoning_out = str(it.get("reasoning_content") or "")
                    break
        except Exception:
            reasoning_out = ""

        # 取出模型文本输出
        # 文档示例：response.output.choices[0].message.content[0]["text"]
        try:
            content_items = resp.output.choices[0].message.content
            # 找到第一个含 text 的 item
            text_out = None
            for it in content_items:
                if isinstance(it, dict) and "text" in it:
                    text_out = it["text"]
                    break
            if text_out is None:
                text_out = str(content_items)
        except Exception as exc:
            raise RuntimeError(f"解析 DashScope 返回失败: {resp}") from exc

        raw_text = str(text_out).strip()
        parsed = extract_json_from_response(raw_text)

        if not isinstance(parsed, dict):
            raise ValueError(f"期望 JSON 对象，但得到: {type(parsed)}")

        parsed["__raw_text__"] = raw_text
        reasoning = reasoning_out or extract_reasoning_before_json(raw_text)
        if reasoning:
            parsed["__reasoning__"] = reasoning
        return parsed


__all__ = ["DashscopeMultiModalClient"]

