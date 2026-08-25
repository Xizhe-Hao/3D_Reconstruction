import json
import base64
import mimetypes
import os
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional


class DashscopeImageGenerationClient:
    """
    DashScope Qwen image-generation client for video-conditioned stress heatmaps.

    The image generation API returns short-lived image URLs, so this client
    downloads the first generated PNG immediately and returns local metadata.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key_env: str,
        max_frames: int = 64,
        n: int = 1,
        size: str = "512*512",
        watermark: bool = False,
        prompt_extend: bool = False,
        negative_prompt: str = "text, labels, axes, colorbar, borders, arrows, extra objects",
        retry_wait_s: float = 15.0,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.max_frames = max(int(max_frames), 1)
        self.n = max(int(n), 1)
        self.size = str(size)
        self.watermark = bool(watermark)
        self.prompt_extend = bool(prompt_extend)
        self.negative_prompt = str(negative_prompt)
        self.retry_wait_s = max(float(retry_wait_s), 0.0)

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

        dashscope.base_http_api_url = self.base_url
        self._mm = MultiModalConversation
        self._cv2 = cv2

    def _extract_video_frames(self, video_path: str) -> List["Any"]:
        cap = self._cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        total_frames = int(cap.get(self._cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            cap.release()
            raise ValueError(f"视频帧数无效: {video_path}")

        sample_count = min(self.max_frames, total_frames)
        indices = [int(round(x)) for x in self._linspace(0, total_frames - 1, sample_count)]
        frames = []
        try:
            for idx in indices:
                cap.set(self._cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if ok and frame is not None:
                    frames.append(frame)
        finally:
            cap.release()
        if not frames:
            raise ValueError(f"无法从视频抽帧: {video_path}")
        return frames

    @staticmethod
    def _linspace(start: int, stop: int, num: int) -> List[float]:
        if num <= 1:
            return [float(start)]
        step = (float(stop) - float(start)) / float(num - 1)
        return [float(start) + step * i for i in range(num)]

    def _video_to_contact_sheet(self, video_path: str) -> Path:
        frames = self._extract_video_frames(video_path)
        target_h = max(1, min(int(frame.shape[0]) for frame in frames))
        resized = []
        for frame in frames:
            h, w = frame.shape[:2]
            target_w = max(1, int(round(float(w) * float(target_h) / float(max(h, 1)))))
            resized.append(self._cv2.resize(frame, (target_w, target_h), interpolation=self._cv2.INTER_AREA))
        sheet = self._cv2.hconcat(resized)
        tmp_dir = Path(tempfile.mkdtemp(prefix="dashscope_qwen_image_"))
        out_path = tmp_dir / "video_contact_sheet.png"
        self._cv2.imwrite(str(out_path), sheet)
        return out_path

    @staticmethod
    def _encode_image_data_url(image_path: Path) -> str:
        mime_type, _ = mimetypes.guess_type(str(image_path))
        if not mime_type or not mime_type.startswith("image/"):
            mime_type = "image/png"
        data = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        return f"data:{mime_type};base64,{data}"

    def _call_image_model(self, *, input_image_path: Path, merged_text: str):
        input_image = self._encode_image_data_url(input_image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"image": input_image},
                    {"text": merged_text},
                ],
            }
        ]
        return self._mm.call(
            api_key=self.api_key,
            model=self.model,
            messages=messages,
            stream=False,
            n=self.n,
            watermark=self.watermark,
            negative_prompt=self.negative_prompt,
            prompt_extend=self.prompt_extend,
            size=self.size,
        )

    @staticmethod
    def _response_to_dict(resp: Any) -> Dict[str, Any]:
        try:
            return json.loads(json.dumps(resp, default=lambda o: getattr(o, "__dict__", str(o)), ensure_ascii=False))
        except Exception:
            return {"raw_response": str(resp)}

    def _extract_image_urls(self, resp: Any) -> List[str]:
        try:
            status_code = int(resp.status_code)
        except Exception:
            status_code = 200
        if status_code != 200:
            code = getattr(resp, "code", "")
            message = getattr(resp, "message", "")
            raise RuntimeError(f"DashScope 图片生成失败: status={status_code} code={code} message={message}")
        try:
            content_items = resp.output.choices[0].message.content
        except Exception as exc:
            raise RuntimeError(f"解析 DashScope 图片返回失败: {resp}") from exc
        urls = []
        for item in content_items:
            if isinstance(item, dict) and item.get("image"):
                urls.append(str(item["image"]))
        if not urls:
            raise RuntimeError(f"DashScope 图片返回中未找到 image URL: {resp}")
        return urls

    @staticmethod
    def _download_image(image_url: str, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(image_url, headers={"User-Agent": "linrui-vlm-benchmark/1.0"})
        with urllib.request.urlopen(req, timeout=300) as resp, output_path.open("wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)

    def generate_stress_image(
        self,
        video_path: str,
        *,
        system_prompt: str,
        user_prompt: str,
        output_path: str,
        timeout: Optional[float] = None,
        video_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"找不到视频文件: {video_path}")

        input_sheet_path = self._video_to_contact_sheet(video_path)
        merged_text = (system_prompt.strip() + "\n\n" + user_prompt.strip()).strip()

        last_exc: Optional[Exception] = None
        image_urls: Optional[List[str]] = None
        raw_response: Dict[str, Any] = {}
        for attempt in range(1, 4):
            try:
                resp = self._call_image_model(input_image_path=input_sheet_path, merged_text=merged_text)
                raw_response = self._response_to_dict(resp)
                image_urls = self._extract_image_urls(resp)
                break
            except Exception as exc:
                last_exc = exc
                wait_s = max(self.retry_wait_s, float(min(2 ** (attempt - 1), 4)))
                print(f"[WARN] DashScope 图片生成失败（第 {attempt}/3 次），{wait_s}s 后重试: {exc}")
                time.sleep(wait_s)
        if image_urls is None:
            raise last_exc  # type: ignore[misc]

        out_path = Path(output_path)
        self._download_image(image_urls[0], out_path)
        return {
            "sample_id": str(video_id or Path(video_path).stem),
            "generated_image_path": str(out_path),
            "generated_image_url": image_urls[0],
            "all_generated_image_urls": image_urls,
            "input_contact_sheet_path": str(input_sheet_path),
            "used_model": self.model,
            "size": self.size,
            "n": self.n,
            "watermark": self.watermark,
            "prompt_extend": self.prompt_extend,
            "__raw_response__": raw_response,
        }


__all__ = ["DashscopeImageGenerationClient"]
