import asyncio
import json
import os
from typing import Any, Dict, Optional

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


class Seed20ArkClient:
    """
    使用火山引擎 Ark 官方 SDK (volcenginesdkarkruntime.AsyncArk)
    调用 Seed2.0-VL 模型进行视频理解。

    调用流程：
      1. 通过 files.create 上传本地视频，并在 preprocess_configs 里设置 fps。
      2. wait_for_processing 等待预处理完成，获得 file_id。
      3. 在 responses.create 的 input 中引用 {"type":"input_video","file_id": file.id}，
         再加 {"type":"input_text","text": <prompt>}。

    为了兼容现有 benchmark，predict_video 仍然要求模型输出一个 JSON，对应：
      - E, nu, density, yield_stress, material_type, motion_type
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key_env: str,
        fps: float = 1.0,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.fps = float(fps)

        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"环境变量 {api_key_env} 未设置，无法调用 Seed2.0-VL。")
        self.api_key = api_key

        try:
            # 官方异步客户端
            from volcenginesdkarkruntime import AsyncArk  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "未安装 volcenginesdkarkruntime。请先执行：pip install volcenginesdkarkruntime"
            ) from exc

        # 关键点：不要在每次 predict_video 里 asyncio.run() 新建/关闭事件循环。
        # 我们为整个 client 创建并复用一个 event loop + AsyncArk client，避免 httpx aclose 在 loop 关闭后触发。
        self._loop = asyncio.new_event_loop()
        self._client = AsyncArk(base_url=self.base_url, api_key=self.api_key)

    async def _async_predict_video(
        self,
        video_path: str,
        system_prompt: str,
        user_prompt: str,
        video_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"找不到视频文件: {video_path}")

        # 1) 上传视频，同时设置 fps
        print(f"[Seed2.0-Ark] 上传视频: {video_path}")
        with open(video_path, "rb") as f:
            file_obj = await self._client.files.create(
                file=f,
                purpose="user_data",
                preprocess_configs={
                    "video": {
                        "fps": self.fps,  # 文档中的 fps 参数
                    }
                },
            )
        print(f"[Seed2.0-Ark] 文件已上传，id={file_obj.id}，等待预处理...")

        # 2) 等待预处理完成
        await self._client.files.wait_for_processing(file_obj.id)
        print(f"[Seed2.0-Ark] 文件预处理完成: {file_obj.id}")

        # 3) 组装文本 prompt：合并 system_prompt + user_prompt
        merged_text = (system_prompt.strip() + "\n\n" + user_prompt.strip()).strip()

        # 4) 调用 responses.create
        resp = await self._client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_video",
                            "file_id": file_obj.id,
                        },
                        {
                            "type": "input_text",
                            "text": merged_text,
                        },
                    ],
                }
            ],
        )

        # 5) 从 resp 中抽取文本内容
        # 根据你贴的实际返回：resp.output 是一个 list，其中包含 reasoning item + message item；
        # message item 的 content 中有 {"type":"output_text","text": "..."}。
        try:
            text_out = ""
            reasoning_out = ""
            output_items = getattr(resp, "output", None)
            if isinstance(output_items, list):
                for out in output_items:
                    out_type = getattr(out, "type", None)
                    # reasoning item: out.summary -> list[Summary(text=...)]
                    if out_type == "reasoning":
                        summary_list = getattr(out, "summary", None)
                        if isinstance(summary_list, list) and summary_list:
                            first = summary_list[0]
                            reasoning_out = getattr(first, "text", "") or ""
                    # message item: out.content -> list[ResponseOutputText(...)]
                    content_items = getattr(out, "content", None)
                    if content_items is None:
                        continue
                    for item in content_items:
                        t = getattr(item, "type", None)
                        if t == "output_text":
                            text_out = getattr(item, "text", "") or ""
                            break
                    if text_out:
                        break
            if not text_out:
                # 兜底：把整个 resp 转字符串，避免 silent failure
                text_out = str(resp)
        except Exception as exc:
            raise RuntimeError(f"解析 Seed2.0-Ark 返回失败: {resp}") from exc

        raw_text = text_out.strip()
        parsed = extract_json_from_response(raw_text)

        if not isinstance(parsed, dict):
            raise ValueError(f"期望 JSON 对象，但得到: {type(parsed)}")

        parsed["__raw_text__"] = raw_text
        reasoning = reasoning_out or extract_reasoning_before_json(raw_text)
        if reasoning:
            parsed["__reasoning__"] = reasoning
        return parsed

    def predict_video(
        self,
        video_path: str,
        system_prompt: str,
        user_prompt: str,
        timeout: Optional[float] = None,
        video_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        同步封装，外部接口与其他 VLM 客户端保持一致。
        """
        return self._loop.run_until_complete(
            self._async_predict_video(
                video_path=video_path,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                video_id=video_id,
            )
        )

    def close(self) -> None:
        """
        主程序结束时可显式调用，清理底层 http 连接并关闭事件循环。
        """
        try:
            aclose = getattr(self._client, "aclose", None)
            if callable(aclose):
                self._loop.run_until_complete(aclose())
        finally:
            if not self._loop.is_closed():
                self._loop.close()

    def __del__(self) -> None:  # pragma: no cover
        # 尽力清理，避免 interpreter 退出时残留告警
        try:
            self.close()
        except Exception:
            pass


__all__ = ["Seed20ArkClient"]

