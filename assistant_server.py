"""Run WebUI and HTTP API on the same port."""

import json
import re
from typing import Generator, List
import json_repair

import gradio as gr
from qwen_agent.gui import WebUI

from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
app = FastAPI()

chatbot_config = {
    'prompt.suggestions': {
        'hello': '你好!',
    },
    'verbose': False,
}
UI = WebUI(
    [],
    chatbot_config=chatbot_config,
)

def generate_streaming_responses(messages, bot, format) -> Generator[str, None, None]:
    """将 bot.run 的响应转为流式输出，兼容 JSON 结构化返回。"""
    count = 0
    response = None
    for response in bot.run(messages):
        count += 1
        # 定期返回空白，保持连接活跃
        if count % 10 == 0:
            yield '\n' if count % 500 == 0 else ' '
    if response is None:
        return

    result = response[-1]['content']

    if format == 'json':
        try:
            result = json_repair.loads(result)
        except Exception:
            # 保持容错，原样返回
            pass

    result = {
        'code': 0,
        'msg': 'success',
        'data': result,
    }
    yield json.dumps(result, ensure_ascii=False)

@app.post('/api/{agent}/{key}/{format}')
def prompt_any(key: str, payload: dict = None, agent: str = '', format: str = ''):
    """将推荐对话作为 prompt 模板，支持占位符替换与流式输出。"""

    key = key.replace('{agent}', agent).replace('{format}', format)
    prompts_map = UI.prompt_suggestions
    if key not in prompts_map:
        raise HTTPException(status_code=404, detail=f"prompt '{key}' not found")

    template = prompts_map[key]
    for k, v in (payload or {}).items():
        template = template.replace('{' + k + '}', str(v))
    messages = [{'role': 'user', 'content': template}]

    n = next((i for i, a in enumerate(UI.agent_list) if getattr(a, 'name', None) == agent), 0)
    agent_runner = UI.agent_hub or UI.agent_list[n]

    return StreamingResponse(
        generate_streaming_responses(messages, bot=agent_runner, format=format),
        media_type='application/json',
        headers={
            'X-Content-Type-Options': 'nosniff',
            'Connection': 'keep-alive',
            'Transfer-Encoding': 'chunked',
        },
    )

#同一端口既跑 Gradio WebUI，又暴露 API。经测试，此句必须放在@app.post(...) 之后，否则会覆盖路由。
gr.mount_gradio_app(app, UI.demo, path="/")

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0")
