# Copyright 2023 The Qwen team, Alibaba Group. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pprint
import re
import json
import uuid
from typing import List, Optional, Union

import yaml

from qwen_agent import Agent, MultiAgentHub
from qwen_agent.agents import Assistant
from qwen_agent.agents.user_agent import PENDING_USER_INPUT
from qwen_agent.gui.gradio_utils import format_cover_html
from qwen_agent.gui.utils import convert_fncall_to_text, convert_history_to_chatbot, get_avatar_image
from qwen_agent.llm.schema import AUDIO, CONTENT, FILE, IMAGE, NAME, ROLE, USER, VIDEO, Message
from qwen_agent.log import logger
from qwen_agent.utils.utils import print_traceback


class WebUI:
    """A Common chatbot application for agent."""

    def __init__(self,
                 agent: Union[Agent, MultiAgentHub, List[Agent]],
                 chatbot_config: Optional[dict] = None,
                 messages: List[Message] = None,
                 enable_mention: bool = False,
                 **kwargs):
        """
        Initialization the chatbot.

        Args:
            agent: The agent or a list of agents,
                supports various types of agents such as Assistant, GroupChat, Router, etc.
            chatbot_config: The chatbot configuration.
                Set the configuration as {'user.name': '', 'user.avatar': '', 'agent.avatar': '', 'input.placeholder': '', 'prompt.suggestions': []}.
            messages: Initial chat history to render in UI.
            enable_mention: 是否允许多 Agent 的 @ 提及。
            kwargs: 传递给 agent.run 的其余参数，供 WebUI 内部复用。
        """
        chatbot_config = chatbot_config or {}

        # === 运行参数保存 ===
        self.run_kwargs = kwargs
        self.enable_mention = enable_mention

        # === agent 相关初始化 ===
        if isinstance(agent, MultiAgentHub):
            initial_agent_list = [agent for agent in agent.nonuser_agents]
            self.agent_hub = agent
        elif isinstance(agent, list):
            initial_agent_list = agent
            self.agent_hub = None
        else:
            initial_agent_list = [agent]
            self.agent_hub = None

        user_name = chatbot_config.get('user.name', 'user')
        self.user_config = {
            'name': user_name,
            'avatar': chatbot_config.get(
                'user.avatar',
                get_avatar_image(user_name),
            ),
        }

        self.input_placeholder = chatbot_config.get('input.placeholder', '跟我聊聊吧～')
        # 初始推荐对话：来自配置
        self.prompt_suggestions = self._normalize_prompt_suggestions(
            chatbot_config.get('prompt.suggestions', {}))
        # 尝试加载历史持久化的推荐对话（YAML）
        persisted_suggestions = self._load_prompt_suggestions_from_yaml()
        if persisted_suggestions:
            # 以历史为主，配置为默认值
            base = dict(self.prompt_suggestions or {})
            base.update(persisted_suggestions)
            self.prompt_suggestions = base

        # 推荐对话初始选中项（用于首屏默认填充）
        self.initial_prompt_name = next(iter(self.prompt_suggestions), None)
        self.initial_prompt_text = ''
        if self.initial_prompt_name:
            _, self.initial_prompt_text = self.load_prompt_suggestion(
                self.initial_prompt_name)
        self.verbose = chatbot_config.get('verbose', False)

        # === 管理功能初始化 ===
        # llm_cfg管理：JSON array，每个元素包含model, model_type, model_server, api_key等
        self.llm_cfg_list = self._load_llm_cfg_list()
        # tools工具管理：JSON array，每个元素可能是字符串或MCP servers JSON
        self.tools_list = self._load_tools_list()
        # agent管理：agent列表，由llm_cfg和tools组合创建
        self.agent_cfg_list = self._load_agent_configs()
        # 动态创建的agent列表（从agent_configs创建）
        self.agent_list = [self._create_agent_from_config(cfg) for cfg in self.agent_cfg_list]
        for agent in initial_agent_list:
            self.agent_list.append(agent)
        
        # 初始化agent_config_list：优先使用动态agent列表，否则使用初始agent列表
        agent_config_list = [{
            'name': agent.name,
            'avatar': chatbot_config.get(
                'agent.avatar',
                get_avatar_image(agent.name),
            ),
            'description': agent.description or "I'm a helpful assistant.",
        } for agent in self.agent_list]

        # === 构建 Gradio UI ===
        from qwen_agent.gui.gradio_dep import gr, mgr, ms

        customTheme = gr.themes.Default(
            primary_hue=gr.themes.utils.colors.blue,
            radius_size=gr.themes.utils.sizes.radius_none,
        )

        with gr.Blocks(
                css=os.path.join(os.path.dirname(__file__), 'assets/appBot.css'),
                theme=customTheme,
        ) as demo:
            history = gr.State([])

            # === 管理面板：水平放置四个管理功能 ===
            with gr.Accordion("管理面板", open=False):
                with gr.Row():
                    with gr.Column():
                        # === 管理功能：LLM配置管理 ===
                        with gr.Group():
                            gr.Markdown("### LLM配置管理")
                            llm_cfg_choices_init = self._get_llm_cfg_choices(self.llm_cfg_list)
                            llm_cfg_selector = gr.Dropdown(
                                label='选择配置',
                                choices=llm_cfg_choices_init,
                                value=None if len(self.llm_cfg_list) == 0 else (llm_cfg_choices_init[-1][1] if llm_cfg_choices_init else None),
                                interactive=True,
                                allow_custom_value=False,
                            )
                            llm_cfg_json = gr.Textbox(
                                label='LLM配置 (JSON对象)',
                                value='',
                                placeholder='{"model": "qwen-plus", "model_type": "qwen_dashscope", "api_key": ""}',
                                lines=8,
                                interactive=True,
                            )
                            with gr.Row():
                                add_llm_cfg_btn = gr.Button('添加', variant='primary')
                                update_llm_cfg_btn = gr.Button('更新')
                                delete_llm_cfg_btn = gr.Button('删除', variant='stop')
                                reload_llm_cfg_btn = gr.Button('重新加载', variant="secondary")

                    with gr.Column():
                        # === 管理功能：工具管理 ===
                        with gr.Group():
                            gr.Markdown("### 工具管理")
                            tools_selector = gr.Dropdown(
                                label='选择工具',
                                choices=self._get_tools_choices(self.tools_list),
                                value=None if len(self.tools_list) == 0 else self._get_tools_choices(self.tools_list)[-1] if self.tools_list else None,
                                interactive=True,
                                allow_custom_value=False,
                            )
                            tools_json = gr.Textbox(
                                label='工具配置 (字符串或JSON对象)',
                                value='',
                                placeholder='例如: "code_interpreter" 或 {"mcpServers": {...}}',
                                lines=8,
                                interactive=True,
                            )
                            with gr.Row():
                                add_tools_btn = gr.Button('添加', variant='primary')
                                update_tools_btn = gr.Button('更新')
                                delete_tools_btn = gr.Button('删除', variant='stop')
                                reload_tools_btn = gr.Button('重新加载', variant="secondary")

                    with gr.Column():
                        # === 管理功能：Agent管理 ===
                        with gr.Group():
                            gr.Markdown("### Agent管理")
                            agent_configs_selector = gr.Dropdown(
                                label='选择Agent',
                                choices=self._get_agent_config_choices(self.agent_cfg_list),
                                value=None if len(self.agent_cfg_list) == 0 else self._get_agent_config_choices(self.agent_cfg_list)[-1] if self.agent_cfg_list else None,
                                interactive=True,
                                allow_custom_value=False,
                            )
                            agent_name_input = gr.Textbox(
                                label='Agent名称',
                                value='',
                                placeholder='例如: Qwen Assistant',
                                interactive=True,
                            )
                            agent_description_input = gr.Textbox(
                                label='Agent描述',
                                value='',
                                placeholder='例如: I\'m a helpful assistant.',
                                lines=2,
                                interactive=True,
                            )
                            agent_llm_cfg_choices_init = self._get_llm_cfg_choices(self.llm_cfg_list)
                            agent_llm_cfg_selector = gr.Dropdown(
                                label='LLM配置',
                                choices=agent_llm_cfg_choices_init,
                                value=None if len(self.llm_cfg_list) == 0 else (agent_llm_cfg_choices_init[0][1] if agent_llm_cfg_choices_init else None),
                                interactive=True,
                                allow_custom_value=False,
                            )
                            agent_tools_selector = gr.CheckboxGroup(
                                label='工具选择',
                                choices=self._get_tools_choices(self.tools_list),
                                value=[],
                                interactive=True,
                            )
                            with gr.Row():
                                add_agent_configs_btn = gr.Button('添加', variant='primary')
                                update_agent_configs_btn = gr.Button('更新')
                                delete_agent_configs_btn = gr.Button('删除', variant='stop')
                                reload_agent_configs_btn = gr.Button('重新加载', variant="secondary")

            with ms.Application():
                with gr.Row(elem_classes='container'):
                    with gr.Column(scale=4):
                        chatbot = mgr.Chatbot(value=convert_history_to_chatbot(messages=messages),
                                              avatar_images=[
                                                  self.user_config,
                                                  agent_config_list,
                                              ],
                                              height=850,
                                              avatar_image_width=80,
                                              flushing=False,
                                              show_copy_button=True,
                                              latex_delimiters=[{
                                                  'left': '\\(',
                                                  'right': '\\)',
                                                  'display': True
                                              }, {
                                                  'left': '\\begin{equation}',
                                                  'right': '\\end{equation}',
                                                  'display': True
                                              }, {
                                                  'left': '\\begin{align}',
                                                  'right': '\\end{align}',
                                                  'display': True
                                              }, {
                                                  'left': '\\begin{alignat}',
                                                  'right': '\\end{alignat}',
                                                  'display': True
                                              }, {
                                                  'left': '\\begin{gather}',
                                                  'right': '\\end{gather}',
                                                  'display': True
                                              }, {
                                                  'left': '\\begin{CD}',
                                                  'right': '\\end{CD}',
                                                  'display': True
                                              }, {
                                                  'left': '\\[',
                                                  'right': '\\]',
                                                  'display': True
                                              }])

                        with gr.Row():
                            if len(self.agent_list) >= 1:
                                agent_selector = gr.Dropdown(
                                    [(agent.name, i) for i, agent in enumerate(self.agent_list)],
                                    label='使用智能体',
                                    info='',
                                    value=0,
                                    interactive=True,
                                    scale=3
                                )
                            else:
                                agent_selector = gr.Dropdown(
                                    [],
                                    label='使用智能体',
                                    info='',
                                    value=None,
                                    interactive=False,
                                    scale=2
                                )

                            audio_input = gr.Audio(
                                sources=["microphone"],
                                type="filepath",
                                scale=3
                            )
                            # 添加清除按钮
                            clear_btn = gr.Button("🗑️ 清除会话", variant="secondary", scale=1)
                        input = mgr.MultimodalInput(placeholder=self.input_placeholder)

                    with gr.Column(scale=1):
                        agent_info_block = self._create_agent_info_block()

                        agent_plugins_block = self._create_agent_plugins_block()

                        # 推荐对话：基于「名称 -> 内容」的可增删改配置
                        with gr.Group():
                            gr.Markdown("### 提示词模板管理")
                            prompt_selector = gr.Dropdown(
                                label='选择提示词模板',
                                choices=list(self.prompt_suggestions.keys()) if self.prompt_suggestions else [],
                                value=self.initial_prompt_name,
                                interactive=True,
                            )
                            prompt_name = gr.Textbox(
                                label='名称', interactive=True, value=self.initial_prompt_name)
                            prompt_text = gr.Textbox(
                                label='内容', lines=4, interactive=True, value=self.initial_prompt_text)
                            with gr.Row():
                                apply_prompt_btn = gr.Button('应用到输入框', variant='primary')
                                save_prompt_btn = gr.Button('保存/更新')
                                delete_prompt_btn = gr.Button('删除', variant='stop')

                        # 选择推荐对话时，加载到编辑区
                        prompt_selector.change(
                            fn=self.load_prompt_suggestion,
                            inputs=[prompt_selector],
                            outputs=[prompt_name, prompt_text],
                            queue=False,
                        )

                        # 保存/更新推荐对话
                        save_prompt_btn.click(
                            fn=self.save_prompt_suggestion,
                            inputs=[prompt_name, prompt_text],
                            outputs=[prompt_selector],
                            queue=False,
                        )

                        # 删除推荐对话
                        delete_prompt_btn.click(
                            fn=self.delete_prompt_suggestion,
                            inputs=[prompt_name],
                            outputs=[prompt_selector, prompt_name, prompt_text],
                            queue=False,
                        )

                        # 将选中的推荐对话应用到输入框
                        apply_prompt_btn.click(
                            fn=self.apply_prompt_suggestion,
                            inputs=[prompt_selector],
                            outputs=[input],
                            queue=False,
                        )

                        # LLM配置管理事件
                        llm_cfg_selector.change(
                            fn=self.load_llm_cfg_item,
                            inputs=[llm_cfg_selector],
                            outputs=[llm_cfg_json],
                            queue=False,
                        )
                        add_llm_cfg_btn.click(
                            fn=self.add_llm_cfg_item,
                            inputs=[llm_cfg_json],
                            outputs=[llm_cfg_selector, llm_cfg_json],
                            queue=False,
                        )
                        update_llm_cfg_btn.click(
                            fn=self.update_llm_cfg_item,
                            inputs=[llm_cfg_selector, llm_cfg_json],
                            outputs=[llm_cfg_selector, llm_cfg_json],
                            queue=False,
                        )
                        delete_llm_cfg_btn.click(
                            fn=self.delete_llm_cfg_item,
                            inputs=[llm_cfg_selector],
                            outputs=[llm_cfg_selector, llm_cfg_json],
                            queue=False,
                        )
                        reload_llm_cfg_btn.click(
                            fn=self.reload_llm_cfg_list,
                            inputs=[],
                            outputs=[llm_cfg_selector, llm_cfg_json],
                            queue=False,
                        )

                        # 工具管理事件
                        tools_selector.change(
                            fn=self.load_tools_item,
                            inputs=[tools_selector],
                            outputs=[tools_json],
                            queue=False,
                        )
                        add_tools_btn.click(
                            fn=self.add_tools_item,
                            inputs=[tools_json],
                            outputs=[tools_selector, tools_json],
                            queue=False,
                        )
                        update_tools_btn.click(
                            fn=self.update_tools_item,
                            inputs=[tools_selector, tools_json],
                            outputs=[tools_selector, tools_json],
                            queue=False,
                        )
                        delete_tools_btn.click(
                            fn=self.delete_tools_item,
                            inputs=[tools_selector],
                            outputs=[tools_selector, tools_json],
                            queue=False,
                        )
                        reload_tools_btn.click(
                            fn=self.reload_tools_list,
                            inputs=[],
                            outputs=[tools_selector, tools_json],
                            queue=False,
                        )

                        # Agent管理事件 - 当LLM配置或工具列表变化时，更新下拉框选项
                        def update_agent_llm_cfg_choices():
                            """更新Agent管理中的LLM配置下拉框"""
                            from qwen_agent.gui.gradio_dep import gr
                            choices = self._get_llm_cfg_choices(self.llm_cfg_list)
                            # choices格式为 [(显示名称, ID), ...]
                            return gr.update(choices=choices, value=choices[0][1] if choices else None)
                        
                        def update_agent_tools_choices():
                            """更新Agent管理中的工具多选框"""
                            from qwen_agent.gui.gradio_dep import gr
                            choices = self._get_tools_choices(self.tools_list)
                            return gr.update(choices=choices, value=[])

                        agent_configs_selector.change(
                            fn=self.load_agent_config_item,
                            inputs=[agent_configs_selector],
                            outputs=[agent_name_input, agent_description_input, agent_llm_cfg_selector, agent_tools_selector],
                            queue=False,
                        )
                        add_agent_configs_btn.click(
                            fn=self.add_agent_config_item,
                            inputs=[agent_name_input, agent_description_input, agent_llm_cfg_selector, agent_tools_selector],
                            outputs=[agent_configs_selector, agent_name_input, agent_description_input, agent_llm_cfg_selector, agent_tools_selector, agent_selector, agent_info_block, agent_plugins_block],
                            queue=False,
                        )
                        update_agent_configs_btn.click(
                            fn=self.update_agent_config_item,
                            inputs=[agent_configs_selector, agent_name_input, agent_description_input, agent_llm_cfg_selector, agent_tools_selector],
                            outputs=[agent_configs_selector, agent_name_input, agent_description_input, agent_llm_cfg_selector, agent_tools_selector, agent_selector, agent_info_block, agent_plugins_block],
                            queue=False,
                        )
                        delete_agent_configs_btn.click(
                            fn=self.delete_agent_config_item,
                            inputs=[agent_configs_selector],
                            outputs=[agent_configs_selector, agent_name_input, agent_description_input, agent_llm_cfg_selector, agent_tools_selector, agent_selector, agent_info_block, agent_plugins_block],
                            queue=False,
                        )
                        reload_agent_configs_btn.click(
                            fn=self.reload_agent_configs,
                            inputs=[],
                            outputs=[agent_configs_selector, agent_name_input, agent_description_input, agent_llm_cfg_selector, agent_tools_selector],
                            queue=False,
                        )

                    # 获取当前可用的agent列表（动态或静态）
                    if len(self.agent_list) > 1:
                        agent_selector.change(
                            fn=self.change_agent,
                            inputs=[agent_selector],
                            outputs=[agent_selector, agent_info_block, agent_plugins_block],
                            queue=False,
                        )

                    # 添加清除按钮的点击事件
                    clear_btn.click(
                        fn=self.clear_chat_history,
                        inputs=[chatbot, history],
                        outputs=[chatbot, history],
                        queue=False
                    )

                    input_promise = input.submit(
                        fn=self.add_text,
                        inputs=[input, audio_input, chatbot, history],
                        outputs=[input, audio_input, chatbot, history],
                        queue=False,
                    )

                    if len(self.agent_list) > 1: # and self.enable_mention:
                        input_promise = input_promise.then(
                            self.add_mention,
                            [chatbot, agent_selector],
                            [chatbot, agent_selector],
                        ).then(
                            self.agent_run,
                            [chatbot, history, agent_selector],
                            [chatbot, history, agent_selector],
                        )
                    else:
                        input_promise = input_promise.then(
                            self.agent_run,
                            [chatbot, history],
                            [chatbot, history],
                        )

                    input_promise.then(lambda _: gr.update(interactive=True), None, [input])

            demo.load(
                fn=self._load_latest_settings,
                inputs=[],
                outputs=[
                    prompt_selector,
                    prompt_name,
                    prompt_text,

                    llm_cfg_selector,
                    llm_cfg_json,

                    tools_selector,
                    tools_json,

                    agent_configs_selector,
                    agent_name_input,
                    agent_description_input,
                    agent_llm_cfg_selector,
                    agent_tools_selector
                ],
                queue=False
            )

        # 暴露 Blocks 与底层 FastAPI app，便于在同一端口上由外部注入自定义 API
        self.demo = demo

    def clear_chat_history(self, _chatbot, _history):
        """清除聊天历史记录"""
        from qwen_agent.gui.gradio_dep import gr
        
        # 重置聊天记录为空列表
        new_chatbot = []
        new_history = []
        
        return new_chatbot, new_history

    def run(self,
            share: bool = False,
            server_name: str = None,
            server_port: int = None,
            concurrency_limit: int = 10):
        """仅负责启动服务。其它初始化已在 __init__ 完成。"""
        self.demo.queue(default_concurrency_limit=concurrency_limit).launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
        )

    def change_agent(self, agent_selector):
        # 获取当前可用的agent列表（动态或静态）
        if agent_selector is None or agent_selector >= len(self.agent_list):
            agent_selector = 0
        yield agent_selector, self._create_agent_info_block(agent_selector), self._create_agent_plugins_block(
            agent_selector)

    def add_text(self, _input, _audio_input, _chatbot, _history):
        _history.append({
            ROLE: USER,
            CONTENT: [{
                'text': _input.text
            }],
        })

        if self.user_config[NAME]:
            _history[-1][NAME] = self.user_config[NAME]
        
        # if got audio from microphone, append it to the multimodal inputs
        if _audio_input:
            from qwen_agent.gui.gradio_dep import gr, mgr, ms
            audio_input_file = gr.data_classes.FileData(path=_audio_input, mime_type="audio/wav")
            _input.files.append(audio_input_file)

        if _input.files:
            for file in _input.files:
                if file.mime_type.startswith('image/'):
                    _history[-1][CONTENT].append({IMAGE: 'file://' + file.path})
                elif file.mime_type.startswith('audio/'):
                    _history[-1][CONTENT].append({AUDIO: 'file://' + file.path})
                elif file.mime_type.startswith('video/'):
                    _history[-1][CONTENT].append({VIDEO: 'file://' + file.path})
                else:
                    _history[-1][CONTENT].append({FILE: file.path})

        _chatbot.append([_input, None])

        from qwen_agent.gui.gradio_dep import gr

        yield gr.update(interactive=False, value=None), None, _chatbot, _history

    def add_mention(self, _chatbot, _agent_selector):
        # 获取当前可用的agent列表（动态或静态）
        if len(self.agent_list) == 1:
            yield _chatbot, _agent_selector

        query = _chatbot[-1][0].text
        match = re.search(r'@\w+\b', query)
        if match:
            _agent_selector = self._get_agent_index_by_name(match.group()[1:], self.agent_list)

        agent_name = self.agent_list[_agent_selector].name

        if ('@' + agent_name) not in query and self.agent_hub is None:
            _chatbot[-1][0].text = '@' + agent_name + ' ' + query

        yield _chatbot, _agent_selector

    def agent_run(self, _chatbot, _history, _agent_selector=None):
        if self.verbose:
            logger.info('agent_run input:\n' + pprint.pformat(_history, indent=2))

        # 获取当前可用的agent列表（动态或静态）
        num_input_bubbles = len(_chatbot) - 1
        num_output_bubbles = 1
        _chatbot[-1][1] = [None for _ in range(len(self.agent_list))]

        agent_runner = self.agent_list[_agent_selector or 0]
        if self.agent_hub:
            agent_runner = self.agent_hub
        responses = []
        for responses in agent_runner.run(_history, **self.run_kwargs):
            if not responses:
                continue
            if responses[-1][CONTENT] == PENDING_USER_INPUT:
                logger.info('Interrupted. Waiting for user input!')
                break

            display_responses = convert_fncall_to_text(responses)
            if not display_responses:
                continue
            if display_responses[-1][CONTENT] is None:
                continue

            while len(display_responses) > num_output_bubbles:
                # Create a new chat bubble
                _chatbot.append([None, None])
                _chatbot[-1][1] = [None for _ in range(len(self.agent_list))]
                num_output_bubbles += 1

            assert num_output_bubbles == len(display_responses)
            assert num_input_bubbles + num_output_bubbles == len(_chatbot)

            for i, rsp in enumerate(display_responses):
                agent_index = self._get_agent_index_by_name(rsp[NAME], self.agent_list)
                _chatbot[num_input_bubbles + i][1][agent_index] = rsp[CONTENT]

            if len(self.agent_list) > 1:
                _agent_selector = agent_index

            if _agent_selector is not None:
                yield _chatbot, _history, _agent_selector
            else:
                yield _chatbot, _history

        if responses:
            _history.extend([res for res in responses if res[CONTENT] != PENDING_USER_INPUT])

        if _agent_selector is not None:
            yield _chatbot, _history, _agent_selector
        else:
            yield _chatbot, _history

        if self.verbose:
            logger.info('agent_run response:\n' + pprint.pformat(responses, indent=2))

    def _normalize_prompt_suggestions(self, raw_suggestions):
        """将各种形式的 prompt.suggestions 统一为 {name: suggestion} 的字典。

        支持：
        - 直接传 dict: {name: suggestion}
        - 传 list: [suggestion1, suggestion2, ...]，自动命名为 示例1、示例2 ...
        """
        if isinstance(raw_suggestions, dict):
            return raw_suggestions
        if isinstance(raw_suggestions, list):
            suggestions = {}
            for i, item in enumerate(raw_suggestions):
                name = f'{i + 1}'
                suggestions[name] = item
            return suggestions
        return {}

    # === 推荐对话 YAML 持久化相关工具函数 ===
    def _get_prompt_yaml_path(self) -> str:
        """获取推荐对话持久化文件路径，位于用户家目录下的 .qwen_agent 目录。"""
        home = os.path.expanduser('~')
        config_dir = os.path.join(home, '.qwen_agent')
        return os.path.join(config_dir, 'prompt_suggestions.yaml')

    def _load_prompt_suggestions_from_yaml(self) -> dict:
        """从 YAML 文件加载推荐对话 map，如果不存在或出错则返回空字典。"""
        yaml_path = self._get_prompt_yaml_path()
        if not os.path.exists(yaml_path):
            return {}
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
            if isinstance(data, dict):
                return data
            return {}
        except Exception:
            # 读取或解析失败时不影响程序运行，只打印错误日志
            print_traceback()
            return {}

    def _load_latest_settings(self):
        """页面加载（首次/刷新）时，从YAML读取最新提示词配置，返回组件更新值"""
        from qwen_agent.gui.gradio_dep import gr

        # 1. 读取磁盘上最新的提示词配置
        latest_suggestions = self._load_prompt_suggestions_from_yaml()
        # 2. 更新实例属性，保持状态一致
        self.prompt_suggestions = latest_suggestions
        # 3. 确定默认选中项
        new_selected_name = next(iter(latest_suggestions.keys()), None) if latest_suggestions else None

        # 4. 获取默认选中项对应的名称和内容
        prompt_name_val = ""
        prompt_text_val = ""
        if new_selected_name:
            prompt_name_val, prompt_text_val = self.load_prompt_suggestion(new_selected_name)

        latest_llm_cfg_list = self._load_llm_cfg_list()
        llm_cfg_choices = self._get_llm_cfg_choices(latest_llm_cfg_list)
        llm_cfg_selected_val = llm_cfg_choices[-1][1] if llm_cfg_choices else None

        latest_tools_list = self._load_tools_list()
        tools_choices = self._get_tools_choices(latest_tools_list)
        tools_selected_val = tools_choices[-1] if tools_choices else None

        latest_agent_configs = self._load_agent_configs()
        agent_configs_choices = self._get_agent_config_choices(latest_agent_configs)
        agent_configs_selected_val = agent_configs_choices[-1] if agent_configs_choices else None

        agent_llm_cfg_choices = self._get_llm_cfg_choices(latest_llm_cfg_list)
        agent_llm_cfg_selected_val = agent_llm_cfg_choices[0][1] if agent_llm_cfg_choices else None
        agent_tools_choices = self._get_tools_choices(latest_tools_list)
        agent_tools_selected_val = []

        # 5. 返回组件更新值
        return (
            gr.update(
                choices=list(latest_suggestions.keys()) if latest_suggestions else [],
                value=new_selected_name
            ),
            gr.update(value=prompt_name_val),
            gr.update(value=prompt_text_val),

            gr.update(
                choices=llm_cfg_choices,
                value=llm_cfg_selected_val
            ),
            gr.update(value=''),

            gr.update(
                choices=tools_choices,
                value=tools_selected_val
            ),
            gr.update(value=''),

            gr.update(
                choices=agent_configs_choices,
                value=agent_configs_selected_val
            ),
            gr.update(value=''),
            gr.update(value=''),
            gr.update(
                choices=agent_llm_cfg_choices,
                value=agent_llm_cfg_selected_val
            ),
            gr.update(
                choices=agent_tools_choices,
                value=agent_tools_selected_val
            )
        )
    
    def _save_prompt_suggestions_to_yaml(self, suggestions: dict) -> None:
        """将当前推荐对话 map 持久化到 YAML 文件。"""
        try:
            yaml_path = self._get_prompt_yaml_path()
            config_dir = os.path.dirname(yaml_path)
            os.makedirs(config_dir, exist_ok=True)
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(
                    suggestions or {},
                    f,
                    allow_unicode=True,
                    sort_keys=True,
                )
        except Exception:
            # 持久化失败不影响前端使用，只打印错误日志
            print_traceback()

    def save_prompt_suggestion(self, name, text):
        """新增/更新推荐对话，并实时刷新下拉列表。"""
        from qwen_agent.gui.gradio_dep import gr

        suggestions = self.prompt_suggestions
        name = (name or '').strip()
        if not name:
            # 忽略空名称
            return suggestions, gr.update(choices=list(suggestions.keys()) if suggestions else [])

        existing = suggestions.get(name)
        if isinstance(existing, dict):
            # 保留原有结构中的其它字段（例如 files），只更新 text
            new_value = dict(existing)
            new_value['text'] = text
        else:
            # 新建或覆盖为简单文本
            new_value = text

        suggestions[name] = new_value

        # 持久化到 YAML
        self._save_prompt_suggestions_to_yaml(suggestions)
        self.prompt_suggestions = suggestions

        return gr.update(choices=list(suggestions.keys()) if suggestions else [], value=name)

    def load_prompt_suggestion(self, selected_name):
        """根据下拉选择将推荐对话加载到右侧编辑区。"""
        suggestions = self.prompt_suggestions
        if not selected_name or selected_name not in suggestions:
            return '', ''

        value = suggestions[selected_name]
        if isinstance(value, dict):
            text = value.get('text', '') or ''
        else:
            text = str(value)
        return selected_name, text

    def delete_prompt_suggestion(self, name):
        from qwen_agent.gui.gradio_dep import gr

        # 1. 直接修改状态变量
        suggestions = self.prompt_suggestions
        name = (name or '').strip()
        if name in suggestions:
            suggestions.pop(name)

        # 2. 持久化 + 同步实例变量
        self._save_prompt_suggestions_to_yaml(suggestions)
        self.prompt_suggestions = suggestions.copy()

        # 3. 直接推送新的 choices 到前端
        new_selected = next(iter(suggestions.keys()), None) if suggestions else None
        return (
            gr.update(choices=list(suggestions.keys()), value=new_selected),  # 直接更新下拉框
            gr.update(value=''),
            gr.update(value='')
        )

    def apply_prompt_suggestion(self, selected_name):
        """将选中的推荐对话内容应用到多模态输入组件中。"""
        from qwen_agent.gui.gradio_dep import gr

        suggestions = self.prompt_suggestions
        if not selected_name or selected_name not in suggestions:
            return gr.update()

        value = suggestions[selected_name]
        # 这里直接将原始值作为 MultimodalInput 的 value，
        # 与 gr.Examples 的行为保持一致（可以是 str 或 dict(text, files, ...)）
        return gr.update(value=value)

    def _get_agent_index_by_name(self, agent_name, agent_list=None):
        if agent_name is None:
            return 0

        try:
            agent_name = agent_name.strip()
            for i, agent in enumerate(agent_list):
                if agent.name == agent_name:
                    return i
            return 0
        except Exception:
            print_traceback()
            return 0

    def _create_agent_info_block(self, agent_index=0):
        from qwen_agent.gui.gradio_dep import gr

        if agent_index >= len(self.agent_cfg_list):
            agent_index = 0

        if agent_index < len(self.agent_cfg_list):
            agent_config_interactive = self.agent_cfg_list[agent_index]
            return gr.HTML(
                format_cover_html(
                    bot_name=agent_config_interactive['name'],
                    bot_description=agent_config_interactive['description'],
                    bot_avatar=agent_config_interactive['avatar'] if 'avatar' in agent_config_interactive else (self.user_config.get('agent.avatar') or get_avatar_image(agent_config_interactive['name'])),
                ))
        else:
            return gr.HTML(
                format_cover_html(
                    bot_name='未知智能体',
                    bot_description='未找到对应的智能体配置。',
                    bot_avatar='',
                ))

    def _create_agent_plugins_block(self, agent_index=0):
        from qwen_agent.gui.gradio_dep import gr

        if agent_index >= len(self.agent_list):
            agent_index = 0

        if agent_index < len(self.agent_list):
            agent_interactive = self.agent_list[agent_index]
        else:
            agent_interactive = None

        if agent_interactive and agent_interactive.function_map:
            capabilities = [key for key in agent_interactive.function_map.keys()]
            return gr.CheckboxGroup(
                label='插件',
                value=capabilities,
                choices=capabilities,
                interactive=False,
            )
        else:
            return gr.CheckboxGroup(
                label='插件',
                value=[],
                choices=[],
                interactive=False,
            )

    # === LLM配置管理相关方法 ===
    def _format_llm_cfg_name(self, llm_cfg: dict, index: int = None) -> str:
        """从LLM配置生成有意义的名称"""
        if not isinstance(llm_cfg, dict):
            return f"配置 {index + 1}" if index is not None else "未知配置"
        
        parts = []
        # 优先使用model字段
        if 'model' in llm_cfg and llm_cfg['model']:
            parts.append(str(llm_cfg['model']))
        
        # 添加model_type信息
        if 'model_type' in llm_cfg and llm_cfg['model_type']:
            parts.append(f"({llm_cfg['model_type']})")
        
        # 如果有model_server，添加服务器信息
        if 'model_server' in llm_cfg and llm_cfg['model_server']:
            server = str(llm_cfg['model_server'])
            # 简化显示，只显示主机名或端口
            if '://' in server:
                server = server.split('://')[-1]
            if '/' in server:
                server = server.split('/')[0]
            parts.append(f"[{server}]")
        
        if parts:
            return ' '.join(parts)
        else:
            # 如果没有关键字段，使用索引
            return f"配置 {index + 1}" if index is not None else "未知配置"

    def _get_llm_cfg_choices(self, llm_cfg_list: list) -> list:
        """生成LLM配置下拉选择器的选项列表，返回格式为 [(显示名称, ID), ...]"""
        choices = []
        for i, cfg in enumerate(llm_cfg_list):
            if isinstance(cfg, dict):
                # 确保有ID
                if 'id' not in cfg:
                    cfg['id'] = str(uuid.uuid4())
                display_name = self._format_llm_cfg_name(cfg, i)
                choices.append((display_name, cfg['id']))
            else:
                # 兼容旧格式，生成ID
                cfg_id = str(uuid.uuid4())
                display_name = self._format_llm_cfg_name(cfg, i)
                choices.append((display_name, cfg_id))
        return choices

    def _get_llm_cfg_by_id(self, cfg_id: str, llm_cfg_list: list) -> Optional[dict]:
        """根据ID查找LLM配置"""
        if not cfg_id:
            return None
        for cfg in llm_cfg_list:
            if isinstance(cfg, dict) and cfg.get('id') == cfg_id:
                return cfg
        return None
    
    def _get_llm_cfg_index_by_id(self, cfg_id: str, llm_cfg_list: list) -> int:
        """根据ID查找LLM配置的索引"""
        if not cfg_id:
            return -1
        for i, cfg in enumerate(llm_cfg_list):
            if isinstance(cfg, dict) and cfg.get('id') == cfg_id:
                return i
        return -1

    def _get_llm_cfg_yaml_path(self) -> str:
        """获取LLM配置持久化文件路径"""
        home = os.path.expanduser('~')
        config_dir = os.path.join(home, '.qwen_agent')
        return os.path.join(config_dir, 'llm_cfg_list.yaml')

    def _load_llm_cfg_list(self) -> list:
        """从YAML文件加载LLM配置列表，确保每个配置都有唯一ID"""
        yaml_path = self._get_llm_cfg_yaml_path()
        if not os.path.exists(yaml_path):
            return []
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or []
            if isinstance(data, list):
                # 确保每个配置都有唯一ID
                for cfg in data:
                    if isinstance(cfg, dict) and 'id' not in cfg:
                        cfg['id'] = str(uuid.uuid4())
                return data
            return []
        except Exception:
            print_traceback()
            return []

    def _save_llm_cfg_list(self, llm_cfg_list: list) -> None:
        """将LLM配置列表持久化到YAML文件"""
        try:
            yaml_path = self._get_llm_cfg_yaml_path()
            config_dir = os.path.dirname(yaml_path)
            os.makedirs(config_dir, exist_ok=True)
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(llm_cfg_list or [], f, allow_unicode=True, sort_keys=False)
        except Exception:
            print_traceback()

    def reload_llm_cfg_list(self):
        """重新加载LLM配置列表"""
        from qwen_agent.gui.gradio_dep import gr
        self.llm_cfg_list = self._load_llm_cfg_list()
        choices = self._get_llm_cfg_choices(self.llm_cfg_list)
        new_value = choices[-1][1] if choices else None
        return gr.update(choices=choices, value=new_value), gr.update(value='')

    def load_llm_cfg_item(self, selector):
        """加载选中的LLM配置项"""
        from qwen_agent.gui.gradio_dep import gr
        if selector is None or not selector:
            return gr.update(value='')
        try:
            cfg = self._get_llm_cfg_by_id(selector, self.llm_cfg_list)
            if cfg:
                # 创建副本，排除ID字段用于显示（ID是只读的）
                display_cfg = {k: v for k, v in cfg.items() if k != 'id'}
                return gr.update(value=json.dumps(display_cfg, ensure_ascii=False, indent=2))
        except Exception:
            print_traceback()
        return gr.update(value='')

    def add_llm_cfg_item(self, llm_cfg_json_str):
        """添加新的LLM配置项"""
        from qwen_agent.gui.gradio_dep import gr
        try:
            llm_cfg = json.loads(llm_cfg_json_str)
            if not isinstance(llm_cfg, dict):
                raise ValueError('LLM配置必须是JSON对象格式')
            # 为新配置生成唯一ID
            llm_cfg['id'] = str(uuid.uuid4())
            llm_cfg_list = list(self.llm_cfg_list)
            llm_cfg_list.append(llm_cfg)
            self.llm_cfg_list = llm_cfg_list
            self._save_llm_cfg_list(llm_cfg_list)
            choices = self._get_llm_cfg_choices(llm_cfg_list)
            new_id = llm_cfg['id']
            return gr.update(choices=choices, value=new_id), gr.update(value='')
        except Exception:
            print_traceback()
            return gr.update(), gr.update()

    def update_llm_cfg_item(self, selector, llm_cfg_json_str):
        """更新选中的LLM配置项，保持ID不变"""
        from qwen_agent.gui.gradio_dep import gr
        if selector is None or not selector:
            return gr.update(), gr.update()
        try:
            llm_cfg = json.loads(llm_cfg_json_str)
            if not isinstance(llm_cfg, dict):
                raise ValueError('LLM配置必须是JSON对象格式')
            llm_cfg_list = list(self.llm_cfg_list)
            index = self._get_llm_cfg_index_by_id(selector, llm_cfg_list)
            if 0 <= index < len(llm_cfg_list):
                # 保持原有的ID不变（ID是只读的）
                old_id = llm_cfg_list[index].get('id')
                if old_id:
                    llm_cfg['id'] = old_id
                else:
                    llm_cfg['id'] = str(uuid.uuid4())
                llm_cfg_list[index] = llm_cfg
                self.llm_cfg_list = llm_cfg_list
                self._save_llm_cfg_list(llm_cfg_list)
                choices = self._get_llm_cfg_choices(llm_cfg_list)
                # 保持选中同一个ID
                return gr.update(choices=choices, value=llm_cfg['id']), gr.update(value=llm_cfg_json_str)
        except Exception:
            print_traceback()
        return gr.update(), gr.update()

    def delete_llm_cfg_item(self, selector):
        """删除选中的LLM配置项，删除前检查是否有Agent引用"""
        from qwen_agent.gui.gradio_dep import gr
        if selector is None or not selector:
            return gr.update(), gr.update()
        try:
            # 检查是否有Agent引用此LLM配置
            referenced_agents = []
            # 获取要删除的配置的索引
            delete_index = self._get_llm_cfg_index_by_id(selector, self.llm_cfg_list)
            
            for agent_cfg in self.agent_cfg_list:
                # 检查新格式：使用ID
                if agent_cfg.get('llm_cfg_id') == selector:
                    referenced_agents.append(agent_cfg.get('name', '未知Agent'))
                # 检查旧格式：使用索引（兼容性）
                elif delete_index >= 0:
                    llm_cfg_index = agent_cfg.get('llm_cfg_index')
                    if llm_cfg_index is not None and llm_cfg_index == delete_index:
                        referenced_agents.append(agent_cfg.get('name', '未知Agent'))
            
            if referenced_agents:
                # 有引用，不允许删除
                raise ValueError(f'无法删除：以下Agent正在使用此LLM配置：{", ".join(referenced_agents)}')
            
            if 0 <= delete_index < len(self.llm_cfg_list):
                self.llm_cfg_list.pop(delete_index)
                self._save_llm_cfg_list(self.llm_cfg_list)
                choices = self._get_llm_cfg_choices(self.llm_cfg_list)
                new_value = choices[0][1] if choices else None
                return gr.update(choices=choices, value=new_value), gr.update(value='')
        except ValueError as e:
            # 返回错误信息，但不删除
            return gr.update(), gr.update(value=str(e))
        except Exception:
            print_traceback()
        return gr.update(), gr.update()

    # === 工具管理相关方法 ===
    def _format_tool_name(self, tool, index: int = None) -> str:
        """从工具配置生成有意义的名称"""
        if isinstance(tool, str):
            # 字符串类型，直接显示字符串本身
            return tool
        elif isinstance(tool, dict):
            # MCP工具，提取名称
            if 'mcpServers' in tool and isinstance(tool['mcpServers'], dict):
                # 获取所有MCP server的名称
                server_names = list(tool['mcpServers'].keys())
                if server_names:
                    # 如果有多个server，用逗号连接；如果只有一个，直接显示
                    if len(server_names) == 1:
                        return f"{server_names[0]}@MCP"
                    else:
                        return f"{','.join(server_names)}@MCP"
                else:
                    return "MCP工具"
            else:
                # 其他字典类型，尝试找name字段
                if 'name' in tool:
                    return str(tool['name'])
                # 如果整个字典被当作字符串显示，尝试JSON序列化看看
                return "工具配置"
        else:
            return f"工具 {index + 1}" if index is not None else "未知工具"

    def _get_tools_choices(self, tools_list: list) -> list:
        """生成工具下拉选择器的选项列表"""
        return [self._format_tool_name(tool, i) for i, tool in enumerate(tools_list)]

    def _get_tool_index_by_name(self, name: str, tools_list: list) -> int:
        """根据名称查找工具的索引"""
        for i, tool in enumerate(tools_list):
            if self._format_tool_name(tool, i) == name:
                return i
        return -1

    def _get_tools_yaml_path(self) -> str:
        """获取工具配置持久化文件路径"""
        home = os.path.expanduser('~')
        config_dir = os.path.join(home, '.qwen_agent')
        return os.path.join(config_dir, 'tools_list.yaml')

    def _load_tools_list(self) -> list:
        """从YAML文件加载工具列表"""
        yaml_path = self._get_tools_yaml_path()
        if not os.path.exists(yaml_path):
            return []
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or []
            if isinstance(data, list):
                return data
            return []
        except Exception:
            print_traceback()
            return []

    def _save_tools_list(self, tools_list: list) -> None:
        """将工具列表持久化到YAML文件"""
        try:
            yaml_path = self._get_tools_yaml_path()
            config_dir = os.path.dirname(yaml_path)
            os.makedirs(config_dir, exist_ok=True)
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(tools_list or [], f, allow_unicode=True, sort_keys=False)
        except Exception:
            print_traceback()

    def reload_tools_list(self):
        """重新加载工具列表"""
        from qwen_agent.gui.gradio_dep import gr
        self.tools_list = self._load_tools_list()
        choices = self._get_tools_choices(self.tools_list)
        return gr.update(choices=choices, value=None), gr.update(value='')

    def load_tools_item(self, selector):
        """加载选中的工具项"""
        from qwen_agent.gui.gradio_dep import gr
        if selector is None or not selector:
            return gr.update(value='')
        try:
            tools_list = self.tools_list
            index = self._get_tool_index_by_name(selector, tools_list)
            if 0 <= index < len(tools_list):
                item = tools_list[index]
                if isinstance(item, str):
                    # 字符串类型，直接显示（不需要JSON转义）
                    return gr.update(value=item)
                else:
                    # MCP工具或其他字典类型，使用pretty JSON格式
                    return gr.update(value=json.dumps(item, ensure_ascii=False, indent=2))
        except Exception:
            print_traceback()
        return gr.update(value='')

    def add_tools_item(self, tools_json_str):
        """添加新的工具项"""
        from qwen_agent.gui.gradio_dep import gr
        try:
            # 尝试解析为JSON，如果失败则作为字符串处理
            tool = None
            try:
                tool = json.loads(tools_json_str)
            except json.JSONDecodeError:
                # 如果不是JSON，则作为字符串处理（去掉引号）
                tool = tools_json_str.strip().strip('"').strip("'")
            
            if tool is None:
                raise ValueError('工具配置不能为空')
            
            tools_list = list(self.tools_list)
            tools_list.append(tool)
            self.tools_list = tools_list
            self._save_tools_list(tools_list)
            choices = self._get_tools_choices(tools_list)
            new_name = choices[-1] if choices else None
            return gr.update(choices=choices, value=new_name), gr.update(value='')
        except Exception:
            print_traceback()
            return gr.update(), gr.update()

    def update_tools_item(self, selector, tools_json_str):
        """更新选中的工具项"""
        from qwen_agent.gui.gradio_dep import gr
        if selector is None or not selector:
            return gr.update(), gr.update()
        try:
            tools_list = list(self.tools_list)
            index = self._get_tool_index_by_name(selector, tools_list)
            
            # 尝试解析为JSON，如果失败则作为字符串处理
            tool = None
            try:
                tool = json.loads(tools_json_str)
            except json.JSONDecodeError:
                # 如果不是JSON，则作为字符串处理（去掉引号）
                tool = tools_json_str.strip().strip('"').strip("'")
            
            if tool is None:
                raise ValueError('工具配置不能为空')
            
            if 0 <= index < len(tools_list):
                tools_list[index] = tool
                self.tools_list = tools_list
                self._save_tools_list(tools_list)
                choices = self._get_tools_choices(tools_list)
                # 更新后的新名称
                new_name = self._format_tool_name(tool, index)
                # 返回更新后的JSON（pretty格式）
                if isinstance(tool, str):
                    display_value = tool
                else:
                    display_value = json.dumps(tool, ensure_ascii=False, indent=2)
                return gr.update(choices=choices, value=new_name), gr.update(value=display_value)
        except Exception:
            print_traceback()
        return gr.update(), gr.update()

    def delete_tools_item(self, selector):
        """删除选中的工具项"""
        from qwen_agent.gui.gradio_dep import gr
        if selector is None or not selector:
            return gr.update(), gr.update()
        try:
            tools_list = list(self.tools_list)
            index = self._get_tool_index_by_name(selector, tools_list)
            if 0 <= index < len(tools_list):
                tools_list.pop(index)
                self.tools_list = tools_list
                self._save_tools_list(tools_list)
                choices = self._get_tools_choices(tools_list)
                new_value = choices[0] if choices else None
                return gr.update(choices=choices, value=new_value), gr.update(value='')
        except Exception:
            print_traceback()
        return gr.update(), gr.update()

    # === Agent管理相关方法 ===
    def _get_agent_config_choices(self, agent_configs: list) -> list:
        """生成Agent配置下拉选择器的选项列表"""
        return [cfg.get('name', f'Agent {i+1}') for i, cfg in enumerate(agent_configs)]
    
    def _get_agent_config_index_by_name(self, name: str, agent_configs: list) -> int:
        """根据名称查找Agent配置的索引"""
        for i, cfg in enumerate(agent_configs):
            if cfg.get('name', f'Agent {i+1}') == name:
                return i
        return -1
    
    def _get_agent_configs_yaml_path(self) -> str:
        """获取Agent配置持久化文件路径"""
        home = os.path.expanduser('~')
        config_dir = os.path.join(home, '.qwen_agent')
        return os.path.join(config_dir, 'agent_configs.yaml')

    def _load_agent_configs(self) -> list:
        """从YAML文件加载Agent配置列表"""
        yaml_path = self._get_agent_configs_yaml_path()
        if not os.path.exists(yaml_path):
            return []
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or []
            if isinstance(data, list):
                return data
            return []
        except Exception:
            print_traceback()
            return []

    def _save_agent_configs(self, agent_configs: list) -> None:
        """将Agent配置列表持久化到YAML文件"""
        try:
            yaml_path = self._get_agent_configs_yaml_path()
            config_dir = os.path.dirname(yaml_path)
            os.makedirs(config_dir, exist_ok=True)
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(agent_configs or [], f, allow_unicode=True, sort_keys=False)
        except Exception:
            print_traceback()

    def reload_agent_configs(self):
        """重新加载Agent配置列表"""
        from qwen_agent.gui.gradio_dep import gr
        self.agent_cfg_list = self._load_agent_configs()
        choices = self._get_agent_config_choices(self.agent_cfg_list)
        llm_cfg_choices = self._get_llm_cfg_choices(self.llm_cfg_list)
        tools_choices = self._get_tools_choices(self.tools_list)
        return (gr.update(choices=choices, value=None),
                gr.update(value=''),
                gr.update(value=''),
                gr.update(choices=llm_cfg_choices, value=llm_cfg_choices[0][1] if llm_cfg_choices else None),
                gr.update(choices=tools_choices, value=[]))

    def load_agent_config_item(self, selector):
        """加载选中的Agent配置项"""
        from qwen_agent.gui.gradio_dep import gr
        if selector is None or not selector:
            llm_cfg_choices = self._get_llm_cfg_choices(self.llm_cfg_list)
            tools_choices = self._get_tools_choices(self.tools_list)
            return (gr.update(value=''),
                    gr.update(value=''),
                    gr.update(choices=llm_cfg_choices, value=llm_cfg_choices[0][1] if llm_cfg_choices else None),
                    gr.update(choices=tools_choices, value=[]))
        try:
            index = self._get_agent_config_index_by_name(selector, self.agent_cfg_list)
            if 0 <= index < len(self.agent_cfg_list):
                cfg = self.agent_cfg_list[index]
                name = cfg.get('name', '')
                description = cfg.get('description', '')
                # 优先使用llm_cfg_id，兼容旧的llm_cfg_index
                llm_cfg_id = cfg.get('llm_cfg_id')
                if not llm_cfg_id:
                    # 兼容旧格式：从索引转换为ID
                    llm_cfg_index = cfg.get('llm_cfg_index', 0)
                    llm_cfg_list = self.llm_cfg_list
                    if 0 <= llm_cfg_index < len(llm_cfg_list):
                        llm_cfg = llm_cfg_list[llm_cfg_index]
                        if isinstance(llm_cfg, dict):
                            llm_cfg_id = llm_cfg.get('id')
                            # 更新配置为使用ID
                            if llm_cfg_id:
                                cfg['llm_cfg_id'] = llm_cfg_id
                                if 'llm_cfg_index' in cfg:
                                    del cfg['llm_cfg_index']
                                self._save_agent_configs(self.agent_cfg_list)
                
                tools_indices = cfg.get('tools_indices', [])
                
                # 获取LLM配置ID
                llm_cfg_list = self.llm_cfg_list
                llm_cfg_choices = self._get_llm_cfg_choices(llm_cfg_list)
                llm_cfg_value = None
                if llm_cfg_id:
                    # 验证ID是否仍然存在
                    if self._get_llm_cfg_by_id(llm_cfg_id, llm_cfg_list):
                        llm_cfg_value = llm_cfg_id
                    elif llm_cfg_choices:
                        llm_cfg_value = llm_cfg_choices[0][1]
                elif llm_cfg_choices:
                    llm_cfg_value = llm_cfg_choices[0][1]
                
                # 获取工具名称列表
                tools_list = self.tools_list
                tools_choices = self._get_tools_choices(tools_list)
                selected_tools = []
                for idx in tools_indices:
                    if 0 <= idx < len(tools_choices):
                        selected_tools.append(tools_choices[idx])
                
                return (gr.update(value=name),
                        gr.update(value=description),
                        gr.update(choices=llm_cfg_choices, value=llm_cfg_value),
                        gr.update(choices=tools_choices, value=selected_tools))
        except Exception:
            print_traceback()
        llm_cfg_choices = self._get_llm_cfg_choices(self.llm_cfg_list)
        tools_choices = self._get_tools_choices(self.tools_list)
        return (gr.update(value=''),
                gr.update(value=''),
                gr.update(choices=llm_cfg_choices, value=llm_cfg_choices[0][1] if llm_cfg_choices else None),
                gr.update(choices=tools_choices, value=[]))

    def add_agent_config_item(self, name, description, llm_cfg_selector, tools_selector):
        """添加新的Agent配置项"""
        from qwen_agent.gui.gradio_dep import gr
        try:
            name = (name or '').strip()
            if not name:
                raise ValueError('Agent名称不能为空')
            description = (description or '').strip() or "I'm a helpful assistant."
            
            # llm_cfg_selector现在是ID
            llm_cfg_id = llm_cfg_selector
            llm_cfg_list = self.llm_cfg_list
            # 验证ID是否存在
            if not llm_cfg_id or not self._get_llm_cfg_by_id(llm_cfg_id, llm_cfg_list):
                # 如果ID无效，使用第一个配置
                llm_cfg_choices = self._get_llm_cfg_choices(llm_cfg_list)
                if llm_cfg_choices:
                    llm_cfg_id = llm_cfg_choices[0][1]
                else:
                    raise ValueError('没有可用的LLM配置')
            
            # 从工具名称列表获取索引列表
            tools_list = self.tools_list
            tools_indices = []
            if tools_selector:
                for tool_name in tools_selector:
                    idx = self._get_tool_index_by_name(tool_name, tools_list)
                    if idx >= 0:
                        tools_indices.append(idx)
            
            agent_config = {
                'name': name,
                'description': description,
                'llm_cfg_id': llm_cfg_id,
                'tools_indices': tools_indices,
            }
            self.agent_cfg_list.append(agent_config)

            agent = self._create_agent_from_config(agent_config)
            self.agent_list.append(agent)
            self._save_agent_configs(self.agent_cfg_list)
            
            # 刷新agent列表
            selector_update, info_update, plugins_update = self.refresh_agent(len(self.agent_cfg_list) - 1)
            
            choices = self._get_agent_config_choices(self.agent_cfg_list)
            llm_cfg_choices = self._get_llm_cfg_choices(llm_cfg_list)
            tools_choices = self._get_tools_choices(tools_list)
            return (gr.update(choices=choices, value=name),
                    gr.update(value=''),
                    gr.update(value=''),
                    gr.update(choices=llm_cfg_choices, value=llm_cfg_id),
                    gr.update(choices=tools_choices, value=[]),
                    selector_update,
                    info_update,
                    plugins_update)
        except Exception:
            print_traceback()
            llm_cfg_choices = self._get_llm_cfg_choices(self.llm_cfg_list)
            tools_choices = self._get_tools_choices(self.tools_list)
            return (gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(choices=llm_cfg_choices),
                    gr.update(choices=tools_choices),
                    gr.update(),
                    gr.update(),
                    gr.update())

    def update_agent_config_item(self, selector, name, description, llm_cfg_selector, tools_selector):
        """更新选中的Agent配置项"""
        from qwen_agent.gui.gradio_dep import gr
        if selector is None or not selector:
            llm_cfg_choices = self._get_llm_cfg_choices(self.llm_cfg_list)
            tools_choices = self._get_tools_choices(self.tools_list)
            return (gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(choices=llm_cfg_choices),
                    gr.update(choices=tools_choices),
                    gr.update(),
                    gr.update(),
                    gr.update())
        try:
            index = self._get_agent_config_index_by_name(selector, self.agent_cfg_list)
            name = (name or '').strip()
            if not name:
                raise ValueError('Agent名称不能为空')
            description = (description or '').strip() or "I'm a helpful assistant."
            
            # llm_cfg_selector现在是ID
            llm_cfg_id = llm_cfg_selector
            llm_cfg_list = self.llm_cfg_list
            # 验证ID是否存在
            if not llm_cfg_id or not self._get_llm_cfg_by_id(llm_cfg_id, llm_cfg_list):
                # 如果ID无效，使用第一个配置
                llm_cfg_choices = self._get_llm_cfg_choices(llm_cfg_list)
                if llm_cfg_choices:
                    llm_cfg_id = llm_cfg_choices[0][1]
                else:
                    raise ValueError('没有可用的LLM配置')
            
            # 从工具名称列表获取索引列表
            tools_list = self.tools_list
            tools_indices = []
            if tools_selector:
                for tool_name in tools_selector:
                    idx = self._get_tool_index_by_name(tool_name, tools_list)
                    if idx >= 0:
                        tools_indices.append(idx)
            
            if 0 <= index < len(self.agent_cfg_list):
                self.agent_cfg_list[index] = {
                    'name': name,
                    'description': description,
                    'llm_cfg_id': llm_cfg_id,
                    'tools_indices': tools_indices,
                }
                self._save_agent_configs(self.agent_cfg_list)
                
                # 刷新agent列表
                self.agent_list[index] = self._create_agent_from_config(self.agent_cfg_list[index])
                selector_update, info_update, plugins_update = self.refresh_agent(index)
                
                choices = self._get_agent_config_choices(self.agent_cfg_list)
                llm_cfg_choices = self._get_llm_cfg_choices(llm_cfg_list)
                tools_choices = self._get_tools_choices(tools_list)
                return (gr.update(choices=choices, value=name),
                        gr.update(value=name),
                        gr.update(value=description),
                        gr.update(choices=llm_cfg_choices, value=llm_cfg_id),
                        gr.update(choices=tools_choices, value=tools_selector),
                        selector_update,
                        info_update,
                        plugins_update)
        except Exception:
            print_traceback()
        llm_cfg_choices = self._get_llm_cfg_choices(self.llm_cfg_list)
        tools_choices = self._get_tools_choices(self.tools_list)
        return (gr.update(),
                gr.update(),
                gr.update(),
                gr.update(choices=llm_cfg_choices),
                gr.update(choices=tools_choices),
                gr.update(),
                gr.update(),
                gr.update())

    def delete_agent_config_item(self, selector):
        """删除选中的Agent配置项"""
        from qwen_agent.gui.gradio_dep import gr
        if selector is None or not selector:
            llm_cfg_choices = self._get_llm_cfg_choices(self.llm_cfg_list)
            tools_choices = self._get_tools_choices(self.tools_list)
            return (gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(choices=llm_cfg_choices),
                    gr.update(choices=tools_choices),
                    gr.update(),
                    gr.update(),
                    gr.update())
        try:
            self.agent_cfg_list = list(self.agent_cfg_list)
            index = self._get_agent_config_index_by_name(selector, self.agent_cfg_list)
            if 0 <= index < len(self.agent_cfg_list):
                self.agent_cfg_list.pop(index)
                self.agent_cfg_list = self.agent_cfg_list
                self.agent_list.pop(index)
                self._save_agent_configs(self.agent_cfg_list)
                
                # 刷新agent列表
                if len(self.agent_cfg_list) > 0:
                    selector_update, info_update, plugins_update = self.refresh_agent(index - 1)
                else:
                    return (gr.update(),
                            gr.update(),
                            gr.update(),
                            gr.update(choices=llm_cfg_choices),
                            gr.update(choices=tools_choices),
                            gr.update(),
                            gr.update(),
                            gr.update())
                
                choices = self._get_agent_config_choices(self.agent_cfg_list)
                llm_cfg_choices = self._get_llm_cfg_choices(self.llm_cfg_list)
                tools_choices = self._get_tools_choices(self.tools_list)
                new_value = choices[0] if choices else None
                if new_value:
                    # 加载第一个配置项
                    cfg = self.agent_cfg_list[0]
                    name = cfg.get('name', '')
                    description = cfg.get('description', '')
                    llm_cfg_id = cfg.get('llm_cfg_id')
                    if not llm_cfg_id:
                        # 兼容旧格式
                        llm_cfg_index = cfg.get('llm_cfg_index', 0)
                        llm_cfg_list = self.llm_cfg_list
                        if 0 <= llm_cfg_index < len(llm_cfg_list):
                            llm_cfg = llm_cfg_list[llm_cfg_index]
                            if isinstance(llm_cfg, dict):
                                llm_cfg_id = llm_cfg.get('id')
                    tools_indices = cfg.get('tools_indices', [])
                    
                    # 获取LLM配置ID
                    llm_cfg_value = None
                    if llm_cfg_id and self._get_llm_cfg_by_id(llm_cfg_id, self.llm_cfg_list):
                        llm_cfg_value = llm_cfg_id
                    elif llm_cfg_choices:
                        llm_cfg_value = llm_cfg_choices[0][1]
                    
                    # 获取工具名称列表
                    selected_tools = []
                    for idx in tools_indices:
                        if 0 <= idx < len(tools_choices):
                            selected_tools.append(tools_choices[idx])
                    
                    return (gr.update(choices=choices, value=new_value),
                            gr.update(value=name),
                            gr.update(value=description),
                            gr.update(choices=llm_cfg_choices, value=llm_cfg_value),
                            gr.update(choices=tools_choices, value=selected_tools),
                            selector_update,
                            info_update,
                            plugins_update)
                else:
                    return (gr.update(choices=choices, value=None),
                            gr.update(value=''),
                            gr.update(value=''),
                            gr.update(choices=llm_cfg_choices, value=None),
                            gr.update(choices=tools_choices, value=[]),
                            selector_update,
                            info_update,
                            plugins_update)
        except Exception:
            print_traceback()
        llm_cfg_choices = self._get_llm_cfg_choices(self.llm_cfg_list)
        tools_choices = self._get_tools_choices(self.tools_list)
        return (gr.update(),
                gr.update(),
                gr.update(),
                gr.update(choices=llm_cfg_choices),
                gr.update(choices=tools_choices),
                gr.update(),
                gr.update(),
                gr.update())

    def _create_agent_from_config(self, agent_cfg) -> List[Agent]:
        try:
            # agent_cfg格式: {"name": "...", "description": "...", "llm_cfg_id": "...", "tools_indices": [0, 1]}
            name = agent_cfg.get('name', 'Agent')
            description = agent_cfg.get('description', "I'm a helpful assistant.")
            
            # 优先使用llm_cfg_id，兼容旧的llm_cfg_index
            llm_cfg_id = agent_cfg.get('llm_cfg_id')
            llm_cfg = None
            
            if llm_cfg_id:
                # 使用ID查找
                llm_cfg = self._get_llm_cfg_by_id(llm_cfg_id, self.llm_cfg_list)
                if not llm_cfg:
                    logger.warning(f'LLM配置ID {llm_cfg_id} 不存在，使用默认配置')
            else:
                # 兼容旧格式：使用索引
                llm_cfg_index = agent_cfg.get('llm_cfg_index', 0)
                if llm_cfg_index < len(self.llm_cfg_list):
                    llm_cfg = self.llm_cfg_list[llm_cfg_index]
                    # 自动迁移到ID格式
                    if isinstance(llm_cfg, dict) and 'id' in llm_cfg:
                        agent_cfg['llm_cfg_id'] = llm_cfg['id']
                        if 'llm_cfg_index' in agent_cfg:
                            del agent_cfg['llm_cfg_index']
                        self._save_agent_configs(self.agent_cfg_list)
                else:
                    logger.warning(f'LLM配置索引{llm_cfg_index}超出范围，使用默认配置')
            
            if not llm_cfg:
                llm_cfg = {'model': 'qwen-plus', 'model_type': 'qwen_dashscope'}

            # 获取tools
            tools = []
            tools_indices = agent_cfg.get('tools_indices', [])
            for tool_idx in tools_indices:
                if tool_idx < len(self.tools_list):
                    tools.append(self.tools_list[tool_idx])
                else:
                    logger.warning(f'工具索引{tool_idx}超出范围，跳过')

            # 创建Assistant时，排除ID字段
            llm_cfg_for_agent = {k: v for k, v in llm_cfg.items() if k != 'id'}

            # 创建Assistant
            agent = Assistant(
                llm=llm_cfg_for_agent,
                function_list=tools if tools else None,
                name=name,
                description=description,
            )
            return agent
        except Exception:
            print_traceback()
            logger.error(f'创建Agent失败: {agent_cfg}')
            return None

    def refresh_agent(self, index):
        """刷新Agent列表，从配置重新创建"""
        from qwen_agent.gui.gradio_dep import gr

        # 更新agent_selector
        if len(self.agent_list) > 0:
            choices = [(agent.name, i) for i, agent in enumerate(self.agent_list)]
            agent_selector_update = gr.update(
                choices=choices,
                value=0,
                interactive=len(self.agent_list) > 1,
            )
        else:
            agent_selector_update = gr.update(
                choices=[],
                value=None,
                interactive=False,
            )

        return agent_selector_update, self._create_agent_info_block(index), self._create_agent_plugins_block(index)
