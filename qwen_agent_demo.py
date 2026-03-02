# -*- coding: utf-8 -*-
import dashscope
import numpy as np
import faiss
import ast
import re
import streamlit as st
import os
from dashscope import Generation, TextEmbedding
from http import HTTPStatus

# ================================
# 1️⃣ 设置 Qwen API
# ================================
api_key = os.environ.get('DASHSCOPE_API_KEY')
dashscope.api_key = api_key

# ================================
# 2️⃣ RAG 引擎
# ================================
class RagEngine:
    def __init__(self, docs):
        self.documents = docs
        self._build_index()

    def _get_embedding(self, text):
        resp = TextEmbedding.call(model=TextEmbedding.Models.text_embedding_v2, input=text)
        if resp.status_code == HTTPStatus.OK:
            return np.array(resp.output['embeddings'][0]['embedding']).astype("float32")
        raise Exception(f"Embedding 失败: {resp.message}")

    def _build_index(self):
        embeddings = [self._get_embedding(doc) for doc in self.documents]
        dimension = len(embeddings[0])
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.stack(embeddings))

    def search(self, query, top_k=2):
        q_emb = self._get_embedding(query)
        distances, indices = self.index.search(np.array([q_emb]), k=top_k)
        return [self.documents[i] for i in indices[0]]

# ================================
# 3️⃣ 计算工具
# ================================
def robust_calc(text):
    exprs = re.findall(r'(\d+[\s\+\-\*\/\(\)\.]+\d+)', text)
    results = []
    for expr in exprs:
        try:
            node = ast.parse(expr.strip(), mode='eval')
            result = eval(compile(node, '<string>', 'eval'), {"__builtins__": None}, {})
            results.append(f"计算结果 [{expr.strip()}] = {result}")
        except: continue
    return "\n".join(results) if results else ""

# ================================
# 4️⃣ 核心对话逻辑 (含联网搜索)
# ================================
def agent_response(prompt, context_docs):
    # 构建包含本地知识库的系统提示词
    system_prompt = f"你是一个专业的助手。参考资料：\n{chr(10).join(context_docs)}"
    
    responses = Generation.call(
        model="qwen-max",
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt}
        ],
        result_format='message',
        stream=True,
        enable_search=True  # ✅ 开启联网搜索功能
    )
    return responses

# ================================
# 5️⃣ Streamlit 界面
# ================================
st.set_page_config(page_title="Qwen Pro Agent", page_icon="🌐")
st.title("🌐 Qwen 搜索 Agent")

# 初始化
if 'rag' not in st.session_state:
    raw_docs = ["RAG是结合检索和生成的AI技术。", "Agent可以自主调用工具。"]
    with st.spinner("正在初始化本地知识库..."):
        st.session_state.rag = RagEngine(raw_docs)

if "messages" not in st.session_state:
    st.session_state.messages = []

# 渲染历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 用户输入
if query := st.chat_input("请输入问题..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        # 1. 状态显示：搜索与检索
        with st.status("🔍 正在处理...", expanded=False) as status:
            st.write("查阅本地知识库...")
            relevant_docs = st.session_state.rag.search(query)
            st.write("联网搜索中...")
            response_generator = agent_response(query, relevant_docs)
            status.update(label="✅ 准备就绪", state="complete")

        # 2. 创建停止按钮
        # 注意：按钮需要放在一个容器里，方便生成结束后自动消失（可选）
        stop_button = st.button("🛑 停止生成")
        
        msg_placeholder = st.empty()
        full_response = ""

        # 3. 流式生成循环
        for response in response_generator:
            # 检查用户是否点击了停止按钮
            if stop_button:
                st.warning("已停止生成。")
                break 
            
            if response.status_code == HTTPStatus.OK:
                content = response.output.choices[0].message.content
                full_response = content
                msg_placeholder.markdown(full_response + "▌")
            else:
                st.error(f"API 错误: {response.message}")
                break

        # 4. 计算工具补全 (仅在有内容时执行)
        if full_response:
            calc_res = robust_calc(full_response)
            if calc_res:
                full_response += f"\n\n---\n✅ **工具辅助验证：**\n{calc_res}"
            
            msg_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        # 强制刷新页面以移除停止按钮（可选交互优化）
        # st.rerun()
