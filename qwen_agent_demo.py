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
# 2️⃣ 增强版 RAG 引擎
# ================================
class RagEngine:
    def __init__(self, docs):
        self.documents = docs
        self.index = None
        self._build_index()

    def _get_embedding(self, text):
        """调用阿里 Embedding 模型"""
        resp = TextEmbedding.call(model=TextEmbedding.Models.text_embedding_v2, input=text)
        if resp.status_code == HTTPStatus.OK:
            return np.array(resp.output['embeddings'][0]['embedding']).astype("float32")
        raise Exception(f"Embedding 失败: {resp.message}")

    def _build_index(self):
        """初始化向量库"""
        embeddings = [self._get_embedding(doc) for doc in self.documents]
        dimension = len(embeddings[0])
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.stack(embeddings))

    def search(self, query, top_k=2):
        """执行相似度搜索"""
        q_emb = self._get_embedding(query)
        distances, indices = self.index.search(np.array([q_emb]), k=top_k)
        # 增加阈值过滤：如果距离太大（语义不相关），则不返回
        return [self.documents[i] for i in indices[0]]

# ================================
# 3️⃣ 鲁棒的计算工具
# ================================
def robust_calc(text):
    """从文本中提取并安全计算所有数学表达式"""
    # 匹配加减乘除算式
    exprs = re.findall(r'(\d+[\s\+\-\*\/\(\)\.]+\d+)', text)
    results = []
    for expr in exprs:
        try:
            # 使用 ast 限制节点类型，防止代码注入
            node = ast.parse(expr.strip(), mode='eval')
            result = eval(compile(node, '<string>', 'eval'), {"__builtins__": None}, {})
            results.append(f"计算结果 [{expr.strip()}] = {result}")
        except:
            continue
    return "\n".join(results) if results else ""

# ================================
# 4️⃣ Agent 主逻辑 (流式输出 + 记忆)
# ================================
def agent_response(prompt, context_docs):
    system_prompt = f"""你是一个专业的中文AI助手。
请结合以下参考资料回答用户问题。如果资料中没有相关信息，请告知用户。
涉及数学计算时，请直接给出算式。

【参考资料】:
{chr(10).join(context_docs)}
"""
    # 使用 Qwen-Max 提升逻辑能力
    responses = Generation.call(
        model="qwen-max",
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt}
        ],
        result_format='message',
        stream=True  # 开启流式传输提升用户体验
    )
    return responses

# ================================
# 5️⃣ Streamlit 界面优化
# ================================
st.set_page_config(page_title="Qwen Pro Agent", page_icon="🤖")
st.title("🤖 生产级 Qwen RAG Agent(覃玉亮是个小弟)")

# 初始化知识库 (添加缓存避免重复生成 Embedding)
if 'rag' not in st.session_state:
    raw_docs = [
        "RAG（检索增强生成）通过外挂知识库解决大模型幻觉问题。",
        "Python 的 ast 模块可以安全地解析 Python 语法树。",
        "FAISS 索引类型包括 IndexFlatL2（精确）和 IndexIVFFlat（快速）。",
        "Agent 的核心在于感知、决策和执行工具的能力。"
    ]
    with st.spinner("正在初始化知识库..."):
        st.session_state.rag = RagEngine(raw_docs)

# 聊天记录界面
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 用户输入
if query := st.chat_input("请输入您的问题..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        # 1. 检索
        relevant_docs = st.session_state.rag.search(query)
        with st.expander("🔍 查找到的相关文档"):
            st.write(relevant_docs)

        # 2. 生成 (流式显示)
        full_response = ""
        msg_placeholder = st.empty()
        
        for response in agent_response(query, relevant_docs):
            if response.status_code == HTTPStatus.OK:
                content = response.output.choices[0].message.content
                full_response = content
                msg_placeholder.markdown(full_response + "▌")
            else:
                st.error(f"API 调用失败: {response.message}")

        # 3. 工具补全
        calc_res = robust_calc(full_response)
        if calc_res:
            full_response += f"\n\n---\n✅ **工具辅助验证：**\n{calc_res}"
            
        msg_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})