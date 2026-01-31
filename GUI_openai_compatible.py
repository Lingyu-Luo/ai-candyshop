import streamlit as st
from openai import OpenAI
import os
import json
import base64
from datetime import datetime
import re

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
HISTORY_DIR = "ChatHistory"
os.makedirs(HISTORY_DIR, exist_ok=True)


def init_session():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'current_convo' not in st.session_state:
        st.session_state.current_convo = None
    if 'convo_list' not in st.session_state:
        st.session_state.convo_list = []
    if 'num_convo_display' not in st.session_state:
        st.session_state.num_convo_display = 10

def generate_filename(content):
    text_content = ""
    if isinstance(content, list):
        texts = [item["text"] for item in content if isinstance(item, dict) and item.get("type") == "text"]
        text_content = " ".join(texts)
    else:
        text_content = str(content)
    
    temp_messages=[
            {"role": "system", "content": "你是一个对话命名助手，帮助提取对话关键词作为对话记录文件名，十五字以内。"},
            {"role": "user", "content": "提取对话的主题（仅输出主题本身）：" + text_content}
        ]
    response = client.chat.completions.create(
        model="gemini-3-flash-preview",
        #reasoning_effort="low",
        messages=temp_messages
    )
    
    clean_content = response.choices[0].message.content.strip().replace("\n", " ")
    clean_content = re.sub(r'[\n\r\t\\/*?:"<>|]', "", clean_content)[:15]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{clean_content}.json" if clean_content else f"{timestamp}_未命名.json"

def refresh_convo_list():
    st.session_state.convo_list = [
        f for f in os.listdir(HISTORY_DIR)
        if f.endswith('.json') and os.path.getsize(os.path.join(HISTORY_DIR, f)) > 0
    ]
    st.session_state.convo_list.reverse()
    
def new_conversation():
    st.session_state.messages = []
    st.session_state.current_convo = None


def load_conversation(filename):
    path = os.path.join(HISTORY_DIR, filename)
    with open(path, 'r', encoding='utf-8') as f:
        st.session_state.messages = json.load(f)
    st.session_state.current_convo = filename

def save_conversation():
    if st.session_state.current_convo and st.session_state.messages:
        path = os.path.join(HISTORY_DIR, st.session_state.current_convo)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(st.session_state.messages, f, ensure_ascii=False, indent=2)

def render_with_latex(text: str):
    text = text.replace(r'\\\\',r"\\")
    text = text.replace(r'\(',r"$")
    text = text.replace(r'\)', r"$")
    text = text.replace(r'\[', r"$$")
    text = text.replace(r'\]', r"$$")
    st.markdown(text)

init_session()

with st.sidebar:
    st.title("对话管理")
    if st.button("➕ 新建对话", width='stretch'):
        st.session_state.messages = []
        st.session_state.current_convo = None
        st.rerun()
    refresh_convo_list()
    convo_render_list = st.session_state.convo_list[:st.session_state.num_convo_display]
    for convo in convo_render_list:
        cols = st.columns([3, 1])
        with cols[0]:
            if st.button(convo[:-5], key=f"btn_{convo}", width='stretch'):
                load_conversation(convo)
                st.rerun()
        with cols[1]:
            if st.button("×", key=f"del_{convo}", type='primary'):
                os.remove(os.path.join(HISTORY_DIR, convo))
                if st.session_state.current_convo == convo:
                    new_conversation()
                st.rerun()
    if st.session_state.num_convo_display < len(st.session_state.convo_list):
        if st.button("加载更多...",key="load_more_convo"):
            st.session_state.num_convo_display += 10
            st.rerun()


st.title("智能对话助手（支持图文）")
for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        # 先显示推理内容（如果有）
        if msg["role"] == "assistant" and msg.get("reasoning"):
            with st.expander("🧠 推理过程（点击展开）"):
                render_with_latex(msg["reasoning"])
        
        if isinstance(msg["content"], list):
            for item in msg["content"]:
                if item["type"] == "image_url":
                    try:
                        base64_str = item["image_url"]["url"].split(",")[1]
                        st.image(base64.b64decode(base64_str), width='stretch')
                    except:
                        st.error("图片加载失败")
                elif item["type"] == "input_audio":
                    try:
                        audio_data = base64.b64decode(item["input_audio"]["data"])
                        st.audio(audio_data, format=f"audio/{item['input_audio']['format']}")
                    except:
                        st.error("音频加载失败")
                elif item["type"] == "text" and item["text"].strip():
                    render_with_latex(item["text"])
        else:
            render_with_latex(msg["content"])


uploaded_files = st.file_uploader(
    "📤 上传图片或音频（支持多选）",
    type=["png", "jpg", "jpeg", "wav", "mp3", "ogg", "flac", "aac", "m4a"],
    accept_multiple_files=True,
    key="file_uploader"
)

if prompt := st.chat_input("请输入您的问题或描述..."):
    # 构建多模态消息内容
    message_content = []

    for uploaded_file in uploaded_files:
        if uploaded_file:
            base64_str = base64.b64encode(uploaded_file.read()).decode("utf-8")
            mime_type = uploaded_file.type
            
            # 判断是图片还是音频
            if mime_type.startswith("image/"):
                message_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_str}"
                    }
                })
            elif mime_type.startswith("audio/"):
                # 获取音频格式
                audio_format = uploaded_file.name.split(".")[-1].lower()
                message_content.append({
                    "type": "input_audio",
                    "input_audio": {
                        "data": base64_str,
                        "format": audio_format
                    }
                })
            uploaded_file.seek(0)  # 重置文件指针

    # 处理文本输入
    if prompt.strip():
        message_content.append({
            "type": "text",
            "text": prompt.strip()
        })
    
    user_message = {
        "role": "user",
        "content": message_content if len(message_content) > 1 else prompt
    }
    st.session_state.messages.append(user_message)
    
    with st.chat_message("user", avatar="🧑"):
        for item in message_content:
            if item["type"] == "image_url":
                try:
                    base64_str = item["image_url"]["url"].split(",")[1]
                    st.image(base64.b64decode(base64_str), width='stretch')
                except:
                    st.error("图片显示失败")
            elif item["type"] == "input_audio":
                try:
                    audio_data = base64.b64decode(item["input_audio"]["data"])
                    st.audio(audio_data, format=f"audio/{item['input_audio']['format']}")
                except:
                    st.error("音频显示失败")
            elif item["type"] == "text":
                render_with_latex(item["text"])
    
    try:
        with ((st.chat_message("assistant", avatar="🤖️"))):
            reasoning_placeholder = st.empty()
            answer_placeholder = st.empty()
            full_reasoning = ""
            full_answer = ""
            
            response = client.chat.completions.create(
                model="gemini-3-flash-preview",
                #reasoning_effort="high",
                messages=st.session_state.messages,
                max_tokens=163840,
                stream=True
            )
            
            for chunk in response:
                print(chunk.model_dump())
                delta = chunk.choices[0].delta
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    full_reasoning += delta.reasoning_content
                    reasoning_placeholder.markdown("🧠 推理过程（点击展开）")
                    with st.expander("🧠 推理过程（点击展开）"):
                        render_with_latex(full_reasoning)
                if delta.content:
                    full_answer += delta.content
                    answer_placeholder.markdown(full_answer)
            
            with reasoning_placeholder:
                if full_reasoning.strip():
                    with st.expander("🧠 推理过程"):
                        render_with_latex(full_reasoning.strip())
            with answer_placeholder:
                render_with_latex(full_answer)
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_answer,
                "reasoning": full_reasoning.strip()
            })
    
    except Exception as e:
        st.error(f"请求失败: {str(e)}")
        st.session_state.messages.append({
            "role": "assistant",
            "content": "响应生成失败",
            "reasoning": f"错误信息: {str(e)}"
        })
    
    filename_content = prompt.strip()
    if not st.session_state.current_convo:
        print("正在生成对话文件名...")
        st.session_state.current_convo = generate_filename(filename_content)
    # 保存对话记录
    if st.session_state.current_convo:
        save_conversation()
        refresh_convo_list()
    print("\n")
    st.rerun()

st.markdown("""
<script>
// 自动滚动到底部
window.addEventListener('DOMContentLoaded', () => {
    const scrollToBottom = () => {
        window.scrollTo(0, document.body.scrollHeight);
    };
    scrollToBottom();
});
</script>
""", unsafe_allow_html=True)