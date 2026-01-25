import streamlit as st
from openai import OpenAI
import os
import json
import re
from datetime import datetime
from exa_py import Exa
import asyncio
import time
import logging

# 配置基础信息
client = OpenAI(
    base_url='https://api.siliconflow.cn/v1/',
    api_key=os.getenv("SILICONFLOW_API_KEY")
)

# Exa API 配置
exa = Exa(os.getenv("EXA_API_KEY"))

RESEARCH_DIR = "output/DeepResearch"
os.makedirs(RESEARCH_DIR, exist_ok=True)

# 模型配置
RESEARCH_MODEL = "Pro/zai-org/GLM-4.7"
ANALYSIS_MODEL = "Pro/zai-org/GLM-4.7"

# Token 配置参数
RESEARCH_MAX_TOKENS = 16384  # 研究模型的最大token数
ANALYSIS_MAX_TOKENS = 163840  # 分析模型的最大token数

def setup_logging():
    """配置日志系统"""
    # 创建日志目录
    log_dir = os.path.join(RESEARCH_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # 配置日志格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # 设置文件日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"research_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # 控制台输出
        ]
    )

    return logging.getLogger("DeepResearch")

logger = setup_logging()

def init_research_session():
    """初始化研究会话状态"""
    if 'research_query' not in st.session_state:
        st.session_state.research_query = ""
    if 'research_steps' not in st.session_state:
        st.session_state.research_steps = []
    if 'current_research' not in st.session_state:
        st.session_state.current_research = None
    if 'research_depth' not in st.session_state:
        st.session_state.research_depth = 3
    if 'max_sources_per_step' not in st.session_state:
        st.session_state.max_sources_per_step = 5
    if 'research_in_progress' not in st.session_state:
        st.session_state.research_in_progress = False


def extract_json_from_response(content: str, default=None):
    """
    从 LLM 的回复文本中鲁棒地提取 JSON 对象。

    策略：
    1. 尝试直接解析。
    2. 尝试提取 Markdown 代码块 (```json ... ```)。
    3. 尝试暴力查找最外层的 {} 或 [] 结构。

    Args:
        content (str): LLM 返回的原始字符串。
        default (Any, optional): 解析失败时的默认返回值。默认为 None。

    Returns:
        dict | list | None: 解析后的 JSON 对象，失败则返回 default。
    """
    if not content:
        return default

    content = content.strip()

    # --- 策略 1: 直接尝试解析 (最快) ---
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # --- 策略 2: 提取 Markdown 代码块 ---
    # 匹配 ```json ... ``` 或 纯 ``` ... ```
    # re.DOTALL 让 . 可以匹配换行符
    code_block_pattern = r"```(?:json)?\s*(.*?)\s*```"
    match = re.search(code_block_pattern, content, re.DOTALL)

    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # 如果代码块里不是 JSON，继续尝试策略 3
            pass

    # --- 策略 3: 暴力查找 JSON 边界 ---
    # 寻找第一个 '{' 和最后一个 '}' (针对 Object)
    # 或者 第一个 '[' 和 最后一个 ']' (针对 Array)

    # 查找 Object {}
    json_obj_match = re.search(r"(\{.*\})", content, re.DOTALL)
    if json_obj_match:
        try:
            return json.loads(json_obj_match.group(1))
        except json.JSONDecodeError:
            pass

    # 查找 Array []
    json_arr_match = re.search(r"(\[.*\])", content, re.DOTALL)
    if json_arr_match:
        try:
            return json.loads(json_arr_match.group(1))
        except json.JSONDecodeError:
            pass

    # --- 失败 ---
    logging.warning(f"JSON解析失败，原始内容前100字符: {content[:100]}")
    return default


class ResearchStep:
    """研究步骤类"""

    def __init__(self, query, step_type, sources=None, analysis="", reasoning=""):
        self.query = query
        self.step_type = step_type  # 'search', 'analysis', 'synthesis'
        self.sources = sources or []
        self.analysis = analysis
        self.reasoning = reasoning
        self.timestamp = datetime.now().isoformat()

def generate_search_queries(main_query, existing_steps, depth_level):
    """生成深度搜索查询"""
    logger.info(f"生成搜索查询 - 深度级别: {depth_level}, 主要问题: {main_query[:100]}")

    context = ""
    if existing_steps:
        context = f"\n已完成的研究步骤：\n"
        for i, step in enumerate(existing_steps[-3:]):  # 只看最近3步
            context += f"{i + 1}. {step.query} -> {step.analysis[:200]}...\n"
        logger.info(f"使用已有步骤上下文，步骤数: {len(existing_steps[-3:])}")

    prompt = f"""
你是一个专业的研究助手。基于主要研究问题和已有进展，生成{st.session_state.max_sources_per_step}个深入的搜索查询。

主要研究问题：{main_query}
当前深度级别：{depth_level}/3
{context}

请生成{st.session_state.max_sources_per_step}个不同角度的搜索查询，每个查询应该：
1. 针对问题的不同方面
2. 避免重复已搜索的内容
3. 逐步深入细节
4. 包含最新信息和趋势

请以JSON格式返回：
{{"queries": ["查询1", "查询2", "查询3", "查询4", "查询5"]}}
"""

    try:
        logger.info("正在调用AI模型生成搜索查询...")
        response = client.chat.completions.create(
            model=RESEARCH_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=1024
        )

        result = extract_json_from_response(response.choices[0].message.content.strip())
        queries = result.get("queries", [main_query])
        logger.info(f"成功生成 {len(queries)} 个搜索查询: {queries}")
        return queries
    except Exception as e:
        logger.error(f"生成搜索查询失败: {str(e)}")
        return [main_query]


def search_with_exa(query, num_results=5):
    """使用Exa API进行搜索"""
    logger.info(f"开始Exa搜索 - 查询: {query}, 结果数: 10")
    try:
        search_result = exa.search_and_contents(
            query,
            include_domains=["arxiv.org", "nature.com", "science.org", "ieee.org", "acm.org"],
            text=True
        )

        sources = []
        for result in search_result.results:
            sources.append({
                "title": result.title,
                "url": result.url,
                "content": result.text[:2000] if result.text else "",
                "highlights": result.highlights[:3] if result.highlights else [],
                "published_date": getattr(result, 'published_date', None),
                "score": getattr(result, 'score', 0.0)
            })

        logger.info(f"Exa搜索完成，获得 {len(sources)} 个有效来源")
        return sources
    except Exception as e:
        logger.error(f"Exa搜索失败: {str(e)}")
        st.error(f"Exa搜索失败: {str(e)}")
        return []


def analyze_sources(query, sources, existing_context=""):
    """分析搜索结果"""
    logger.info(f"开始分析来源 - 查询: {query}, 来源数: {len(sources)}")
    sources_text = ""
    for i, source in enumerate(sources):
        sources_text += f"\n--- 来源 {i + 1} ---\n"
        sources_text += f"标题: {source['title']}\n"
        sources_text += f"链接: {source['url']}\n"
        sources_text += f"内容: {source['content']}\n"
        if source['highlights']:
            sources_text += f"重点: {'; '.join(source['highlights'])}\n"

    logger.debug(f"准备分析内容长度: {len(sources_text)} 字符")

    prompt = f"""
你是一个专业的研究分析师。请深入分析以下搜索结果，针对查询问题提供详细的分析。

查询问题：{query}

已有研究背景：
{existing_context}

搜索结果：
{sources_text}

请提供：
1. 关键发现和洞察
2. 不同来源间的关联和对比
3. 潜在的研究方向
4. 需要进一步探索的问题
5. 基于证据的结论

请结构化输出，使用markdown格式。
"""

    try:
        logger.info("正在调用AI模型进行来源分析...")
        stream = client.chat.completions.create(
            model=ANALYSIS_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=ANALYSIS_MAX_TOKENS,
            stream=True
        )

        full_analysis = ""
        full_reasoning = ""

        for chunk in stream:
            if chunk.choices[0].delta.content:
                full_analysis += chunk.choices[0].delta.content or ""

            if hasattr(chunk.choices[0].delta, 'reasoning_content'):
                reasoning = chunk.choices[0].delta.reasoning_content or ""
                full_reasoning += reasoning

        logger.info(f"分析完成 - 生成内容长度: {len(full_analysis)} 字符。")
        return full_analysis, full_reasoning
    except Exception as e:
        logger.error(f"来源分析失败: {str(e)}")
        return f"分析失败: {str(e)}", ""


def analyze_sources_streaming(query, sources, existing_context="", placeholder=None):
    """分析搜索结果 - 支持流式显示"""
    logger.info(f"开始分析来源 - 查询: {query}, 来源数: {len(sources)}")
    sources_text = ""
    for i, source in enumerate(sources):
        sources_text += f"\n--- 来源 {i + 1} ---\n"
        sources_text += f"标题: {source['title']}\n"
        sources_text += f"链接: {source['url']}\n"
        sources_text += f"内容: {source['content']}\n"
        if source['highlights']:
            sources_text += f"重点: {'; '.join(source['highlights'])}\n"

    prompt = f"""
你是一个专业的研究分析师。请深入分析以下搜索结果，针对查询问题提供详细的分析。

查询问题：{query}

已有研究背景：
{existing_context}

搜索结果：
{sources_text}

请提供：
1. 关键发现和洞察
2. 不同来源间的关联和对比
3. 潜在的研究方向
4. 需要进一步探索的问题
5. 基于证据的结论

请结构化输出，使用markdown格式。
"""

    try:
        logger.info("正在调用AI模型进行来源分析...")
        stream = client.chat.completions.create(
            model=ANALYSIS_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=ANALYSIS_MAX_TOKENS,
            stream=True
        )

        full_analysis = ""
        full_reasoning = ""

        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content or ""
                full_analysis += content

                # 实时更新显示
                if placeholder:
                    placeholder.markdown(full_analysis + "▌")

            if hasattr(chunk.choices[0].delta, 'reasoning_content'):
                reasoning = chunk.choices[0].delta.reasoning_content or ""
                full_reasoning += reasoning

        # 移除光标
        if placeholder:
            placeholder.markdown(full_analysis)

        logger.info(f"分析完成 - 生成内容长度: {len(full_analysis)} 字符。")
        return full_analysis, full_reasoning
    except Exception as e:
        logger.error(f"来源分析失败: {str(e)}")
        return f"分析失败: {str(e)}", ""


def synthesize_research(main_query, all_steps, placeholder=None):
    """综合所有研究结果"""
    logger.info(f"开始综合研究 - 主要问题: {main_query}, 总步骤数: {len(all_steps)}")
    research_summary = ""
    for step in all_steps:
        research_summary += f"\n=== {step.query} ===\n"
        research_summary += f"类型: {step.step_type}\n"
        research_summary += f"分析: {step.analysis}\n"
        research_summary += f"来源数量: {len(step.sources)}\n\n"

    logger.info(f"研究摘要准备完成，总长度: {len(research_summary)} 字符")

    prompt = f"""
你是一个顶级研究专家。��基于以下完整的深度研究结果，为主要研究问题提供comprehensive final report。

主要研究问题：{main_query}

完整研究过程：
{research_summary}

请提供一份专业的研究报告，包括：

## 执行摘要
- 核心发现
- 主要结论

## 详细分析
- 关键洞察
- 趋势分析
- 技术细节

## 实践建议
- 可行的解决方案
- 实施建议
- 潜在风险

## 进一步研究方向
- 未解决的问题
- 研究空白
- 未来机会

## 参考文献总结
- 关键文献分类
- 可信度评估

请使用专业的学术语言，确保逻辑清晰、结��完整。
"""

    try:
        logger.info("正在生成最终综合报告...")
        stream = client.chat.completions.create(
            model=ANALYSIS_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=ANALYSIS_MAX_TOKENS,
            stream=True
        )

        synthesis = ""
        reasoning = ""

        for chunk in stream:
            if chunk.choices[0].delta.content:
                synthesis += chunk.choices[0].delta.content or ""

                # 实时更新显示
                if placeholder:
                    placeholder.markdown(synthesis + "▌")

            if hasattr(chunk.choices[0].delta, 'reasoning_content'):
                reasoning += chunk.choices[0].delta.reasoning_content or ""

        # 移除光标
        if placeholder:
            placeholder.markdown(synthesis)

        logger.info(f"综合报告生成完成 - 长度: {len(synthesis)} 字符。")
        return synthesis, reasoning
    except Exception as e:
        return f"综合分析失败: {str(e)}", ""


def synthesize_research_streaming(main_query, all_steps, placeholder=None):
    """综合所有研究结果 - 支持流式显示"""
    logger.info(f"开始综合研究 - 主要问题: {main_query}, 总步骤数: {len(all_steps)}")
    research_summary = ""
    for step in all_steps:
        research_summary += f"\n=== {step.query} ===\n"
        research_summary += f"类型: {step.step_type}\n"
        research_summary += f"分析: {step.analysis}\n"
        research_summary += f"来源数量: {len(step.sources)}\n\n"

    prompt = f"""
你是一个顶级研究专家。请基于以下完整的深度研究结果，为主要研究问题提供comprehensive final report。

主要研究问题：{main_query}

完整研究过程：
{research_summary}

请提供一份专业的研究报告，包括：

## 执行摘要
- 核心发现
- 主要结论

## 详细分析
- 关键洞察
- 趋势分析
- 技术细节

## 实践建议
- 可行的解决方案
- 实施建议
- 潜在风险

## 进一步研究方向
- 未解决的问题
- 研究空白
- 未来机会

## 参考文献总结
- 关键文献分类
- 可信度评估

请使用专业的学术语言，确保逻辑清晰、结构完整。
"""

    try:
        logger.info("正在生成最终综合报告...")
        stream = client.chat.completions.create(
            model=ANALYSIS_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=ANALYSIS_MAX_TOKENS,
            stream=True
        )

        synthesis = ""
        reasoning = ""

        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content or ""
                synthesis += content

                # 实时更新显示
                if placeholder:
                    placeholder.markdown(synthesis + "▌")

            if hasattr(chunk.choices[0].delta, 'reasoning_content'):
                reasoning += chunk.choices[0].delta.reasoning_content or ""

        # 移除光标
        if placeholder:
            placeholder.markdown(synthesis)

        logger.info(f"综合报告生成完成 - 长度: {len(synthesis)} 字符。")
        return synthesis, reasoning
    except Exception as e:
        return f"综合分析失败: {str(e)}", ""


def save_research(filename, query, steps):
    """保存研究结果"""
    logger.info(f"保存研究结果到文件: {filename}")
    research_data = {
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "steps": []
    }

    for step in steps:
        research_data["steps"].append({
            "query": step.query,
            "step_type": step.step_type,
            "analysis": step.analysis,
            "reasoning": step.reasoning,
            "sources": step.sources,
            "timestamp": step.timestamp
        })

    filepath = os.path.join(RESEARCH_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(research_data, f, ensure_ascii=False, indent=2)


def load_research(filename):
    """加载研究结果"""
    filepath = os.path.join(RESEARCH_DIR, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    steps = []
    for step_data in data["steps"]:
        step = ResearchStep(
            query=step_data["query"],
            step_type=step_data["step_type"],
            sources=step_data.get("sources", []),
            analysis=step_data.get("analysis", ""),
            reasoning=step_data.get("reasoning", "")
        )
        steps.append(step)

    return data["query"], steps


# Streamlit 界面
st.set_page_config(
    page_title="DeepResearch - 深度研究助手",
    page_icon="🔬",
    layout="wide"
)

init_research_session()

# 侧边栏配置
with st.sidebar:
    st.title("🔬 DeepResearch")
    st.write("基于AI和Exa API的深度研究工具")

    st.subheader("研究配置")
    st.session_state.research_depth = st.selectbox(
        "研究深度",
        [1, 2, 3, 4, 5],
        index=2,
        help="每个深度级别会进行更深入的分析"
    )

    st.session_state.max_sources_per_step = st.selectbox(
        "每步最大来源数",
        [3, 5, 7, 10],
        index=1,
        help="每个研究步骤搜索的最大来源数量"
    )

    st.subheader("历史研究")
    research_files = [f for f in os.listdir(RESEARCH_DIR) if f.endswith('.json')]
    research_files.sort(reverse=True)

    for filename in research_files[:10]:
        if st.button(f"📄 {filename[:-5]}", key=f"load_{filename}"):
            query, steps = load_research(filename)
            st.session_state.research_query = query
            st.session_state.research_steps = steps
            st.session_state.current_research = filename
            st.rerun()

# 主界面
st.title("🔬 DeepResearch - 深度研究助手")
st.write("输入您的研究问题，我将进行深入的多步骤研究分析")

# 研究查询输入
research_query = st.text_area(
    "🎯 研究问题",
    value=st.session_state.research_query,
    height=100,
    placeholder="例如：大语言模型在科学研究中的应用现状和发展趋势是什么？"
)

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    if st.button("🚀 开始研究", disabled=st.session_state.research_in_progress):
        if research_query.strip():
            st.session_state.research_query = research_query
            st.session_state.research_steps = []
            st.session_state.research_in_progress = True
            st.rerun()

with col2:
    if st.button("🔄 新研究"):
        st.session_state.research_query = ""
        st.session_state.research_steps = []
        st.session_state.current_research = None
        st.session_state.research_in_progress = False
        st.rerun()

# 在执行深度研究的部分，修改进度计算逻辑
if st.session_state.research_in_progress and st.session_state.research_query:
    logger.info("=" * 50)
    logger.info(f"开始新的研究会话")
    logger.info(f"研究问题: {st.session_state.research_query}")
    logger.info(f"研究深度: {st.session_state.research_depth}")
    logger.info(f"每步最大来源数: {st.session_state.max_sources_per_step}")
    logger.info("=" * 50)

    # 创建实时显示区域
    progress_container = st.container()
    process_container = st.container()

    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        current_time = st.empty()

    with process_container:
        st.subheader("🔄 研究进行中...")
        step_containers = []

    try:
        # 修复进度计算：为每个深度级别计算查询数量
        queries_per_depth = st.session_state.max_sources_per_step
        total_queries = st.session_state.research_depth * queries_per_depth
        total_steps = total_queries + 1  # +1 for synthesis
        current_step = 0
        existing_context = ""
        start_time = time.time()

        logger.info(
            f"总步骤数计算: {st.session_state.research_depth} 深度 × {queries_per_depth} 查询 + 1 综合 = {total_steps}")

        # 多轮深度研究
        for depth in range(st.session_state.research_depth):
            logger.info(f"开始第 {depth + 1} 轮研究 (深度级别: {depth + 1}/{st.session_state.research_depth})")

            # 显示当前深度级别
            depth_container = process_container.container()
            with depth_container:
                st.markdown(f"### 🔍 第 {depth + 1} 轮深度研究")
                depth_status = st.empty()
                depth_time = st.empty()

            step_start_time = time.time()

            # 更新状态和时间
            elapsed_time = time.time() - start_time
            status_text.text(f"🔍 第 {depth + 1} 轮搜索中...")
            current_time.text(f"⏱️ 已用时: {elapsed_time:.1f}秒")
            depth_status.text("🔍 生成搜索查询中...")

            # 生成搜索查询
            queries = generate_search_queries(
                st.session_state.research_query,
                st.session_state.research_steps,
                depth + 1
            )

            # 显示生成的查询
            with depth_container:
                st.write("**📝 生成的搜索查询:**")
                for i, query in enumerate(queries):
                    st.write(f"{i + 1}. {query}")

            for query_idx, query in enumerate(queries):
                step_start = time.time()

                # 创建步骤容器
                query_container = depth_container.container()
                with query_container:
                    st.markdown(f"#### 查询 {query_idx + 1}: {query}")
                    search_status = st.empty()
                    search_results = st.empty()
                    analysis_container = st.container()

                search_status.text("🔍 搜索中...")

                # 搜索
                sources = search_with_exa(query, st.session_state.max_sources_per_step)

                # 显示搜索结果
                with search_results:
                    if sources:
                        st.success(f"✅ 找到 {len(sources)} 个相关来源")
                        with st.expander("查看搜索结果"):
                            for i, source in enumerate(sources):
                                st.write(f"**{i + 1}.** [{source['title']}]({source['url']})")
                    else:
                        st.warning("⚠️ 未找到相关来源")

                # 分析
                search_status.text("📊 分析中...")

                with analysis_container:
                    st.write("**📊 分析结果:**")
                    analysis_placeholder = st.empty()

                analysis, reasoning = analyze_sources_streaming(
                    query, sources, existing_context, analysis_placeholder
                )

                # 更新进度 - 确保不超过1.0
                current_step += 1
                progress_value = min(current_step / total_steps, 0.95)  # 最大95%，为综合分析留空间
                progress_bar.progress(progress_value)

                logger.info(f"进度更新: 步骤 {current_step}/{total_steps}, 进度值: {progress_value:.3f}")

                elapsed_time = time.time() - start_time
                step_elapsed = time.time() - step_start
                current_time.text(f"⏱️ 已用时: {elapsed_time:.1f}秒")
                search_status.text(f"✅ 完成 (用时: {step_elapsed:.1f}秒)")

                # 创建研究步骤
                step = ResearchStep(
                    query=query,
                    step_type="search_analysis",
                    sources=sources,
                    analysis=analysis,
                    reasoning=reasoning
                )

                st.session_state.research_steps.append(step)
                existing_context += f"\n{query}: {analysis[:500]}...\n"

            # 完成该深度级别
            depth_elapsed = time.time() - step_start_time
            depth_time.text(f"⏱️ 本轮用时: {depth_elapsed:.1f}秒")

        # 最终综合
        progress_bar.progress(0.96)  # 设置为96%
        status_text.text("🎯 生成综合报告中...")

        with process_container:
            st.markdown("### 🎯 最终综合分析")
            synthesis_status = st.empty()
            synthesis_placeholder = st.empty()

        synthesis_status.text("🎯 综合分析中...")

        final_synthesis, final_reasoning = synthesize_research_streaming(
            st.session_state.research_query,
            st.session_state.research_steps,
            synthesis_placeholder
        )

        synthesis_step = ResearchStep(
            query="最终综合分析",
            step_type="synthesis",
            sources=[],
            analysis=final_synthesis,
            reasoning=final_reasoning
        )
        st.session_state.research_steps.append(synthesis_step)

        # 保存研究
        timestamp = datetime.now().strftime("%m%d_%H%M")
        filename = f"{timestamp}_research.json"
        save_research(filename, st.session_state.research_query, st.session_state.research_steps)
        st.session_state.current_research = filename

        # 完成
        total_time = time.time() - start_time
        progress_bar.progress(1.0)  # 最终设置为100%
        status_text.text("✅ 研究完成！")
        current_time.text(f"⏱️ 总用时: {total_time:.1f}秒")
        synthesis_status.text(f"✅ 综合分析完成")

        logger.info("=" * 50)
        logger.info("研究会话完成")
        logger.info(f"总步骤数: {len(st.session_state.research_steps)}")
        logger.info(f"保存文件: {filename}")
        logger.info("=" * 50)

        st.session_state.research_in_progress = False

        # 显示完成消息
        st.success(f"🎉 研究完成！总共用时 {total_time:.1f} 秒，已保存为 {filename}")
        time.sleep(2)
        st.rerun()

    except Exception as e:
        logger.error(f"研究过程中出错: {str(e)}", exc_info=True)
        st.error(f"研究过程中出错: {str(e)}")
        st.session_state.research_in_progress = False

# 显示研究结果
if st.session_state.research_steps:
    st.subheader("🎯 研究问题")
    st.write(st.session_state.research_query)

    # 显示最终综合（如果存在）
    synthesis_steps = [step for step in st.session_state.research_steps if step.step_type == "synthesis"]
    if synthesis_steps:
        st.subheader("📋 综合研究报告")
        synthesis = synthesis_steps[-1]

        if synthesis.reasoning:
            with st.expander("🧠 推理过程"):
                st.markdown(synthesis.reasoning)

        st.markdown(synthesis.analysis)

    # 显示详细研究步骤
    st.subheader("🔍 详细研究过程")

    search_steps = [step for step in st.session_state.research_steps if step.step_type == "search_analysis"]

    for i, step in enumerate(search_steps):
        with st.expander(f"步骤 {i + 1}: {step.query}"):
            col1, col2 = st.columns([2, 1])

            with col1:
                if step.reasoning:
                    st.write("**🧠 推理过程:**")
                    st.markdown(step.reasoning[:500] + "..." if len(step.reasoning) > 500 else step.reasoning)

                st.write("**📊 分析结果:**")
                st.markdown(step.analysis)

            with col2:
                st.write("**📚 参考来源:**")
                for j, source in enumerate(step.sources):
                    st.write(f"**来源 {j + 1}:** [{source['title']}]({source['url']})")
                    if source.get('score'):
                        st.write(f"相关度: {source['score']:.2f}")
                    st.write("---")

else:
    st.info("👆 请输入您的研究问题并点击'开始研究'来开始深度研究")

# 状态显示
if st.session_state.current_research:
    st.success(f"✅ 当前研究已保存为: {st.session_state.current_research}")