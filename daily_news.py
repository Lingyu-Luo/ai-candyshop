import os
import json
import requests
import feedparser
import datetime

# 请确保环境变量中有 SILICONFLOW_API_KEY，或者直接填在这里
API_KEY = os.getenv("SILICONFLOW_API_KEY")
BASE_URL = "https://api.siliconflow.cn/v1/chat/completions"
MODEL_NAME = "deepseek-ai/DeepSeek-V3.2"
KEYWORDS = ["LLM", "Transformer", "GPT", "Claude", "Gemini", "DeepSeek",
            "RAG", "Agent", "Diffusion", "Quantization", "MoE"]


def fetch_hackernews():
    """抓取 Hacker News 前 50 条中的 AI 相关新闻"""
    print("📡 正在抓取 Hacker News...")
    try:
        top_ids = requests.get("https://hacker-news.firebaseio.com/v0/topstories.json").json()[:50]
        news_list = []
        for pid in top_ids:
            item = requests.get(f"https://hacker-news.firebaseio.com/v0/item/{pid}.json").json()
            if not item or 'title' not in item: continue

            # 简单的关键词匹配
            if any(k.lower() in item['title'].lower() for k in KEYWORDS):
                news_list.append({
                    "source": "Hacker News",
                    "title": item['title'],
                    "url": item.get('url', f"https://news.ycombinator.com/item?id={pid}"),
                    "desc": f"Score: {item.get('score', 0)}"
                })
        return news_list
    except Exception as e:
        print(f"❌ Hacker News 抓取失败: {e}")
        return []


def fetch_arxiv():
    """抓取 ArXiv (cs.CL 计算语言学) 最新论文"""
    print("📡 正在抓取 ArXiv (cs.CL)...")
    try:
        url = 'http://export.arxiv.org/api/query?search_query=cat:cs.CL&start=0&max_results=10&sortBy=submittedDate&sortOrder=descending'
        feed = feedparser.parse(url)
        papers = []
        for entry in feed.entries:
            # 仅保留摘要中包含关键词的论文，或者无条件保留前5篇
            if any(k.lower() in entry.title.lower() for k in KEYWORDS):
                papers.append({
                    "source": "ArXiv",
                    "title": entry.title.replace('\n', ' '),
                    "url": entry.link,
                    "desc": entry.summary[:150] + "..."  # 只取摘要前150字
                })
        return papers
    except Exception as e:
        print(f"❌ ArXiv 抓取失败: {e}")
        return []


def fetch_huggingface_daily():
    """抓取 Hugging Face Daily Papers (热门论文)"""
    print("📡 正在抓取 Hugging Face Daily Papers...")
    try:
        # 使用 Hugging Face 的公开 API
        resp = requests.get("https://huggingface.co/api/daily_papers")
        if resp.status_code != 200:
            return []

        data = resp.json()
        papers = []
        # 获取今天的热门论文（API返回的是列表）
        for item in data[:8]:  # 取前8篇
            paper = item['paper']
            papers.append({
                "source": "Hugging Face",
                "title": paper['title'],
                "url": f"https://huggingface.co/papers/{paper['id']}",
                "desc": f"Votes: {item.get('numComments', 0)} | {paper['summary'][:100] if 'summary' in paper else 'No summary'}"
            })
        return papers
    except Exception as e:
        print(f"❌ Hugging Face 抓取失败: {e}")
        return []


def chat_with_llm_stream(model: str, prompt: str):
    """
    调用 SiliconFlow API 进行总结 (基于你提供的代码修改)
    返回完整的总结文本字符串。
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    messages = [{"role": "user", "content": prompt}]

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 163840,
        "temperature": 0.5,
        "enable_thinking": True,
        "stream": True
    }

    print(f"\n🧠 正在调用 {model} 进行深度总结...\n")
    print("-" * 40)

    try:
        response = requests.post(BASE_URL, json=payload, headers=headers, stream=True)
        response.raise_for_status()

        full_content = ""
        full_reasoning = ""

        for chunk in response.iter_lines():
            if chunk:
                chunk_str = chunk.decode('utf-8').replace('data: ', '')
                if chunk_str == "[DONE]": break

                try:
                    chunk_data = json.loads(chunk_str)
                    delta = chunk_data.get('choices', [{}])[0].get('delta', {})

                    # 处理思维链 (如果你用 R1)
                    if 'reasoning_content' in delta and delta['reasoning_content']:
                        print(delta['reasoning_content'], end="", flush=True)  # 打印思考过程
                        full_reasoning += delta['reasoning_content']

                    # 处理正文
                    if 'content' in delta and delta['content']:
                        content = delta['content']
                        print(content, end="", flush=True)  # 实时打印正文
                        full_content += content

                except json.JSONDecodeError:
                    continue

        print("\n" + "-" * 40 + "\n")
        return full_content

    except Exception as e:
        print(f"\n❌ API 调用出错: {e}")
        return "AI 总结失败，请检查 API Key 或网络连接。"


def generate_daily_report():
    # 1. 获取数据
    news_items = []
    news_items.extend(fetch_huggingface_daily())  # 优先看 HF 论文
    news_items.extend(fetch_hackernews())
    news_items.extend(fetch_arxiv())

    if not news_items:
        print("😴 今天好像没什么大新闻。")
        return

    # 2. 构造 Prompt
    # 将新闻列表转换为文本块
    news_context = ""
    for idx, item in enumerate(news_items, 1):
        news_context += f"{idx}. [{item['source']}] {item['title']}\n   Link: {item['url']}\n   Info: {item['desc']}\n\n"

    prompt = f"""
    你是一位极其专业、眼光独到的 AI 技术日报主编。
    请阅读以下今天抓取到的原始 AI 资讯/论文列表：

    {news_context}

    请完成以下任务，生成一份高质量的 Markdown 日报：

    1. **筛选与去重**：从列表中挑选出最重要、最具技术价值的 5-8 条新闻/论文。忽略同质化严重或无意义的内容。
    2. **中文深度点评**：
       - 将标题翻译为中文。
       - 为每一条写一段简短但深刻的点评（2-3句话）。不要只复述摘要，要指出它的技术创新点、解决了什么问题，或者对行业意味着什么。
    3. **分类展示**：请按以下类别分类：
       - 🔥 **重磅头条** (Must Read)
       - 📝 **硬核论文** (Research)
       - 🛠️ **开源/工具** (Engineering)
    4. **格式要求**：使用 Markdown 格式，包含原文链接。

    输出风格要干练、极客，拒绝废话。
    """

    # 3. AI 处理
    report_content = chat_with_llm_stream(MODEL_NAME, prompt)

    # 4. 保存文件
    output_dir = "./DailyNews/"
    os.makedirs(output_dir, exist_ok=True)
    today_str = datetime.date.today().isoformat()
    filename = os.path.join(output_dir, f"AI_Daily_Report_{today_str}.md")

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# 🤖 AI 每日深度简报 ({today_str})\n\n")
        f.write(f"> 由 {MODEL_NAME} 自动生成\n\n")
        f.write(report_content)
        f.write("\n\n---\n")
        f.write("### 🔗 原始资讯数据源\n")
        f.write(f"共抓取 {len(news_items)} 条原始数据，精选如上。")

    print(f"✅ 报告生成完毕！文件已保存为: {filename}")


if __name__ == "__main__":
    generate_daily_report()