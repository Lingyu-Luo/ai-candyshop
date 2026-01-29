import os
import json
import re
import requests
from colorama import Fore, Style, init

init(autoreset=True)

# 配置
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
EXA_API_KEY = os.getenv("EXA_API_KEY")
BASE_URL = "https://api.siliconflow.cn/v1/chat/completions"
MODEL_NAME = "deepseek-ai/DeepSeek-V3.2" 

# --- 工具函数：JSON 提取器 ---
def extract_json_from_text(text):
    if not text: return None
    try: return json.loads(text)
    except: pass
    try:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match: return json.loads(match.group(1))
    except: pass
    try:
        start = text.find('{')
        if start == -1: return None
        bracket_count = 0
        for i in range(start, len(text)):
            if text[i] == '{': bracket_count += 1
            elif text[i] == '}': bracket_count -= 1
            if bracket_count == 0: return json.loads(text[start:i+1])
    except: pass
    return None

# --- 1. LLM 调用层 ---
def call_llm_step(messages):
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.5,
        "response_format": {"type": "json_object"}, 
        "stream": True,
    }
    if "DeepSeek" in MODEL_NAME: payload["enable_thinking"] = True

    try:
        resp = requests.post(BASE_URL, headers=headers, json=payload, stream=True, timeout=120)
        if resp.status_code != 200:
            print(f"{Fore.RED}API Error: {resp.text}{Style.RESET_ALL}")
            return {"content": "", "reasoning": ""}
        resp.raise_for_status()
    except Exception as e:
        print(f"{Fore.RED}Request Failed: {e}{Style.RESET_ALL}")
        return {"content": "", "reasoning": ""}
    
    full_content = []
    full_reasoning = []
    
    print(f"\n{Fore.CYAN}--- 正在思考 (Internal Monologue) ---{Style.RESET_ALL}")
    for line in resp.iter_lines():
        if not line: continue
        s = line.decode("utf-8")
        if s.startswith("data: "): s = s[len("data: "):]
        if s.strip() == "[DONE]": break
        try:
            chunk = json.loads(s)
            if "choices" not in chunk or not chunk["choices"]: continue
            delta = chunk["choices"][0]["delta"]
            if "reasoning_content" in delta:
                r = delta["reasoning_content"]
                if r:
                    print(f"{Fore.CYAN}{r}{Style.RESET_ALL}", end="", flush=True)
                    full_reasoning.append(r)
            if "content" in delta:
                c = delta["content"]
                if c: full_content.append(c)
        except: continue
    print(f"\n{Fore.CYAN}--- 思考结束 ---{Style.RESET_ALL}\n")
    return {"content": "".join(full_content), "reasoning": "".join(full_reasoning)}

# --- 2. 增强工具层 (Search + Visit) ---

def tools_search(query):
    """搜索工具：返回标题、URL和智能摘要(Highlights)"""
    if not EXA_API_KEY: return "Error: EXA_API_KEY not set."
    print(f"{Fore.GREEN}>>> [Tool: Search] {query}{Style.RESET_ALL}")
    
    url = "https://api.exa.ai/search"
    headers = {"x-api-key": EXA_API_KEY, "content-type": "application/json"}
    payload = {
        "query": query,
        "useAutoprompt": True,
        "numResults": 3,
        # 关键改进：使用 highlights 而不是简单的 text 切片
        "contents": {
            "highlights": {
                "numSentences": 3, # 每个结果提取3句最相关的话
                "query": query     # 基于问题进行高亮
            },
            "text": False # 初次搜索不需要全文，节省Token
        }
    }
    try:
        res = requests.post(url, json=payload, headers=headers).json()
        results = res.get("results", [])
        if not results: return "No results found."
        
        obs = f"Found {len(results)} pages. If you need details, use 'visit' action on a URL.\n\n"
        for i, r in enumerate(results):
            highlights = r.get("highlights", [])
            snippet = " ... ".join(highlights) if highlights else "No highlights."
            obs += f"[{i+1}] Title: {r.get('title')}\n    URL: {r.get('url')}\n    Highlights: {snippet}\n\n"
        return obs
    except Exception as e:
        return f"Search failed: {e}"

def tools_visit(url):
    """
    阅读工具：利用长上下文能力，抓取网页的大量正文
    """
    if not EXA_API_KEY: return "Error: EXA_API_KEY not set."
    
    # 增加反馈提示
    print(f"{Fore.BLUE}>>> [Tool: Visit] Fetching massive content from: {url}{Style.RESET_ALL}")
    
    url_api = "https://api.exa.ai/contents"
    headers = {"x-api-key": EXA_API_KEY, "content-type": "application/json"}
    
    payload = {
        "ids": [url],
        "text": { 
            # === 关键修改 ===
            # DeepSeek-V3 支持 128k context。
            # 我们直接请求 30,000 ~ 50,000 字符，通常足以覆盖一篇长论文的核心部分。
            # 如果是 Gemini 1.5 Pro，甚至可以开到 1,000,000。
            "maxCharacters": 40000, 
            "includeHtmlTags": False 
        }
    }
    try:
        res = requests.post(url_api, json=payload, headers=headers).json()
        results = res.get("results", [])
        if not results: return "Failed to retrieve content."
        
        raw_text = results[0].get("text", "")
        
        # 简单的清洗，去掉过多的空行
        cleaned_text = re.sub(r'\n\s*\n', '\n\n', raw_text)
        
        # 统计一下抓了多少字，给控制台一个反馈
        content_len = len(cleaned_text)
        print(f"{Fore.BLUE}>>> [System] Successfully retrieved {content_len} characters.{Style.RESET_ALL}")
        
        if content_len > 40000:
            return f"--- Content of {url} (Truncated at 40k chars) ---\n{cleaned_text[:40000]}\n--- End of Content ---"
        else:
            return f"--- Full Content of {url} ---\n{cleaned_text}\n--- End of Content ---"
            
    except Exception as e:
        return f"Visit failed: {e}"

# --- 3. Agent 循环 ---
def run_gemini_simulation(user_query):
    # 更新 System Prompt，告知有两个工具
    system_prompt = """
    You are Gemini 3 Pro simulator.
    
    TOOLS:
    1. "search": Use to find URLs. (Returns summaries only).
    2. "visit": Use to read the EXTENSIVE content of a URL. 
       Note: This tool can fetch up to 40,000 characters. 
       USE THIS for technical reports, papers, or when looking for specific data points buried deep in text.
    
    PROTOCOL:
    1. Output decision in STRICT JSON format.
    
    JSON FORMATS:
    
    [Search]:
    {"thought": "Need to find external info...", "action": "search", "action_input": "query"}
    
    [Visit URL]:
    {"thought": "Result #1 looks promising but summary is too short...", "action": "visit", "action_input": "https://url..."}
    
    [Answer]:
    {"thought": "I have enough info...", "action": "answer", "action_input": "Final answer"}
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]
    
    max_steps = 7 # 增加步数，因为可能需要 Search -> Visit -> Answer
    step = 0
    
    print(f"{Fore.YELLOW}User Query: {user_query}{Style.RESET_ALL}")

    while step < max_steps:
        step += 1
        result = call_llm_step(messages)
        
        if not result["content"] and not result["reasoning"]:
            break

        decision = extract_json_from_text(result["content"])
        if not decision:
            print(f"{Fore.YELLOW}[System] Looking into reasoning trace...{Style.RESET_ALL}")
            decision = extract_json_from_text(result["reasoning"])
            
        if not decision:
            print(f"{Fore.RED}JSON parse failed.{Style.RESET_ALL}")
            messages.append({"role": "user", "content": "Error: No valid JSON found. Please output ONLY JSON."})
            continue
            
        thought = decision.get("thought", "")
        action = decision.get("action", "")
        action_input = decision.get("action_input", "")
        
        print(f"{Fore.MAGENTA}[Step {step} Decision] {thought}{Style.RESET_ALL}")
        
        observation = ""
        if action == "search":
            observation = tools_search(action_input)
        elif action == "visit":
            observation = tools_visit(action_input)
        elif action == "answer":
            print(f"\n{Fore.WHITE}{Style.BRIGHT}=== Gemini 3 Pro Final Answer ==={Style.RESET_ALL}")
            print(action_input)
            return action_input
        else:
            observation = f"Unknown action: {action}"

        # 更新历史
        messages.append({"role": "assistant", "content": json.dumps(decision, ensure_ascii=False)})
        messages.append({
            "role": "user", 
            "content": f"Observation from {action}:\n{observation}"
        })

    print(f"{Fore.RED}Max steps reached.{Style.RESET_ALL}")

if __name__ == "__main__":
    if not SILICONFLOW_API_KEY:
        print("请设置 SILICONFLOW_API_KEY")
    else:
        # 测试需要深度阅读的问题
        run_gemini_simulation("DeepSeek-V3.2 在数学能力上相比 V3.1-Terminus 有多少提升？请引用具体分数。")