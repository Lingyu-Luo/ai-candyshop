import os
import argparse
import sys
import json
import requests
import subprocess
import shutil
import tempfile

headers = {
    "Authorization": f"Bearer {os.getenv('SILICONFLOW_API_KEY')}",
    "Content-Type": "application/json"
}

hybrid_model_list = [
    "deepseek-ai/DeepSeek-V3.1-Terminus",
    "Pro/deepseek-ai/DeepSeek-V3.1-Terminus",
    "deepseek-ai/DeepSeek-V3.2",
    "Pro/deepseek-ai/DeepSeek-V3.2",
    "zai-org/GLM-4.5V",
    "Qwen/Qwen3-VL-235B-A22B-Thinking"
]

base_url = "https://api.siliconflow.cn/v1/chat/completions"
default_model = "deepseek-ai/DeepSeek-V3.2"

# 需要忽略的目录和文件
IGNORE_DIRS = {
    '.git', '.idea', '.vscode', '__pycache__', 'node_modules', 
    'dist', 'build', 'venv', 'env', '.DS_Store', 'target', 'out'
}

IGNORE_FILES = {
    '.DS_Store', 'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml', 
    'LICENSE', '.gitignore'
}

BINARY_EXTENSIONS = {
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.svg', '.webp',
    '.mp4', '.mp3', '.wav', '.pdf', '.zip', '.tar', '.gz', '.7z', '.rar',
    '.pyc', '.exe', '.dll', '.so', '.dylib', '.class', '.jar', '.bin',
    '.eot', '.woff', '.woff2', '.ttf', '.lock'
}

SYSTEM_PROMPT = """你是一位资深的代码审查专家。请对提供的代码进行全面审查，包括：

1. **代码质量**: 可读性、命名规范、代码结构
2. **潜在问题**: Bug、安全漏洞、性能问题
3. **最佳实践**: 是否遵循语言/框架的最佳实践
4. **改进建议**: 具体的优化方案和重构建议

请用 Markdown 格式输出审查报告，结构清晰，重点突出。"""


def make_payload(model: str, messages: list, enable_thinking: bool | None = None, stream: bool = True):
    """构建请求负载"""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 131072,
        "temperature": 0.3,
        "top_p": 0.95,
        "stream": stream
    }

    if enable_thinking is not None and model in hybrid_model_list:
        payload["enable_thinking"] = enable_thinking

    return payload


def is_binary_file(filename):
    """判断是否为二进制文件"""
    _, ext = os.path.splitext(filename)
    return ext.lower() in BINARY_EXTENSIONS


def clone_repo(url):
    """克隆 GitHub 仓库到临时目录"""
    try:
        temp_dir = tempfile.mkdtemp()
        print(f"正在克隆仓库 {url} 到临时目录...")
        subprocess.check_call(['git', 'clone', '--depth', '1', url, temp_dir], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return temp_dir
    except subprocess.CalledProcessError:
        print("错误: Git 克隆失败。请检查 URL 是否正确或是否安装了 git。")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return None


def read_single_file(file_path):
    """读取单个文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        filename = os.path.basename(file_path)
        return f"// {filename}\n{content}"
    except UnicodeDecodeError:
        print(f"[跳过] 无法解码文件: {file_path}")
        return None
    except Exception as e:
        print(f"[错误] 读取 {file_path} 失败: {e}")
        return None


def read_directory(dir_path):
    """读取目录下所有文件并合并"""
    contents = []
    file_count = 0
    
    for root, dirs, files in os.walk(dir_path):
        # 过滤目录
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        for file in files:
            if file in IGNORE_FILES or is_binary_file(file):
                continue
            
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, dir_path)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if not content.strip():
                        continue
                    
                    contents.append(f"// {rel_path}\n{content}")
                    file_count += 1
                    print(f"已读取: {rel_path}")
            except UnicodeDecodeError:
                print(f"[跳过] 无法解码文件: {rel_path}")
            except Exception as e:
                print(f"[错误] 读取 {rel_path} 失败: {e}")
    
    print(f"共读取 {file_count} 个文件")
    return "\n\n".join(contents)


def read_content(path):
    """根据路径类型读取内容"""
    is_temp = False
    source_path = path
    
    # 检查是否为 GitHub URL
    if path.startswith("http://") or path.startswith("https://"):
        source_path = clone_repo(path)
        is_temp = True
        if not source_path:
            return None, False
    
    if not os.path.exists(source_path):
        print(f"错误: 路径 '{source_path}' 不存在。")
        return None, is_temp
    
    # 判断是文件还是目录
    if os.path.isfile(source_path):
        content = read_single_file(source_path)
    else:
        content = read_directory(source_path)
    
    return content, is_temp


def stream_review(content, model, output_path):
    """流式请求并实时输出审查结果"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"请审查以下代码：\n\n{content}"}
    ]
    
    enable_thinking = True if model in hybrid_model_list else None
    payload = make_payload(model, messages, enable_thinking=enable_thinking, stream=True)
    
    print("\n" + "="*60)
    print("🔍 AI 代码审查中...")
    print("="*60 + "\n")
    
    full_answer = ""
    full_reasoning = ""
    
    try:
        response = requests.post(base_url, json=payload, headers=headers, stream=True)
        response.raise_for_status()
        
        for chunk in response.iter_lines():
            if chunk:
                chunk_str = chunk.decode('utf-8').replace('data: ', '')
                if chunk_str == "[DONE]":
                    break
                
                try:
                    chunk_data = json.loads(chunk_str)
                except json.JSONDecodeError:
                    continue
                
                delta = chunk_data.get('choices', [{}])[0].get('delta', {})
                content_piece = delta.get('content', '')
                reasoning_content = delta.get('reasoning_content', '')
                
                if content_piece:
                    full_answer += content_piece
                    print(content_piece, end='', flush=True)
                
                if reasoning_content:
                    full_reasoning += reasoning_content
        
        print("\n")
        
        # 保存到文件
        with open(output_path, 'w', encoding='utf-8') as f:
            if full_reasoning.strip():
                f.write("# AI 代码审查报告\n\n")
                f.write("<details>\n<summary>🧠 推理过程</summary>\n\n")
                f.write(full_reasoning.strip())
                f.write("\n\n</details>\n\n")
                f.write("---\n\n")
            f.write(full_answer)
        
        print("="*60)
        print(f"✅ 审查完成！结果已保存到: {output_path}")
        print("="*60)
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ 请求失败: {str(e)}")
        sys.exit(1)


def main():
    argparser = argparse.ArgumentParser(description="AI Code Reviewer - 智能代码审查工具")
    argparser.add_argument("path", type=str, help="代码路径 (文件/文件夹/GitHub URL)")
    argparser.add_argument("-o", "--output", default="AI Review.md", type=str, help="审查报告输出路径")
    argparser.add_argument("-m", "--model", default=default_model, type=str, help=f"使用的模型 (默认: {default_model})")
    
    args = argparser.parse_args()
    
    # 检查 API Key
    if not os.getenv('SILICONFLOW_API_KEY'):
        print("❌ 错误: 请设置环境变量 SILICONFLOW_API_KEY")
        sys.exit(1)
    
    print(f"📂 输入路径: {args.path}")
    print(f"🤖 使用模型: {args.model}")
    print(f"📝 输出文件: {args.output}")
    print()
    
    # 读取代码内容
    content, is_temp = read_content(args.path)
    temp_dir = args.path if is_temp else None
    
    if not content:
        print("❌ 无法读取代码内容")
        sys.exit(1)
    
    try:
        # 执行审查
        stream_review(content, args.model, args.output)
    finally:
        # 清理临时目录
        if is_temp and temp_dir and args.path.startswith("http"):
            # 需要从 read_content 返回临时目录路径
            pass


if __name__ == "__main__":
    main()