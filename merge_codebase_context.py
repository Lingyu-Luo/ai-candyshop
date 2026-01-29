import os
import argparse
import subprocess
import shutil
import tempfile
from urllib.parse import urlparse

def is_binary_file(filename):
    """判断是否为常见的二进制/非文本文件"""
    BINARY_EXTENSIONS = {
        '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.svg', '.webp',
        '.mp4', '.mp3', '.wav', '.pdf', '.zip', '.tar', '.gz', '.7z', '.rar',
        '.pyc', '.exe', '.dll', '.so', '.dylib', '.class', '.jar', '.bin',
        '.eot', '.woff', '.woff2', '.ttf', '.lock'
    }
    _, ext = os.path.splitext(filename)
    return ext.lower() in BINARY_EXTENSIONS

def get_comment_prefix(filename):
    """根据文件扩展名决定注释符号，为了让AI更容易识别代码块"""
    _, ext = os.path.splitext(filename)
    ext = ext.lower()
    
    # 使用 # 的语言
    if ext in ['.py', '.sh', '.yaml', '.yml', '.conf', '.ini', '.rb', '.pl', '.dockerfile', 'makefile']:
        return "# "
    # HTML/XML 虽然是 ，但在 prompt 中使用 // 或 # 往往更节省token且AI也能懂
    # 这里为了通用性，除了上述脚本语言外，默认使用 //
    return "// "

def clone_repo(url):
    """克隆 GitHub 仓库到临时目录"""
    try:
        temp_dir = tempfile.mkdtemp()
        print(f"正在克隆仓库 {url} 到临时目录...")
        # --depth 1 浅克隆，速度更快，不下载历史记录
        subprocess.check_call(['git', 'clone', '--depth', '1', url, temp_dir])
        return temp_dir
    except subprocess.CalledProcessError:
        print("错误: Git 克隆失败。请检查 URL 是否正确或是否安装了 git。")
        shutil.rmtree(temp_dir)
        return None

def process_path(source_path, output_file):
    # 需要忽略的目录
    IGNORE_DIRS = {
        '.git', '.idea', '.vscode', '__pycache__', 'node_modules', 
        'dist', 'build', 'venv', 'env', '.DS_Store', 'target', 'out'
    }
    
    # 需要忽略的特定文件
    IGNORE_FILES = {
        '.DS_Store', 'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml', 
        output_file, 'LICENSE', '.gitignore'
    }

    total_files = 0
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for root, dirs, files in os.walk(source_path):
            # 过滤目录
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

            for file in files:
                if file in IGNORE_FILES or is_binary_file(file):
                    continue

                file_path = os.path.join(root, file)
                # 计算相对路径
                rel_path = os.path.relpath(file_path, source_path)
                
                # 获取注释前缀 (// 或 #)
                prefix = get_comment_prefix(file)

                try:
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        content = infile.read()
                        if not content.strip(): # 跳过空文件
                            continue
                        
                        # 核心格式：注释行 + 内容 + 两个换行
                        outfile.write(f"{prefix}{rel_path}\n")
                        outfile.write(content)
                        outfile.write("\n\n")
                        
                        total_files += 1
                        print(f"已处理: {rel_path}")

                except UnicodeDecodeError:
                    print(f"[跳过] 无法解码文件: {rel_path}")
                except Exception as e:
                    print(f"[错误] 读取 {rel_path} 失败: {e}")

    return total_files

def main():
    parser = argparse.ArgumentParser(description="将代码库合并为一个文本文件以便投喂给 AI。")
    parser.add_argument("source", help="本地文件夹路径 OR GitHub 仓库 URL")
    parser.add_argument("-o", "--output", default="context.txt", help="输出文件名 (默认: context.txt)")
    
    args = parser.parse_args()
    
    source = args.source
    is_temp = False

    # 检查是否为 URL
    if source.startswith("http://") or source.startswith("https://"):
        source = clone_repo(source)
        is_temp = True
        if not source:
            return

    if not os.path.exists(source):
        print(f"错误: 路径 '{source}' 不存在。")
        return

    try:
        print(f"开始处理...")
        count = process_path(source, args.output)
        print(f"\n成功! 共合并 {count} 个文件到 -> {args.output}")
    finally:
        # 如果是临时克隆的仓库，处理完后删除
        if is_temp and os.path.exists(source):
            print("正在清理临时文件...")
            # ignore_errors=True 防止 Windows 下git文件占用导致删除失败
            shutil.rmtree(source, ignore_errors=True) 

if __name__ == "__main__":
    main()