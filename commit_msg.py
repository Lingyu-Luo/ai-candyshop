import os
import subprocess
import sys
import json
import requests

# ================= 配置区域 =================
API_KEY = os.getenv("SILICONFLOW_API_KEY")
BASE_URL = "https://api.siliconflow.cn/v1/chat/completions"
MODEL_NAME = "deepseek-ai/DeepSeek-V3.2"

SYSTEM_PROMPT = """你是一个 Git commit message 生成专家。根据提供的 git diff 内容，生成规范的 commit message。

规则：
1. 使用 Conventional Commits 格式：<type>(<scope>): <description>
2. type 包括：feat(新功能), fix(修复), docs(文档), style(格式), refactor(重构), perf(性能), test(测试), chore(构建/工具)
3. scope 是可选的，表示影响范围（如文件名或模块名）
4. description 用英文，简洁明了，不超过 50 字符
5. 如果改动较大，可以在正文中补充说明

只输出 commit message，不要有其他解释。"""

# ===========================================


def run_git_command(args):
    """执行 git 命令并返回输出"""
    try:
        result = subprocess.run(
            ['git'] + args,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        return result.stdout, result.stderr, result.returncode
    except FileNotFoundError:
        print("❌ 错误: 未找到 git，请确保已安装并添加到 PATH")
        sys.exit(1)


def check_git_repo():
    """检查当前目录是否为 git 仓库"""
    stdout, stderr, code = run_git_command(['rev-parse', '--git-dir'])
    if code != 0:
        print("❌ 错误: 当前目录不是 git 仓库")
        sys.exit(1)


def get_staged_diff():
    """获取已暂存的更改 (git add 后的)"""
    stdout, stderr, code = run_git_command(['diff', '--cached'])
    return stdout.strip()


def get_unstaged_diff():
    """获取未暂存的更改"""
    stdout, stderr, code = run_git_command(['diff'])
    return stdout.strip()


def get_status():
    """获取 git status 简要信息"""
    stdout, stderr, code = run_git_command(['status', '--short'])
    return stdout.strip()


def generate_commit_message(diff_content):
    """调用 AI 生成 commit message"""
    if not API_KEY:
        print("❌ 错误: 请设置环境变量 SILICONFLOW_API_KEY")
        sys.exit(1)
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # 如果 diff 太长，截取前面部分
    max_diff_length = 8000
    if len(diff_content) > max_diff_length:
        diff_content = diff_content[:max_diff_length] + "\n\n... (diff truncated)"
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"请根据以下 git diff 生成 commit message:\n\n```diff\n{diff_content}\n```"}
        ],
        "max_tokens": 256,
        "temperature": 0.3,
        "stream": False
    }
    
    print("🤖 正在生成 commit message...")
    
    try:
        response = requests.post(BASE_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"❌ API 请求失败: {e}")
        sys.exit(1)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="🚀 AI Commit Message Generator - 自动生成规范的 commit message",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python commit_msg.py              # 查看 staged 的 diff 并生成 message
  python commit_msg.py -a           # 自动 stage 所有更改并生成
  python commit_msg.py -c           # 生成后直接提交
  python commit_msg.py -a -c        # stage 所有 + 生成 + 提交 (一条龙)
        """
    )
    parser.add_argument("-a", "--all", action="store_true", 
                        help="自动 git add -A (暂存所有更改)")
    parser.add_argument("-c", "--commit", action="store_true", 
                        help="生成后自动执行 git commit")
    parser.add_argument("-p", "--push", action="store_true",
                        help="提交后自动 git push")
    
    args = parser.parse_args()
    
    # 检查是否在 git 仓库中
    check_git_repo()
    
    # 显示当前状态
    status = get_status()
    if not status:
        print("✨ 工作区很干净，没有需要提交的更改")
        sys.exit(0)
    
    print("📋 当前更改状态:")
    print("-" * 40)
    print(status)
    print("-" * 40)
    print()
    
    # 如果指定 -a，先执行 git add -A
    if args.all:
        print("📦 正在暂存所有更改 (git add -A)...")
        run_git_command(['add', '-A'])
    
    # 获取 staged diff
    diff = get_staged_diff()
    
    if not diff:
        # 如果没有 staged 的内容，提示用户
        unstaged = get_unstaged_diff()
        if unstaged:
            print("⚠️  没有已暂存的更改。")
            print("   提示: 使用 -a 参数自动暂存，或先执行 git add <file>")
            sys.exit(1)
        else:
            print("✨ 没有可提交的更改")
            sys.exit(0)
    
    # 生成 commit message
    commit_msg = generate_commit_message(diff)
    
    print()
    print("=" * 50)
    print("📝 生成的 Commit Message:")
    print("=" * 50)
    print()
    print(commit_msg)
    print()
    print("=" * 50)
    
    # 如果指定 -c，自动提交
    if args.commit:
        print()
        confirm = input("确认提交? [Y/n]: ").strip().lower()
        if confirm in ['', 'y', 'yes']:
            stdout, stderr, code = run_git_command(['commit', '-m', commit_msg])
            if code == 0:
                print("✅ 提交成功!")
                print(stdout)
                
                # 如果指定 -p，自动 push
                if args.push:
                    print("🚀 正在推送到远程...")
                    stdout, stderr, code = run_git_command(['push'])
                    if code == 0:
                        print("✅ 推送成功!")
                    else:
                        print(f"❌ 推送失败: {stderr}")
            else:
                print(f"❌ 提交失败: {stderr}")
        else:
            print("已取消提交")
    else:
        # 不自动提交，输出可复制的命令
        print()
        print("💡 复制以下命令执行提交:")
        print()
        # 处理 message 中的引号
        escaped_msg = commit_msg.replace('"', '\\"')
        print(f'git commit -m "{escaped_msg}"')


if __name__ == "__main__":
    main()