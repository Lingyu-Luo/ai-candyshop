import os
import fitz  # PyMuPDF
import base64
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ================= 配置区域 =================
API_KEY = os.getenv("SILICONFLOW_API_KEY")
BASE_URL = "https://api.siliconflow.cn/v1"

# 模型配置
"""
吐槽：这个GLM-4.6V输出速度实在是慢的令人匪夷所思了，推荐换模型
"""
VLM_MODEL = "zai-org/GLM-4.6V"
LLM_MODEL = "deepseek-ai/DeepSeek-V3.2"

# ===========================================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def pdf_to_images(pdf_path, target_dpi=300):
    """
    将 PDF 转换为 base64 编码的图片列表，精准控制 DPI
    """
    doc = fitz.open(pdf_path)
    images_b64 = []

    # PDF 默认基础分辨率是 72 DPI
    # 计算缩放比例: 300 / 72 ≈ 4.166
    zoom = target_dpi / 72
    mat = fitz.Matrix(zoom, zoom)

    print(f"[-] 正在处理 PDF: {pdf_path}")
    print(f"    目标 DPI: {target_dpi} (缩放系数: {zoom:.2f}x)")

    for page_num, page in enumerate(doc):
        # get_pixmap 渲染
        pix = page.get_pixmap(matrix=mat, alpha=False)  # alpha=False 去除透明通道，稍微减小体积

        # 调试用：打印一下图片尺寸，防止过大炸显存
        # A4 纸 @ 300 DPI 大约是 2480 x 3508 像素
        if page_num == 0:
            print(f"    单页分辨率: {pix.width} x {pix.height}")

        # 转换为 PNG 格式的二进制数据
        # 压缩级别 (compression) 可以适当调高以减小网络传输压力，但不影响画质
        img_data = pix.tobytes("png")

        base64_str = base64.b64encode(img_data).decode("utf-8")
        images_b64.append((page_num + 1, base64_str))

    doc.close()
    return images_b64


def process_single_page_vlm(page_num, image_b64):
    """
    调用 GLM-4.6V 处理单页图片
    """
    prompt = (
        "你是一个专业的学术文档转换助手。"
        "请将这张图片中的内容转换为标准的 Markdown 格式。"
        "要求：\n"
        "1. 严格保留所有的数学公式，使用 LaTeX 格式（例如 $E=mc^2$）。\n"
        "2. 保持标题层级（# ## ###）。\n"
        "3. 即使图片中有页眉页脚，也请忽略它们，只提取正文。\n"
        "4. 如果有表格，请还原为 Markdown 表格。\n"
        "5. 不要输出任何闲聊，直接输出 Markdown 内容。"
    )

    try:
        response = client.chat.completions.create(
            model=VLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                        }
                    ]
                }
            ],
            temperature=0.1,  # 低温度保证准确性
            max_tokens=16384  # 防止截断
        )
        return page_num, response.choices[0].message.content
    except Exception as e:
        print(f"[!] 第 {page_num} 页处理失败: {e}")
        return page_num, ""


def refine_full_text_llm(raw_markdown):
    """
    调用 DeepSeek-V3 整理全文
    """
    print("[-] 正在进行全文校对与合并...")

    prompt = (
        "以下是从 PDF 逐页 OCR 提取的 Markdown 文本，可能包含跨页断句、重复的页码或格式不一致。"
        "请你作为编辑，整理这篇文档：\n"
        "1. 修复跨页导致的断句（将上一页未完的句子与下一页连接）。\n"
        "2. 修正明显的 OCR 拼写错误（根据上下文）。\n"
        "3. 统一数学公式的 LaTeX 格式风格。\n"
        "4. 输出最终的、干净的 Markdown 全文。\n"
        "5. 不要摘要，保留所有细节信息。"
    )

    # 如果文本过长，可能需要分段处理，这里假设是论文长度，DeepSeek-V3 的 Context 足够
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一个专业的科技文档编辑。"},
                {"role": "user", "content": f"{prompt}\n\n---\n\n{raw_markdown}"}
            ],
            temperature=0.3,
            stream=True  # 使用流式输出体验更好
        )

        full_content = ""
        print("[-] DeepSeek 输出中: ", end="", flush=True)
        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_content += content
        print("\n")
        return full_content
    except Exception as e:
        print(f"[!] 全文整理失败: {e}")
        return raw_markdown


def main(pdf_path, output_path):
    # 1. PDF 转图片
    images = pdf_to_images(pdf_path)

    # 2. VLM 并发处理 (多线程加速，因为主要是 IO 等待)
    results = {}
    with ThreadPoolExecutor(max_workers=5) as executor:  # 并发数别太高，防止触发 API Rate Limit
        futures = {executor.submit(process_single_page_vlm, p, img): p for p, img in images}

        for future in tqdm(as_completed(futures), total=len(images), desc="VLM 识别中"):
            page_num, content = future.result()
            results[page_num] = content

    # 按页码排序拼接
    sorted_pages = sorted(results.keys())
    raw_full_text = "\n\n".join([results[p] for p in sorted_pages])

    # 临时保存 OCR 结果（防止 LLM 这一步挂了白跑）
    with open("raw.md", "w", encoding="utf-8") as f:
        f.write(raw_full_text)

    # 3. LLM 最终整理
    final_text = refine_full_text_llm(raw_full_text)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_text)

    print(f"[+] 转换完成！已保存至 {output_path}")


if __name__ == "__main__":
    # 使用示例
    pdf_file = "D:/Downloads/Course Syllabus.pdf"  # 替换你的文件名
    if not os.path.exists(pdf_file):
        print(f"错误: 找不到文件 {pdf_file}，请先创建一个测试 PDF。")
    else:
        main(pdf_file, "output.md")