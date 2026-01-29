# 🍭 AI Candy Shop

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=flat&logo=python)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-open%20for%20business-orange)

> **A playful collection of bite-sized Python AI scripts.**  
> *Small code, sweet results.*

Welcome to the **AI Candy Shop**. This repository houses a variety of lightweight, single-purpose scripts designed to automate tasks, process data, and explore AI capabilities without the overhead of heavy frameworks.

## 🍬 The Menu

Here is what's currently on the shelf. Pick your flavor:

### 🔬 Research & Analysis

| Script | Description |
|--------|-------------|
| **`research_exa.py`** | Deep research assistant powered by Exa API. Multi-step academic search (arXiv, Nature, IEEE, ACM) with AI-driven analysis, configurable depth, and streaming output. |
| **`thinking_ReAct.py`** | ReAct agent with search + visit tools. Leverages DeepSeek's thinking capabilities for reasoning-intensive tasks with real-time thought visualization. |
| **`code_reviewer.py`** | AI-powered code review tool. Analyzes files, directories, or GitHub repos and generates detailed Markdown reports. |

### 📄 Document Processing

| Script | Description |
|--------|-------------|
| **`pdf2md.py`** | Converts PDF documents to clean Markdown using VLM (GLM-4.6V) for OCR and LLM (DeepSeek-V3) for post-processing. Handles LaTeX formulas and tables. |

### 🗞️ Information & Media

| Script | Description |
|--------|-------------|
| **`daily_news.py`** | AI news aggregator. Fetches from Hacker News, ArXiv, and Hugging Face Daily Papers, then generates curated daily reports with DeepSeek analysis. |

### 🌐 Language & Translation

| Script | Description |
|--------|-------------|
| **`translategemma.py`** | Streamlit UI for TranslateGemma model running on Ollama. Supports 17+ languages with professional translation prompts. |

## 🛠️ Prerequisites

- Python 3.11+
- API Keys (set as environment variables):
  - `SILICONFLOW_API_KEY` - For DeepSeek/GLM models
  - `EXA_API_KEY` - For Exa search (research scripts)
- [Ollama](https://ollama.ai/) - For local model inference (translategemma)

## 👩‍🍳 Usage

```bash
# Deep Research (Streamlit App)
streamlit run research_exa.py

# ReAct Agent with thinking
python thinking_ReAct.py

# Code Review (file, folder, or GitHub URL)
python code_reviewer.py ./my_project -o review.md
python code_reviewer.py https://github.com/user/repo

# PDF to Markdown
python pdf2md.py

# Daily AI News
python daily_news.py

# Translation UI
streamlit run translategemma.py
```

## 📁 Output Directories

| Script | Output Location |
|--------|-----------------|
| `research_exa.py` | `./output/DeepResearch/` |
| `daily_news.py` | `./DailyNews/` |
| `code_reviewer.py` | `./AI Review.md` (configurable) |
| `pdf2md.py` | `./output.md` + `./raw.md` |

## 📜 License

This project is licensed under the MIT License - feel free to taste, fork, and modify!

---

*Created by [Lingyu Luo](https://github.com/Lingyu-Luo)*