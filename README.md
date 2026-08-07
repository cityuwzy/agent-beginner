# Agent Beginner — 从零开始学 AI Agent

一个渐进式的 AI Agent 学习项目，使用 **LangGraph + DeepSeek** 从单 agent 讲到 multi-agent 协作。

## 项目结构

```
agent-beginner/
├── src/
│   ├── agent.py              # v0.1：单 agent 代码审查（入门）
│   ├── graph_agent.py        # v0.2：串行流水线审查（LangGraph 入门）
│   ├── graph_agent_select.py # v0.3：条件分支（智能路由）
│   ├── graph_agent_tool.py   # v0.4：Tool Calling（LLM 自主调用工具）
│   └── phase2_multi_agent/   # 真正的 Multi-Agent 协作（开发中）
├── tests/
│   └── sample_code.py        # 测试用的 Python 样本代码
└── requirements.txt
```

## 学习路线

| 阶段 | 文件 | 学到什么 |
|------|------|----------|
| v0.1 | `agent.py` | LangChain 基础：LLM + PromptTemplate + Chain |
| v0.2 | `graph_agent.py` | LangGraph 基础：StateGraph、节点、边、状态流转 |
| v0.3 | `graph_agent_select.py` | 条件路由：根据中间结果动态选择下一步 |
| v0.4 | `graph_agent_tool.py` | Tool Calling：LLM 自主决定何时调用外部工具 |
| Phase 2 | `phase2_multi_agent/` | Multi-Agent 协作：agent 之间互相阅读报告、补充纠正 |

## 技术栈

- **框架**：LangChain + LangGraph
- **模型**：DeepSeek (`deepseek-chat`)
- **语言**：Python 3.13+

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/cityuwzy/agent-beginner.git
cd agent-beginner
```

### 2. 创建虚拟环境

```bash
# Windows PowerShell
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

```bash
# 国内用户（推荐）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 海外用户
pip install -r requirements.txt
```

### 4. 配置 API Key

在项目根目录创建 `.env` 文件：

```env
DEEPSEEK_API_KEY=你的DeepSeek_API_Key
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

### 5. 运行

```bash
# 单 agent 审查
python src/agent.py

# 串行流水线审查
python src/graph_agent.py

# 条件分支审查
python src/graph_agent_select.py

# Tool Calling 审查
python src/graph_agent_tool.py
```

## 环境要求

- Python 3.10+
- DeepSeek API Key（[申请地址](https://platform.deepseek.com)）
