"""
Code Review Agent v0.4 - Tool 版：LLM 自主决定是否调用工具

核心学习点：
1. @tool 装饰器 — 把普通函数变成 LLM 能"看见"的工具
2. llm.bind_tools() — 把工具装到 LLM 上
3. ToolNode — LangGraph 内置的执行工具节点
4. 条件路由 — LLM 的 tool_calls 决定走"执行工具"还是"继续"
"""
import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# ========== 1. State ==========
class ReviewState(TypedDict):
    messages: Annotated[list, add_messages]  # 对话历史（自动追加）
    target_file: str                          # 要审查的文件路径
    code: str                                 # 文件内容
    bug_report: str
    security_report: str
    style_report: str
    final_report: str


# ========== 2. LLM ==========
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0.3,
)


# ========== 3. ★ 工具 — @tool 装饰器 ★ ==========

@tool
def read_file(filepath: str) -> str:
    """读取指定文件的全部内容。当你需要查看某个文件的源代码时调用此工具。
    
    Args:
        filepath: 文件的完整路径，例如 D:/projects/test.py
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"错误：文件 {filepath} 不存在"
    except Exception as e:
        return f"读取失败：{e}"


@tool
def count_lines(filepath: str) -> str:
    """统计文件的代码行数、注释行数和空行数。
    
    Args:
        filepath: 文件的完整路径
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        total = len(lines)
        code = sum(1 for l in lines if l.strip() and not l.strip().startswith("#"))
        return f"总行数={total}, 代码行={code}, 注释/空行={total - code}"
    except FileNotFoundError:
        return f"文件 {filepath} 不存在"


# 工具列表
tools = [read_file, count_lines]

# ★ 绑定工具到 LLM：LLM 现在"知道自己有 read_file 和 count_lines 两把刀"
llm_with_tools = llm.bind_tools(tools)


# ========== 4. 节点函数 ==========

def load_code(state: ReviewState) -> dict:
    """节点0：读取目标文件"""
    with open(state["target_file"], "r", encoding="utf-8") as f:
        return {"code": f.read()}


def analyze_bugs(state: ReviewState) -> dict:
    """节点1：Bug 分析（纯 LLM，不用工具）"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是 Bug 审查专家。逐行分析代码中的 Bug、边界问题、未处理异常。标注：🔴严重/🟡中等/🟢建议。"),
        ("human", "审查以下代码：\n```python\n{code}\n```"),
    ])
    chain = prompt | llm
    result = chain.invoke({"code": state["code"]})
    return {"bug_report": result.content}


def security_agent(state: ReviewState) -> dict:
    """
    ★ 节点2：安全审查 Agent（带工具）★
    
    这是整个 Tool 机制的核心。LLM 绑定工具后，会自己决定：
    - "我需要读文件" → 返回 tool_calls → Graph 路由到 ToolNode
    - "我能直接分析了" → 返回纯文本 → Graph 继续下一步
    
    关键：用 state["messages"] 作为对话历史，ToolNode 的结果会自动追加进去。
    """
    # 如果对话历史为空（第一次进来），创建初始消息
    if not state.get("messages"):
        user_msg = HumanMessage(
            content=f"你是一个安全审查专家。请用 read_file 读取 {state['target_file']}，"
                    f"然后用 count_lines 统计规模，最后输出安全审查报告。"
                    f"标注：🔴严重/🟡中等/🟢建议。"
        )
        response = llm_with_tools.invoke([
            SystemMessage(content="你有 read_file 和 count_lines 两个工具。请先读文件，再统计，最后分析。"),
            user_msg,
        ])
    else:
        # 已有历史（工具执行完回来了），基于历史继续
        response = llm_with_tools.invoke(state["messages"])

    return {"messages": [response]}


def after_security(state: ReviewState) -> dict:
    """工具链结束后，提取最终安全报告"""
    # 找最后一条 AI 消息作为报告
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content and not (
            hasattr(msg, "tool_calls") and msg.tool_calls
        ):
            return {"security_report": msg.content}
    return {"security_report": "安全审查未完成"}


def review_style(state: ReviewState) -> dict:
    """节点3：风格审查"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是代码风格专家。检查 PEP 8、命名、注释、函数长度。标注：🔴严重/🟡中等/🟢建议。"),
        ("human", "审查：\n```python\n{code}\n```"),
    ])
    chain = prompt | llm
    result = chain.invoke({"code": state["code"]})
    return {"style_report": result.content}


def generate_summary(state: ReviewState) -> dict:
    """节点4：汇总"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是技术主管。整合三份报告，输出最关键问题（≤5条）、评分（1-10）、100字建议。"),
        ("human", "【Bug】\n{bug}\n\n【安全】\n{security}\n\n【风格】\n{style}"),
    ])
    chain = prompt | llm
    result = chain.invoke({
        "bug": state["bug_report"],
        "security": state["security_report"],
        "style": state["style_report"],
    })
    return {"final_report": result.content}


# ========== 5. 构建 Graph ==========

builder = StateGraph(ReviewState)

# 注册节点
builder.add_node("load_code", load_code)
builder.add_node("analyze_bugs", analyze_bugs)
builder.add_node("security_agent", security_agent)        # LLM+工具
builder.add_node("call_tools", ToolNode(tools))            # 执行工具
builder.add_node("after_security", after_security)         # 提取报告
builder.add_node("review_style", review_style)
builder.add_node("generate_summary", generate_summary)

# 固定边
builder.add_edge(START, "load_code")
builder.add_edge("load_code", "analyze_bugs")
builder.add_edge("analyze_bugs", "security_agent")

# ★ 条件路由：LLM 要求调工具？去 call_tools；否则去 after_security ★
def route_security(state: ReviewState) -> str:
    last_msg = state["messages"][-1] if state["messages"] else None
    if isinstance(last_msg, AIMessage) and hasattr(last_msg, "tool_calls") \
       and last_msg.tool_calls:
        names = [tc["name"] for tc in last_msg.tool_calls]
        print(f"  🔧 LLM 调用工具: {names}")
        return "call_tools"
    return "after_security"

builder.add_conditional_edges(
    "security_agent",
    route_security,
    {"call_tools": "call_tools", "after_security": "after_security"},
)

# 工具执行完 → 回到 security_agent（让 LLM 看到结果后继续）
builder.add_edge("call_tools", "security_agent")

# 正常路径
builder.add_edge("after_security", "review_style")
builder.add_edge("review_style", "generate_summary")
builder.add_edge("generate_summary", END)

app = builder.compile()


# ========== 6. 运行 ==========
if __name__ == "__main__":
    test_file = os.path.join(os.path.dirname(__file__) or ".", "test_sample.py")
    with open(test_file, "w", encoding="utf-8") as f:
        f.write('''"""用户管理模块"""
DB_PASSWORD = "admin123"

def get_user(user_id):
    query = "SELECT * FROM users WHERE id = " + user_id
    return query

def calculate(x,y):
    result = x+y
    return result
''')

    print("=" * 60)
    print(f"审查目标：{test_file}")
    print("=" * 60)

    initial_state = {
        "messages": [],
        "target_file": test_file,
        "code": "",
        "bug_report": "",
        "security_report": "",
        "style_report": "",
        "final_report": "",
    }

    final_state = app.invoke(initial_state)

    print("\n" + "─" * 40)
    print("【Bug 分析】")
    print("─" * 40)
    print(final_state["bug_report"])

    print("\n" + "─" * 40)
    print("【安全检查】（LLM 自主调用工具完成）")
    print("─" * 40)
    print(final_state["security_report"])

    print("\n" + "─" * 40)
    print("【风格审查】")
    print("─" * 40)
    print(final_state["style_report"])

    print("\n" + "─" * 40)
    print("【汇总评分】")
    print("─" * 40)
    print(final_state["final_report"])

    os.remove(test_file)
