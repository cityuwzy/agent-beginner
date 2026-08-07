"""
Code Review Agent v0.3 - 条件分支：安全检查发现严重漏洞时中断流程
"""
import os
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END


# ========== 1. State ==========
class ReviewState(TypedDict):
    code: str
    bug_report: str
    security_report: str
    style_report: str
    final_report: str
    alert_triggered: bool   # 新增：是否触发告警


# ========== 2. LLM ==========
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0.3,
)


# ========== 3. 节点函数 ==========

def analyze_bugs(state: ReviewState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是 Python 代码审查专家，只关注 Bug 和逻辑错误。逐行分析，标注严重程度：🔴严重 / 🟡中等 / 🟢建议。"),
        ("human", "{code}"),
    ])
    chain = prompt | llm
    result = chain.invoke({"code": state["code"]})
    return {"bug_report": result.content}


def check_security(state: ReviewState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是安全专家，只关注安全风险。分析 SQL 注入、命令注入、敏感信息泄露等。标注严重程度：🔴严重 / 🟡中等 / 🟢建议。"),
        ("human", "{code}"),
    ])
    chain = prompt | llm
    result = chain.invoke({"code": state["code"]})
    return {"security_report": result.content}


def alert_emergency(state: ReviewState) -> dict:
    """安全告警：跳过后续步骤，直接返回告警信息"""
    return {
        "alert_triggered": True,
        "final_report": "🚨 紧急告警：安全检查发现严重漏洞，流程已中断！请立即修复安全问题后再提交审查。",
    }


def review_style(state: ReviewState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是代码风格专家，检查 PEP 8、命名规范、注释等。标注严重程度：🔴严重 / 🟡中等 / 🟢建议。"),
        ("human", "{code}"),
    ])
    chain = prompt | llm
    result = chain.invoke({"code": state["code"]})
    return {"style_report": result.content}


def generate_summary(state: ReviewState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是技术主管，整合 Bug/安全/风格三份报告，列出最关键问题（≤5条）、评分（1-10）、100字建议。"),
        ("human", "【Bug】\n{bug}\n\n【安全】\n{security}\n\n【风格】\n{style}"),
    ])
    chain = prompt | llm
    result = chain.invoke({
        "bug": state["bug_report"],
        "security": state["security_report"],
        "style": state["style_report"],
    })
    return {"final_report": result.content}


# ========== ★ 条件路由 ★ ==========

def route_after_security(state: ReviewState) -> str:
    """读取安全检查报告，如果发现严重漏洞就跳转到告警"""
    report = state["security_report"]
    keywords = ["SQL 注入", "🔴 严重", "命令注入", "远程代码执行"]
    for kw in keywords:
        if kw in report:
            return "alert"
    return "continue"


# ========== 4. 构建 Graph ==========

builder = StateGraph(ReviewState)

builder.add_node("analyze_bugs", analyze_bugs)
builder.add_node("check_security", check_security)
builder.add_node("alert_emergency", alert_emergency)
builder.add_node("review_style", review_style)
builder.add_node("generate_summary", generate_summary)

# 固定路线
builder.add_edge(START, "analyze_bugs")
builder.add_edge("analyze_bugs", "check_security")

# ★ 条件分支：安全检查后，走告警还是继续 ★
builder.add_conditional_edges(
    "check_security",
    route_after_security,
    {"alert": "alert_emergency", "continue": "review_style"},
)

builder.add_edge("alert_emergency", END)
builder.add_edge("review_style", "generate_summary")
builder.add_edge("generate_summary", END)

app = builder.compile()


# ========== 5. 运行 ==========
if __name__ == "__main__":
    # --- 测试1：有 SQL 注入的代码（会触发告警）---
    bad_code = '''
def get_user_data(user_id):
    query = "SELECT * FROM users WHERE id = " + user_id
    return query
'''

    # --- 测试2：安全的代码（走正常流程）---
    safe_code = '''
def add(a: int, b: int) -> int:
    """返回两数之和"""
    return a + b

def greet(name: str) -> str:
    """返回问候语"""
    return f"Hello, {name}!"
'''

    for label, code in [("有漏洞", bad_code), ("安全", safe_code)]:
        print("\n" + "=" * 60)
        print(f"  测试：{label}代码")
        print("=" * 60)

        initial_state = {
            "code": code,
            "bug_report": "",
            "security_report": "",
            "style_report": "",
            "final_report": "",
            "alert_triggered": False,
        }

        final_state = app.invoke(initial_state)

        if final_state["alert_triggered"]:
            print("\n" + "🚨" * 20)
            print("  安全告警触发！流程在安全检查后中断")
            print("🚨" * 20)
            print("\n安全检查报告：")
            print(final_state["security_report"])
            print("\n" + final_state["final_report"])
        else:
            print("\n" + "─" * 40)
            print("【Bug 分析】")
            print("─" * 40)
            print(final_state["bug_report"])

            print("\n" + "─" * 40)
            print("【安全检查】")
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
