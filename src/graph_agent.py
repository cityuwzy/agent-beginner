"""
Code Review Agent v0.2 - 基于 LangGraph 的多步骤审查工作流
"""
import os
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END


# ========== 1. 定义 State（状态） ==========
class ReviewState(TypedDict):
    code: str              # 输入的代码
    bug_report: str        # Bug 分析结果
    security_report: str   # 安全检查结果
    style_report: str      # 风格审查结果
    final_report: str      # 最终汇总报告


# ========== 2. 初始化 LLM（全局复用） ==========
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0.3,
)

# ========== 3. 节点函数 ==========

def analyze_bugs(state: ReviewState) -> dict:
    """节点1：分析 Bug 风险"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是 Python 代码审查专家，只关注 Bug 和逻辑错误。请逐行分析以下代码，指出所有潜在的 Bug、边界问题、异常未处理等情况。标注严重程度：🔴严重 / 🟡中等 / 🟢建议。"),
        ("human", "{code}"),
    ])
    chain = prompt | llm
    result = chain.invoke({"code": state["code"]})
    return {"bug_report": result.content}


def check_security(state: ReviewState) -> dict:
    """节点2：检查安全问题"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是应用安全专家，只关注代码中的安全风险。请分析以下代码：SQL 注入、命令注入、敏感信息泄露、权限问题等。标注严重程度：🔴严重 / 🟡中等 / 🟢建议。"),
        ("human", "{code}"),
    ])
    chain = prompt | llm
    result = chain.invoke({"code": state["code"]})
    return {"security_report": result.content}


def review_style(state: ReviewState) -> dict:
    """节点3：审查代码风格"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是 Python 代码风格专家，只关注代码风格和可维护性。请检查：PEP 8 规范、命名规范、注释完整性、函数长度、复杂度。标注严重程度：🔴严重 / 🟡中等 / 🟢建议。"),
        ("human", "{code}"),
    ])
    chain = prompt | llm
    result = chain.invoke({"code": state["code"]})
    return {"style_report": result.content}


def generate_summary(state: ReviewState) -> dict:
    """节点4：汇总三个维度的报告，给出最终评分"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一位技术主管，手下的三位专家分别审查了同一段代码的三个维度：
- Bug 风险
- 安全问题
- 代码风格

请整合他们的报告，生成一份最终的代码审查总结。要求：
1. 列出最关键的问题（不超过5条）
2. 给出总体评分（1-10分）
3. 写一段不超过 100 字的总结建议"""),
        ("human", """【Bug 分析报告】
{bug}

【安全审查报告】
{security}

【风格审查报告】
{style}"""),
    ])
    chain = prompt | llm
    result = chain.invoke({
        "bug": state["bug_report"],
        "security": state["security_report"],
        "style": state["style_report"],
    })
    return {"final_report": result.content}


# ========== 4. 构建 Graph（状态图） ==========

# 创建状态图，指定 State 类型
builder = StateGraph(ReviewState)

# 添加节点：每个节点是一个函数
builder.add_node("analyze_bugs", analyze_bugs)
builder.add_node("check_security", check_security)
builder.add_node("review_style", review_style)
builder.add_node("generate_summary", generate_summary)

# 定义边（执行顺序）：
# START → analyze_bugs → check_security → review_style → generate_summary → END
builder.add_edge(START, "analyze_bugs")
builder.add_edge("analyze_bugs", "check_security")
builder.add_edge("check_security", "review_style")
builder.add_edge("review_style", "generate_summary")
builder.add_edge("generate_summary", END)

# 编译成可执行的 app
app = builder.compile()


# ========== 5. 运行 ==========
if __name__ == "__main__":
    sample_code = '''
def calc(x,y):
    result=x+y
    return result

def get_user_data(user_id):
    query = "SELECT * FROM users WHERE id = " + user_id
    return query

class user:
    def __init__(self,name,age):
        self.name=name
        self.age=age
'''

    print("=" * 60, flush=True)
    print("LangGraph Code Review - 多步骤审查工作流", flush=True)
    print("=" * 60, flush=True)

    # 初始状态：只给代码，其他字段留空
    initial_state = {
        "code": sample_code,
        "bug_report": "",
        "security_report": "",
        "style_report": "",
        "final_report": "",
    }

    # 运行整个工作流
    final_state = app.invoke(initial_state)

    # 输出每个节点的中间结果
    print("\n" + "─" * 40)
    print("【节点1：Bug 分析】")
    print("─" * 40)
    print(final_state["bug_report"])

    print("\n" + "─" * 40)
    print("【节点2：安全检查】")
    print("─" * 40)
    print(final_state["security_report"])

    print("\n" + "─" * 40)
    print("【节点3：风格审查】")
    print("─" * 40)
    print(final_state["style_report"])

    print("\n" + "─" * 40)
    print("【节点4：汇总评分】")
    print("─" * 40)
    print(final_state["final_report"])
