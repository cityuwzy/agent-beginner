"""Code Review Agent - 基于 LangChain 的智能代码审查助手"""
import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 配置 DeepSeek LLM
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0.3,
)

# Prompt 模板
review_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一位资深代码审查专家。请对以下 Python 代码进行全面审查，
按四个维度输出结构化报告：

1. **Bug 风险**：逻辑错误、边界问题、潜在崩溃
2. **代码风格**：命名规范、可读性、注释
3. **性能问题**：低效算法、不必要开销
4. **安全风险**：注入漏洞、敏感信息暴露

每条问题请标注：
- 严重程度：🔴严重 / 🟡中等 / 🟢建议
- 所在行号
- 具体问题和修改建议

最后给出一个总体评分（1-10分）。"""),
    ("human", "请审查以下代码：\n\n```python\n{code}\n```"),
])

# 组装链
chain = review_prompt | llm


def review_code(code: str) -> str:
    """审查一段 Python 代码，返回审查报告"""
    response = chain.invoke({"code": code})
    return response.content


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

    print("=" * 60)
    print("Code Review Agent - 代码审查报告")
    print("=" * 60)
    report = review_code(sample_code)
    print(report)
