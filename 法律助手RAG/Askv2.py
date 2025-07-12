from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage
from Search import MyEmbeddingFunction
import os
import re

# 定义Prompt模板
identity = "你是专业的法律知识问答助手，必须严谨地按要求回答用户的提问。用户会向你提供参考文档（以Doc_{{i}}命名，包含问题与回答，请直接忽略仅包含问题的片段。），如果回答时你需要引用文档请直接叙述其中内容，不允许在回答中包含文档编号或提到用户提供了文档（这对用户不可见）。"
template = """请结合Context中文档的上下文片段来回答问题，禁止根据常识或其它已知信息回答问题。如果你不知道答案，直接简明扼要地回答“未找到相关答案”。
Question: {question}
Context: {context}
Answer:
"""

# 参考当前提问和检索生成回答
question = "没有赡养老人就无法继承财产吗？"
knowledges = MyEmbeddingFunction().get_docs(question)

# 设置DASHSCOPE_API_KEY
DASHSCOPE_API_KEY = "sk-"
os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY

# 构建消息
messages = [
    SystemMessage(content=identity),
    HumanMessage(content=template.format(question=question, context=knowledges)),
]

chatLLM = ChatTongyi(streaming=True, dashscope_api_key=DASHSCOPE_API_KEY)
response = chatLLM.invoke(messages)

answer = response.content
print(answer)

file_name = re.sub(r'[\\/*?:"<>|]', "", question) + "_v2.txt"
with open(file_name, "w", encoding="utf-8") as file:
    for message in messages:
        if isinstance(message, SystemMessage):
            role = "System"
        elif isinstance(message, HumanMessage):
            role = "Human"
        else:
            role = "Unknown"
        file.write(f"{role}: {message.content}\n")
    file.write("\n\n")
    file.write(answer)
