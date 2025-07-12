from dashscope import Generation

# from Search import MyEmbeddingFunction
import os
import re

# 定义Prompt模板
identity = "你是专业的法律知识问答助手，必须严谨地按要求回答用户的提问。用户会向你提供参考文档（以Doc_{{i}}命名，包含问题与回答，请直接忽略仅包含问题的片段。），如果回答时你需要引用文档请直接叙述其中内容，不允许在回答中包含文档编号或提到用户提供了文档（这对用户不可见）。"
template = """请结合Context中文档的上下文片段来回答问题，禁止根据常识或其它已知信息回答问题。如果你不知道答案，直接简明扼要地回答“未找到相关答案”。
Question: {question}
Context: {context}
Answer:
"""

DASHSCOPE_API_KEY = "sk-"
os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY

# 参考当前提问和检索生成回答
question = "借款人去世，继承人是否应履行偿还义务？"
knowledges = "AAAAAAA"

# 构建消息
messages = [
    {"role": "system", "content": identity},
    {"role": "user", "content": template.format(question=question, context=knowledges)},
]

# 调用大模型
response = Generation.call(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="qwen-plus",  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    messages=messages,
    result_format="message",
)

# 处理响应
if response.status_code == 200:
    answer = response.output.choices[0].message.content
    # 处理文件名，去除特殊字符
    file_name = re.sub(r'[\\/*?:"<>|]', "", question) + ".txt"
    print(answer)
    answer = messages + "\n\n" + answer
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(answer)
else:
    print(f"HTTP返回码：{response.status_code}")
    print(f"错误码：{response.code}")
    print(f"错误信息：{response.message}")
    print(
        "请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code"
    )
