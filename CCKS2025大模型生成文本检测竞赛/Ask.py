# %%
from dashscope import Generation
import os
import json

# 定义Prompt模板
identity = "You are a professional and rigorous AI-generated text detection assistant, specializing in determining whether a text is AI-generated."
template = """Please determine whether each non-empty line in the following text is AI-generated. For each individual non-empty line, if it is AI-generated, process it as the number '1'; otherwise, process it as the number '0'. Detect all non-empty lines in order and return all processed results in sequence, separated by commas. These results should be combined into a single string before returning.

{texts}
"""

DASHSCOPE_API_KEY = "sk-831337d11e384c33a2963267b9ee94ef"
os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY

TEST_PATH = r"D:\Study_Work\Electronic_data\CS\AAAUniversity\Machine_Learning\sdxxdl\co_lab\datasets\test.jsonl"

with open(TEST_PATH, "r", encoding="utf-8") as f:
    texts = [json.loads(line)["text"] for line in f if line.strip()]

print(len(texts))  # 2800

texts1 = texts[:1000]
texts2 = texts[1000:2000]
texts3 = texts[2000:2800]

# %%

# 构建消息
messages = [
    {"role": "system", "content": identity},
    {"role": "user", "content": template.format(texts="\n\n".join(texts2))},
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
    results = response.output.choices[0].message.content.strip()

    # 将结果写入同目录下的 results.txt 文件
    with open("texts2.txt", "w", encoding="utf-8") as file:
        file.write(results)

    print("检测结果已写入 texts2.txt 文件")
    print(results)
else:
    print(f"HTTP返回码：{response.status_code}")
    print(f"错误码：{response.code}")
    print(f"错误信息：{response.message}")
    print(
        "请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code"
    )

# %%
