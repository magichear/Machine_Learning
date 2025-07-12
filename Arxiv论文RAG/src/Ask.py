from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage
from dashscope import Generation
from search import Search
import os


class Ask:
    def __init__(self, search_engine=None):
        api_key = "sk-"
        os.environ["DASHSCOPE_API_KEY"] = api_key
        self.search_engine = search_engine if search_engine else Search()
        try:
            self.chatLLM = ChatTongyi(streaming=True, dashscope_api_key=api_key)
        except:
            self.chatLLM = None

    def ask(self, question, origin_query, start_year=1993, end_year=2026, k=5):
        identity, template = self.search_engine.search(
            question, origin_query, start_year, end_year, k
        )

        return self.chat(identity, template)

    def chat(self, identity, template):
        """
        提供两种调用方法: dashscope | langchain
            后者也需要安装 dashscope 包，因此建议直接用前者
        """
        try:
            answer = self.chat_dashscope(identity, template)
        except Exception as e:
            print(e)

            answer = self.chat_langchain(identity, template)
        finally:
            return answer

    def chat_langchain(self, identity, template):
        messages = [
            SystemMessage(content=identity),
            HumanMessage(content=template),
        ]
        response = self.chatLLM.invoke(messages)
        return response.content

    def chat_dashscope(self, identity, template):
        messages = [
            {"role": "system", "content": identity},
            {
                "role": "user",
                "content": template,
            },
        ]

        response = Generation.call(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            model="qwen-plus",  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            messages=messages,
            result_format="message",
        )

        if response.status_code == 200:
            answer = response.output.choices[0].message.content
            return answer
        else:
            raise Exception(
                f"HTTP返回码：{response.status_code}\n错误码：{response.code}\n错误信息：{response.message}\n"
                "请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code"
            )


if __name__ == "__main__":
    ask_chat = Ask()

    question = ["什么是注意力机制？"]

    try:
        answer = ask_chat.ask(question=question, k=5)
        print(answer)
    except Exception as e:
        print(e)
