from chromadb import EmbeddingFunction
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from langchain_chroma import Chroma


# 自定义嵌入函数
class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer(
            "D:\Study_Work\Electronic_data\CS\AAAUniversity\Web\Lab\L3\m3e-base"
        )

    def embed_documents(self, texts):
        # 使用 tqdm 显示嵌入进度条
        embeddings = [
            self.model.encode(text, convert_to_numpy=True)
            for text in tqdm(texts, desc="Embedding Texts", unit="text")
        ]
        # 返回嵌入结果
        return [embedding.tolist() for embedding in embeddings]

    def embed_query(self, query):
        # 对查询生成嵌入
        return self.model.encode(query, convert_to_numpy=True).tolist()

    def get_docs(self, query):
        embedding_function = MyEmbeddingFunction()
        # 数据库
        persist_directory = "lab3_1\chroma_data_300"
        # 加载 Chroma 数据库
        db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_function,
        )

        def fetch_docs(query, num_results):
            try:
                docs = db.similarity_search(query, num_results)
                res = ""
                j = 0
                if docs:
                    for i, doc in enumerate(docs):
                        # 筛选输出的结果
                        if len(doc.page_content) >= 35 or ("\n" in doc.page_content):
                            doc_content = (
                                doc.page_content
                                if doc.page_content[:5] != "data:"
                                else doc.page_content[6:]
                            )
                            res = res + f"Doc_{j + 1}:" + doc_content + "\n"
                            j += 1
                return res, j
            except Exception as e:
                print(f"Error during similarity search: {e}")
                return "", 0

        # 首先尝试获取前10个相似文档
        res, count = fetch_docs(query, 10)

        # 如果结果少于5个，再尝试获取前20个相似文档
        if count < 5:
            res, count = fetch_docs(query, 20)

        return res


if __name__ == "__main__":
    print(MyEmbeddingFunction().get_docs("没有赡养老人就无法继承财产吗？"))
