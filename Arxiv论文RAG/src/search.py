from datasets.faiss_search import FaissSearch
from datasets.year_screening import FromYear
from datasets.fromCategory import FromCategory
from prompts import Prompts


class Search:
    def __init__(self):
        self.faiss_search = FaissSearch()
        self.from_year = FromYear()
        self.from_category = FromCategory()
        self.prompts = Prompts()

    def queryAutoGen(self, index):
        """
        序号从0-49, 共五十条预设问题
        """
        return self.queryGen(self.prompts.questions[index])

    def queryGen(self, query):
        prompt = self.prompts.get_extraction(query)
        return prompt["identity"], prompt["template"]

    def search(
        self, query, origin_query, start_year=1993, end_year=2026, k=5, categories=None
    ):
        """
        仅支持单向量查询，返回身份与提示词
        """
        # 检索 Top-k ids
        faiss_results = self.faiss_search.search_top_k_ids(query, k)

        # 年份范围
        year_results = self.from_year.search(start_year, end_year)

        # 类别范围
        if categories:
            category_results = self.from_category.search(categories)
        else:
            # 未指定则默认为全集
            category_results = None

        # 筛选符合两项限制条件的结果
        filtered_results = []
        for result in faiss_results:
            if result["id"] in year_results and (
                category_results is None or result["id"] in category_results
            ):
                # 获取id对应数据
                content = self.faiss_search.get_data(result["id"])
                if content:
                    result["content"] = content
                    filtered_results.append(result)

        context = "\n".join(
            [
                f"ID: {res['id']}, Distance: {res['distance']}, Content: {res['content']}"
                for res in filtered_results
            ]
        )

        prompt = self.prompts.get_query(origin_query[0], context)

        return prompt["identity"], prompt["template"]

    def search_test(self, query, start_year=1993, end_year=2026, k=5, categories=None):
        """
        仅测试使用，返回查询到的内容
        """
        # 检索 Top-k ids
        faiss_results = self.faiss_search.search_top_k_ids(query, k)

        # 年份范围
        year_results = self.from_year.search(start_year, end_year)

        # 类别范围
        if categories:
            category_results = self.from_category.search(categories)
        else:
            # 未指定则默认为全集
            category_results = None

        # 筛选符合两项限制条件的结果
        filtered_results = []
        for result in faiss_results:
            if result["id"] in year_results and (
                category_results is None or result["id"] in category_results
            ):
                # 获取id对应数据
                content = self.faiss_search.get_data(result["id"])
                if content:
                    result["content"] = content
                    filtered_results.append(result)

        context = "\n".join(
            [
                f"ID: {res['id']}, Distance: {res['distance']}, Content: {res['content']}"
                for res in filtered_results
            ]
        )

        return context


if __name__ == "__main__":
    search = Search()

    # 查询问题、年份范围和类别范围
    query = ["Transformer, RNN, machine translation, performance comparison"]
    origin_query = ["对比Transformer与RNN在机器翻译中的性能"]
    start_year = 1993
    end_year = 2026
    categories = ["cs.CL", "cs.AI"]

    identity, results = search.search(
        query, origin_query, start_year, end_year, k=5, categories=categories
    )

    print(identity)
    print(results)
