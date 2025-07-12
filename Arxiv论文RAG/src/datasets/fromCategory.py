import os
import json


class FromCategory:
    def __init__(self, data_path=None):
        """
        这里cs.CL分类中存储的是单领域数据（跨领域的不会存储在这里）
        """
        if not data_path:
            file_path = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(file_path, "datasets/arxiv_category2ids.json")
        with open(data_path, "r", encoding="utf-8") as input_file:
            self.category_to_ids = json.load(input_file)

    def search(self, categories):
        """
        查询多个分类的并集
        """
        results = set()  # 去重
        for category in categories:
            if category in self.category_to_ids:
                results.update(self.category_to_ids[category])
        return list(results)


if __name__ == "__main__":
    from_category = FromCategory()
    categories = ["cs.CL", "cs.AI"]
    results = from_category.search(categories)
    print(f"分类 {categories} 的检索结果:", results)
