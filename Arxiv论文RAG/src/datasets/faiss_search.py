import os
import faiss  # pip install faiss-cpu
import json

try:
    from bge import BGE
except:
    from datasets.bge import BGE  # 方便从父目录调用

import numpy as np


class FaissSearch:
    def __init__(self, id_map_path=None, index_path=None, simplify_data_path=None):
        """
        此处查询相似id，在rag中再加入年份范围等限制，最后再于此处使用get_data封装具体数据
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if not index_path:
            index_path = os.path.join(current_dir, "datasets/faiss_cscl_ivfpq.idx")
        if not id_map_path:
            id_map_path = os.path.join(current_dir, "datasets/bge_id_map_cscl.json")
        if not simplify_data_path:
            simplify_data_path = os.path.join(
                current_dir, "datasets/simplify_data.json"
            )
        self.index = faiss.read_index(index_path)
        # print("[DEBUG] Index type:", self.index.metric_type)  # 输出 1
        with open(id_map_path, "r") as f:
            self.ids = json.load(f)
        with open(simplify_data_path, "r") as f:
            self.simplify_data = json.load(f)

        self.bge = BGE()

    def search_top_k_ids(self, query, k=5):
        """
        一次仅对单条向量查询，返回欧几里得距离
        [DEBUG] Index type: <class 'faiss.swigfaiss_avx2.IndexIVFPQ'>
        """
        # 向量化
        query_vector = np.array(self.bge.embed_texts(query), dtype=np.float32)
        # print("[DEBUG] Query vector shape:", query_vector.shape)

        # Top-k
        distances, indices = self.index.search(query_vector[0], k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                results.append({"id": self.ids[idx], "distance": distances[0][i]})
        return results

    def get_data(self, id):
        if id in self.simplify_data:
            return self.simplify_data[id]
        else:
            return None


if __name__ == "__main__":
    faiss_search = FaissSearch()

    query = [
        "anism within the context of neural network architectures, with a focus on its role in enhancing model performance and interpretability."
    ]
    top_k_results = faiss_search.search_top_k_ids(query, k=5)

    print("召回结果：")
    for result in top_k_results:
        print(result)
        # print(faiss_search.get_data(result["id"]))
