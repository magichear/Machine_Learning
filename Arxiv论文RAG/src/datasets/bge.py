from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
import os


class BGE:
    def __init__(self, model_path=None):
        if not model_path:
            file_path = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(file_path, "models/bge-base-en-v1.5")
        self.model = SentenceTransformer(
            model_path, device="cuda" if torch.cuda.is_available() else "cpu"
        )

    def embed_texts(self, texts):
        embeddings = []
        for text in tqdm(texts, desc="生成嵌入中"):
            embedding = self.model.encode([text], convert_to_tensor=True)
            embeddings.append(embedding.cpu().numpy())
        return embeddings
