import faiss
import numpy as np


class VectorStore:
    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatL2(dimension)
        self.texts = []

    def add(self, embedding, text: str):
        vector = np.array([embedding]).astype("float32")
        self.index.add(vector)
        self.texts.append(text)

    def search(self, query_embedding):
        query_vector = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_vector, 1)
        return self.texts[indices[0][0]]
    

    


