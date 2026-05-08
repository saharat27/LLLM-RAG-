import faiss
from embed import embeded

def build_index():
    index = faiss.IndexFlatL2(embeded().shape[1])
    index.add(embeded())
    return index