import faiss
from embed import embedded

def build_index():
    vecs = embedded()
    index = faiss.IndexFlatL2(vecs.shape[1])
    index.add(vecs)
    return index