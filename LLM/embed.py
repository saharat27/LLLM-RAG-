from sentence_transformers import SentenceTransformer
import numpy as np
from data import load_and_chunk

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def embedded():
    docs = load_and_chunk()
    vectors = model.encode(docs)
    return np.array(vectors).astype("float32")