from sentence_transformers import SentenceTransformer
import numpy as np
from data import data_doc
from index import build_index

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

index = build_index()

def search():
    docs = data_doc()
    query = input("Enter : ")

    q_vec = model.encode(query).astype("float32")
    q_vec = np.array([q_vec])

    D, I = index.search(q_vec, k=1)

    print([docs[i] for i in I[0]])

search()