from sentence_transformers import SentenceTransformer
import numpy as np
from data import load_and_chunk
from Vector_Database import build_index

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
index = build_index()

def search():
    docs = load_and_chunk()

    query = input("Enter : ").strip()

    # 🔥 1. exact match ก่อน (สำคัญมาก)
    for doc in docs:
        if query in doc:
            print("[EXACT]", doc)
            return

    # 🔥 2. vector fallback
    q_vec = model.encode([query]).astype("float32")

    D, I = index.search(q_vec, k=3)

    print("[VECTOR]")
    for i in I[0]:
        print(docs[i])

while True:
    search()
    if input("Search again? (y/n): ").lower() != "y":
        break