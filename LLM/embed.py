from sentence_transformers import SentenceTransformer
import numpy as np
from data import data_doc
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


# print(docs)
def embeded ():
    docs = data_doc()
    embeddings = model.encode(docs)
    embeddings = np.array(embeddings).astype("float32")
    return embeddings