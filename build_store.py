import os, pickle
from pdf_csv_handler import load_docs
from utils import chunk_and_embed
import faiss

os.makedirs("data", exist_ok=True)
docs = load_docs("data")
texts, embeddings = chunk_and_embed(docs)

index = faiss.IndexFlatL2(len(embeddings[0]))
index.add(embeddings)
faiss.write_index(index, "index.faiss")
with open("docs.pkl", "wb") as f:
    pickle.dump(texts, f)
print("✅ Index and docs built")