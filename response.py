import faiss, pickle
from sentence_transformers import SentenceTransformer
from utils import summarize_text
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

model = SentenceTransformer("all-MiniLM-L6-v2")
flan_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
flan_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
index = faiss.read_index("index.faiss")
with open("docs.pkl", "rb") as f:
    texts = pickle.load(f)

def semantic_rerank(query, candidates, top_n=3):
    inputs = [f"score: {query} <sep> {c}" for c in candidates]
    tokenized = flan_tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
    outputs = flan_model.generate(**tokenized, max_length=4)
    scores = [float(flan_tokenizer.decode(o, skip_special_tokens=True).replace("%", "")) or 0 for o in outputs]
    sorted_items = sorted(zip(candidates, scores), key=lambda x: -x[1])
    return [item[0] for item in sorted_items[:top_n]]

def get_answer(query):
    q_embed = model.encode([query])
    D, I = index.search(q_embed, k=5)
    candidates = [texts[i] for i in I[0]]
    reranked = semantic_rerank(query, candidates)
    combined = "\n".join(reranked)
    summary = summarize_text(combined + "\nAnswer the question: " + query)
    return summary.strip()
