from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


def chunk_text(text, chunk_size=400):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]


def summarize_text(text):
    chunks = chunk_text(text, 400)
    return summarizer(" ".join(chunks), max_length=120, min_length=30, do_sample=False)[0]['summary_text']


def chunk_and_embed(docs):
    texts, embeddings = [], []
    for doc in docs:
        chunks = chunk_text(doc)
        embs = embed_model.encode(chunks)
        texts.extend(chunks)
        embeddings.extend(embs)
    return texts, np.array(embeddings)