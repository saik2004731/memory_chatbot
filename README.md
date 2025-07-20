# Finance Memory Chatbot (Local, CPU-Optimized)

This is a memory-enhanced chatbot for finance-related documents. It supports:

 Uploading PDF and CSV files  
 FAISS-based semantic memory search  
 Automatic document summarization (PDF/CSV)  
 Semantic reranking with FLAN-T5 (CPU-friendly)  
 Short, relevant answers  
 Saves chat conversations

##Tech Stack

- Streamlit (UI)
- FAISS (Memory Search)
- SentenceTransformers (Embedding)
- Transformers (Summarization, Reranking)
- PyMuPDF, pandas (PDF/CSV Handling)

## 📂 Folder Structure

memory_chatbot/
├── app.py
├── build_store.py
├── response.py
├── pdf_csv_handler.py
├── utils.py
├── requirements.txt
├── index.faiss
├── docs.pkl
├── conversations/
├── data/
└── README.md