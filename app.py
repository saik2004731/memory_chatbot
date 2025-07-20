import streamlit as st
from response import get_answer
from pdf_csv_handler import save_file
import os
import datetime

st.set_page_config(page_title="Memory Chatbot", layout="wide")
st.title("🤖 Memory-Enhanced Chatbot")

uploaded_file = st.file_uploader("Upload PDF/CSV", type=["pdf", "csv"])
if uploaded_file:
    file_path = save_file(uploaded_file)
    st.success(f"Saved: {file_path}")

query = st.text_input("Ask a question")
if query:
    answer = get_answer(query)
    st.markdown(f"**Answer:** {answer}")

    # Save Q&A
    os.makedirs("conversations", exist_ok=True)
    with open(f"conversations/chat_{datetime.date.today()}.txt", "a") as f:
        f.write(f"Q: {query}\nA: {answer}\n\n")