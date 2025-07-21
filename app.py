import streamlit as st
from response import get_answer, initialize_models
from pdf_csv_handler import save_file, load_docs
from build_store import build_vector_store
import os
import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Finance Memory Chatbot", 
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        background-color: #f8f9fa;
    }
    .success-message {
        background-color: #d4edda;
        border-color: #28a745;
        color: #155724;
        padding: 0.75rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-message {
        background-color: #f8d7da;
        border-color: #dc3545;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'vector_store_ready' not in st.session_state:
    st.session_state.vector_store_ready = os.path.exists("index.faiss") and os.path.exists("docs.pkl")

# Header
st.markdown("""
<div class="main-header">
    <h1>📊 Finance Memory Chatbot</h1>
    <p>Upload financial documents and ask intelligent questions with AI-powered memory</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for file management and settings
with st.sidebar:
    st.header("📁 Document Management")
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Upload Financial Documents", 
        type=["pdf", "csv"],
        help="Upload PDF reports or CSV data files for analysis"
    )
    
    if uploaded_file:
        try:
            with st.spinner("Processing file..."):
                file_path = save_file(uploaded_file)
                st.markdown(f"""
                <div class="success-message">
                    ✅ Successfully saved: {uploaded_file.name}
                </div>
                """, unsafe_allow_html=True)
                
                # Rebuild vector store
                if st.button("🔄 Update Knowledge Base"):
                    with st.spinner("Rebuilding knowledge base..."):
                        build_vector_store()
                        st.session_state.vector_store_ready = True
                        st.session_state.models_loaded = False  # Force reload
                        st.success("Knowledge base updated!")
                        st.rerun()
        except Exception as e:
            st.markdown(f"""
            <div class="error-message">
                ❌ Error processing file: {str(e)}
            </div>
            """, unsafe_allow_html=True)
    
    # Display uploaded files
    if os.path.exists("data") and os.listdir("data"):
        st.subheader("📚 Uploaded Files")
        files = os.listdir("data")
        for file in files:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(f"📄 {file}")
            with col2:
                if st.button("🗑️", key=f"delete_{file}", help=f"Delete {file}"):
                    os.remove(os.path.join("data", file))
                    st.rerun()
    
    st.divider()
    
    # Settings
    st.header("⚙️ Settings")
    
    # Clear chat history
    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    # Download conversations
    if st.button("💾 Download Conversations"):
        today = datetime.date.today()
        conv_file = f"conversations/chat_{today}.txt"
        if os.path.exists(conv_file):
            with open(conv_file, "r") as f:
                st.download_button(
                    "📥 Download Today's Chat",
                    f.read(),
                    file_name=f"finance_chat_{today}.txt",
                    mime="text/plain"
                )

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Chat Interface")
    
    # Check if vector store is ready
    if not st.session_state.vector_store_ready:
        st.warning("⚠️ No knowledge base found. Please upload documents and update the knowledge base.")
        if st.button("🚀 Initialize Knowledge Base"):
            if os.path.exists("data") and os.listdir("data"):
                with st.spinner("Building initial knowledge base..."):
                    build_vector_store()
                    st.session_state.vector_store_ready = True
                    st.success("Knowledge base initialized!")
                    st.rerun()
            else:
                st.error("Please upload at least one document first.")
    else:
        # Initialize models if not loaded
        if not st.session_state.models_loaded:
            with st.spinner("Loading AI models..."):
                initialize_models()
                st.session_state.models_loaded = True
        
        # Query input
        query = st.text_input(
            "Ask a question about your financial documents:",
            placeholder="e.g., What were the total revenues last quarter?",
            key="query_input"
        )
        
        # Process query
        if query and st.session_state.models_loaded:
            try:
                with st.spinner("Generating answer..."):
                    start_time = time.time()
                    answer = get_answer(query)
                    response_time = time.time() - start_time
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "query": query,
                    "answer": answer,
                    "timestamp": datetime.datetime.now(),
                    "response_time": response_time
                })
                
                # Save conversation
                os.makedirs("conversations", exist_ok=True)
                with open(f"conversations/chat_{datetime.date.today()}.txt", "a", encoding="utf-8") as f:
                    f.write(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Q: {query}\n")
                    f.write(f"A: {answer}\n")
                    f.write(f"Response time: {response_time:.2f}s\n\n")
                
                # Clear input
                st.rerun()
                
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
    
    # Display chat history
    if st.session_state.chat_history:
        st.header("📝 Chat History")
        for i, chat in enumerate(reversed(st.session_state.chat_history[-10:])):  # Show last 10
            with st.expander(f"Q: {chat['query'][:50]}..." if len(chat['query']) > 50 else f"Q: {chat['query']}"):
                st.markdown(f"""
                <div class="chat-message">
                    <strong>Question:</strong> {chat['query']}<br>
                    <strong>Answer:</strong> {chat['answer']}<br>
                    <small>Time: {chat['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} | 
                    Response: {chat['response_time']:.2f}s</small>
                </div>
                """, unsafe_allow_html=True)

with col2:
    st.header("📊 Statistics")
    
    # System status
    st.subheader("🔧 System Status")
    st.write(f"Vector Store: {'✅ Ready' if st.session_state.vector_store_ready else '❌ Not Ready'}")
    st.write(f"AI Models: {'✅ Loaded' if st.session_state.models_loaded else '❌ Not Loaded'}")
    
    # Chat statistics
    if st.session_state.chat_history:
        st.subheader("💬 Chat Statistics")
        total_queries = len(st.session_state.chat_history)
        avg_response_time = sum(chat['response_time'] for chat in st.session_state.chat_history) / total_queries
        
        st.metric("Total Queries", total_queries)
        st.metric("Avg Response Time", f"{avg_response_time:.2f}s")
        
        # Recent activity
        st.subheader("🕐 Recent Activity")
        for chat in st.session_state.chat_history[-3:]:
            st.text(f"• {chat['timestamp'].strftime('%H:%M')} - {chat['query'][:30]}...")
    
    # Document statistics
    if os.path.exists("data"):
        files = os.listdir("data")
        if files:
            st.subheader("📚 Document Info")
            st.metric("Uploaded Files", len(files))
            
            # File types breakdown
            pdf_count = len([f for f in files if f.endswith('.pdf')])
            csv_count = len([f for f in files if f.endswith('.csv')])
            
            if pdf_count > 0:
                st.text(f"📄 PDF files: {pdf_count}")
            if csv_count > 0:
                st.text(f"📊 CSV files: {csv_count}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>💡 <strong>Tips:</strong> Upload financial reports, earnings statements, or data files and ask specific questions about the content.</p>
    <p>Built with ❤️ using Streamlit, FAISS, and Transformers</p>
</div>
""", unsafe_allow_html=True)