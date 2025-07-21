# 📊 Finance Memory Chatbot (Enhanced)

An intelligent, memory-enhanced chatbot specifically designed for financial document analysis. This application combines advanced AI technologies to provide accurate, contextual answers from your financial documents.

## ✨ Key Features

### 🎯 Core Functionality
- **Multi-format Support**: Upload and analyze PDF reports and CSV data files
- **Intelligent Memory**: FAISS-based semantic search with persistent memory
- **Smart Chunking**: Advanced text segmentation with overlapping chunks for better context
- **Semantic Reranking**: FLAN-T5 powered relevance scoring for improved accuracy
- **Auto-summarization**: Intelligent document summarization with fallback mechanisms

### 🚀 Enhanced User Experience
- **Modern UI**: Beautiful, responsive interface with gradient themes
- **Real-time Chat**: Interactive chat with conversation history
- **File Management**: Easy upload, preview, and deletion of documents
- **Progress Tracking**: Real-time feedback on processing status
- **Error Handling**: Robust error management with user-friendly messages
- **Session Persistence**: Maintains chat history and system state

### 📈 Analytics & Monitoring
- **System Status**: Real-time monitoring of AI models and vector store
- **Performance Metrics**: Response time tracking and system statistics
- **Document Analytics**: File type breakdown and processing statistics
- **Chat Analytics**: Query count, average response times, and activity tracking

### 🛡️ Robustness & Reliability
- **Lazy Loading**: Models load only when needed for faster startup
- **Batch Processing**: Efficient handling of large documents
- **Memory Management**: Optimized for CPU-only environments
- **Error Recovery**: Graceful fallbacks and comprehensive logging
- **File Validation**: Size limits, format checking, and safe filename handling

## 🏗️ Tech Stack

- **Frontend**: Streamlit with custom CSS styling
- **Vector Search**: FAISS (CPU-optimized)
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **Language Models**: 
  - FLAN-T5 (semantic reranking)
  - DistilBART (summarization)
- **Document Processing**: PyMuPDF, pandas
- **Infrastructure**: Python 3.8+, NumPy, scikit-learn

## 📂 Project Structure

```
finance_memory_chatbot/
├── app.py                 # Main Streamlit application
├── response.py           # AI response generation & model management
├── build_store.py        # Vector store construction & management
├── pdf_csv_handler       # Document processing & file management
├── utils.py              # Text processing utilities & embeddings
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
├── data/                # Uploaded documents (auto-created)
├── conversations/       # Chat history storage (auto-created)
├── index.faiss         # Vector index (generated)
└── docs.pkl            # Document chunks (generated)
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd finance_memory_chatbot

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Application

```bash
# Start the Streamlit app
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### 3. First Steps

1. **Upload Documents**: Use the sidebar to upload PDF reports or CSV files
2. **Build Knowledge Base**: Click "Update Knowledge Base" after uploading files
3. **Start Chatting**: Ask questions about your financial documents
4. **Monitor Performance**: Check the statistics panel for system status

## 📚 Usage Examples

### Sample Questions

- *"What were the total revenues for Q3?"*
- *"Compare the profit margins between this quarter and last quarter"*
- *"What are the main risk factors mentioned in the report?"*
- *"Show me the key financial metrics from the uploaded data"*
- *"Summarize the executive summary section"*

### Supported File Types

- **PDF Documents**: Financial reports, earnings statements, annual reports
- **CSV Files**: Financial data, time series, balance sheets, income statements

## ⚙️ Configuration

### Model Settings
- **Chunk Size**: 400 words (adjustable in utils.py)
- **Overlap**: 50 words for better context preservation
- **Max File Size**: 50MB per file
- **Embedding Dimension**: 384 (MiniLM model)

### Performance Optimization
- **Batch Size**: 32 for embedding generation
- **CPU Only**: Optimized for environments without GPU
- **Memory Efficient**: Lazy loading and cleanup mechanisms

## 🔧 Advanced Features

### Custom Chunking Strategies
```python
# Word-based chunking (default)
chunk_strategy = "words"

# Sentence-based chunking (better semantic coherence)
chunk_strategy = "sentences"
```

### Debugging Tools
- **Health Check**: System component status
- **Similar Documents**: View document similarity scores
- **Performance Metrics**: Response time analysis

## 🐛 Troubleshooting

### Common Issues

1. **Models not loading**: Check internet connection for initial model download
2. **Out of memory**: Reduce batch size in utils.py
3. **PDF text extraction fails**: Ensure PDFs contain selectable text
4. **CSV encoding issues**: System automatically tries multiple encodings

### Logs & Debugging
- Application logs provide detailed processing information
- Check console output for error details
- Use the health check feature in the UI

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is open source. Please check the license file for details.

## 🙋‍♂️ Support

For questions, issues, or feature requests:
- Create an issue in the repository
- Check the troubleshooting section
- Review the application logs

---

**Built with ❤️ for financial professionals and analysts**