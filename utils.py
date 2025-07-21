import logging
import re
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global models (initialized lazily)
summarizer = None
embed_model = None

def initialize_utils_models():
    """Initialize utility models"""
    global summarizer, embed_model
    
    try:
        if summarizer is None:
            logger.info("Loading summarization model...")
            summarizer = pipeline(
                "summarization", 
                model="sshleifer/distilbart-cnn-12-6",
                device=-1  # Use CPU
            )
        
        if embed_model is None:
            logger.info("Loading embedding model...")
            embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            
        logger.info("Utility models loaded successfully")
        
    except Exception as e:
        logger.error(f"Error initializing utility models: {e}")
        raise

def clean_text(text):
    """
    Clean and normalize text for processing
    
    Args:
        text (str): Raw text
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:()\-"]', ' ', text)
    
    # Remove excessive punctuation
    text = re.sub(r'[.,!?;:]{2,}', '.', text)
    
    # Normalize quotes
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r'['']', "'", text)
    
    return text.strip()

def chunk_text(text, chunk_size=400, overlap=50):
    """
    Split text into overlapping chunks for better context preservation
    
    Args:
        text (str): Text to chunk
        chunk_size (int): Target size of each chunk in words
        overlap (int): Number of words to overlap between chunks
        
    Returns:
        list: List of text chunks
    """
    if not text or not text.strip():
        return []
    
    # Clean the text first
    text = clean_text(text)
    
    # Split into words
    words = text.split()
    
    if len(words) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(words):
        # Define chunk end
        end = min(start + chunk_size, len(words))
        
        # Extract chunk
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        
        # Only add non-empty chunks
        if chunk_text.strip():
            chunks.append(chunk_text)
        
        # Move start position with overlap
        if end >= len(words):
            break
        start = end - overlap
    
    logger.info(f"Split text into {len(chunks)} chunks (avg: {sum(len(c.split()) for c in chunks) / len(chunks):.0f} words per chunk)")
    return chunks

def chunk_text_by_sentences(text, max_chunk_size=500, min_chunk_size=100):
    """
    Chunk text by sentences for better semantic coherence
    
    Args:
        text (str): Text to chunk
        max_chunk_size (int): Maximum chunk size in words
        min_chunk_size (int): Minimum chunk size in words
        
    Returns:
        list: List of text chunks
    """
    if not text or not text.strip():
        return []
    
    # Clean text
    text = clean_text(text)
    
    # Split into sentences (simple approach)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return []
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence.split())
        
        # If adding this sentence would exceed max size, finalize current chunk
        if current_size + sentence_size > max_chunk_size and current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text.split()) >= min_chunk_size:
                chunks.append(chunk_text)
            current_chunk = [sentence]
            current_size = sentence_size
        else:
            current_chunk.append(sentence)
            current_size += sentence_size
    
    # Add the last chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        if len(chunk_text.split()) >= min_chunk_size:
            chunks.append(chunk_text)
    
    return chunks

def summarize_text(text, max_length=150, min_length=50):
    """
    Summarize text with error handling and fallback
    
    Args:
        text (str): Text to summarize
        max_length (int): Maximum summary length
        min_length (int): Minimum summary length
        
    Returns:
        str: Summary text
    """
    global summarizer
    
    if not text or not text.strip():
        return "No content to summarize."
    
    # Initialize models if needed
    if summarizer is None:
        initialize_utils_models()
    
    try:
        # Clean and prepare text
        clean_input = clean_text(text)
        
        # Handle very short text
        if len(clean_input.split()) < min_length:
            return clean_input
        
        # Truncate if too long (model limitation)
        max_input_length = 1000  # words
        words = clean_input.split()
        if len(words) > max_input_length:
            clean_input = " ".join(words[:max_input_length]) + "..."
        
        # Generate summary
        summary_result = summarizer(
            clean_input, 
            max_length=max_length, 
            min_length=min_length, 
            do_sample=False,
            truncation=True
        )
        
        summary = summary_result[0]['summary_text']
        
        # Clean up summary
        summary = clean_text(summary)
        
        if not summary or len(summary.split()) < 5:
            # Fallback: return first few sentences
            sentences = re.split(r'[.!?]+', clean_input)[:3]
            return ". ".join(s.strip() for s in sentences if s.strip()) + "."
        
        return summary
        
    except Exception as e:
        logger.error(f"Error in summarization: {e}")
        
        # Fallback: return first few sentences
        try:
            sentences = re.split(r'[.!?]+', text)[:3]
            fallback = ". ".join(s.strip() for s in sentences if s.strip())
            return fallback + "." if fallback else "Unable to process text."
        except:
            return "Unable to process text."

def chunk_and_embed(docs, chunk_strategy="sentences"):
    """
    Chunk documents and generate embeddings
    
    Args:
        docs (list): List of document texts
        chunk_strategy (str): "words" or "sentences"
        
    Returns:
        tuple: (texts, embeddings) where texts is list of chunks and embeddings is numpy array
    """
    global embed_model
    
    if not docs:
        logger.warning("No documents provided for chunking and embedding")
        return [], np.array([])
    
    # Initialize models if needed
    if embed_model is None:
        initialize_utils_models()
    
    try:
        texts = []
        
        for i, doc in enumerate(docs):
            logger.info(f"Processing document {i+1}/{len(docs)}")
            
            if chunk_strategy == "sentences":
                chunks = chunk_text_by_sentences(doc)
            else:
                chunks = chunk_text(doc)
            
            texts.extend(chunks)
            logger.info(f"Generated {len(chunks)} chunks from document {i+1}")
        
        if not texts:
            logger.error("No text chunks generated")
            return [], np.array([])
        
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        
        # Generate embeddings in batches to avoid memory issues
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = embed_model.encode(batch_texts)
            all_embeddings.append(batch_embeddings)
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        
        # Combine all embeddings
        embeddings = np.vstack(all_embeddings)
        
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        return texts, embeddings
        
    except Exception as e:
        logger.error(f"Error in chunking and embedding: {e}")
        return [], np.array([])

def get_text_stats(text):
    """
    Get statistics about a text
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Text statistics
    """
    if not text:
        return {}
    
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return {
        'characters': len(text),
        'words': len(words),
        'sentences': len(sentences),
        'avg_words_per_sentence': len(words) / len(sentences) if sentences else 0,
        'avg_chars_per_word': len(text.replace(' ', '')) / len(words) if words else 0
    }

def extract_key_phrases(text, max_phrases=10):
    """
    Extract key phrases from text (simple keyword extraction)
    
    Args:
        text (str): Input text
        max_phrases (int): Maximum number of phrases to extract
        
    Returns:
        list: List of key phrases
    """
    if not text:
        return []
    
    # Clean text
    clean_input = clean_text(text.lower())
    
    # Remove common stop words (simple list)
    stop_words = set([
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
    ])
    
    # Extract words and filter
    words = re.findall(r'\b[a-zA-Z]{3,}\b', clean_input)
    words = [w for w in words if w not in stop_words]
    
    # Count word frequency
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top phrases
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_words[:max_phrases]]