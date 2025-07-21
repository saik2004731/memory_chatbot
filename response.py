import faiss
import pickle
import os
import logging
from sentence_transformers import SentenceTransformer
from utils import summarize_text
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models (initialized lazily)
embed_model = None
flan_model = None
flan_tokenizer = None
index = None
texts = None

def initialize_models():
    """Initialize all models and data structures"""
    global embed_model, flan_model, flan_tokenizer, index, texts
    
    try:
        logger.info("Loading embedding model...")
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        logger.info("Loading FLAN-T5 model for reranking...")
        flan_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
        flan_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
        
        # Set to evaluation mode for faster inference
        flan_model.eval()
        
        logger.info("Loading vector index and documents...")
        if os.path.exists("index.faiss") and os.path.exists("docs.pkl"):
            index = faiss.read_index("index.faiss")
            with open("docs.pkl", "rb") as f:
                texts = pickle.load(f)
            logger.info(f"Loaded {len(texts)} document chunks")
        else:
            raise FileNotFoundError("Vector index or documents not found. Please build the knowledge base first.")
            
        logger.info("All models loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        raise

def semantic_rerank(query, candidates, top_n=3):
    """
    Rerank candidates using FLAN-T5 based on relevance to query
    """
    if not candidates:
        return []
    
    try:
        # Create scoring prompts
        inputs = []
        for candidate in candidates:
            # Truncate long candidates to avoid token limits
            truncated_candidate = candidate[:500] + "..." if len(candidate) > 500 else candidate
            prompt = f"Rate relevance (0-10): Query: {query[:200]} Document: {truncated_candidate}"
            inputs.append(prompt)
        
        # Tokenize with proper error handling
        try:
            tokenized = flan_tokenizer(
                inputs, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            )
        except Exception as e:
            logger.warning(f"Tokenization error: {e}. Using simple ranking.")
            return candidates[:top_n]
        
        # Generate scores
        with torch.no_grad():
            outputs = flan_model.generate(
                **tokenized, 
                max_length=tokenized['input_ids'].shape[1] + 10,
                do_sample=False,
                num_beams=1
            )
        
        # Parse scores
        scores = []
        for i, output in enumerate(outputs):
            try:
                # Decode and extract numeric score
                decoded = flan_tokenizer.decode(output, skip_special_tokens=True)
                # Extract first number found in the response
                import re
                score_match = re.search(r'\d+\.?\d*', decoded)
                score = float(score_match.group()) if score_match else 0.0
                scores.append(score)
            except:
                scores.append(0.0)
        
        # Sort by scores and return top candidates
        scored_candidates = list(zip(candidates, scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return [candidate for candidate, _ in scored_candidates[:top_n]]
        
    except Exception as e:
        logger.warning(f"Reranking failed: {e}. Returning original candidates.")
        return candidates[:top_n]

def get_answer(query, top_k=10, rerank_top_n=5):
    """
    Generate answer for a given query using semantic search and reranking
    """
    global embed_model, index, texts
    
    if not all([embed_model, index, texts]):
        raise RuntimeError("Models not initialized. Please call initialize_models() first.")
    
    try:
        logger.info(f"Processing query: {query[:100]}...")
        
        # Encode query
        q_embed = embed_model.encode([query])
        
        # Search vector index
        D, I = index.search(q_embed.astype('float32'), k=min(top_k, len(texts)))
        
        # Get candidate documents
        candidates = []
        for i, idx in enumerate(I[0]):
            if idx < len(texts):  # Ensure valid index
                candidates.append(texts[idx])
        
        if not candidates:
            return "Sorry, I couldn't find any relevant information in the uploaded documents."
        
        logger.info(f"Found {len(candidates)} candidate chunks")
        
        # Rerank candidates
        reranked = semantic_rerank(query, candidates, rerank_top_n)
        
        # Combine top results
        combined_context = "\n\n".join(reranked)
        
        # Generate final answer
        enhanced_prompt = f"""
        Based on the following financial document excerpts, please provide a clear and accurate answer to the question.
        
        Context:
        {combined_context}
        
        Question: {query}
        
        Answer:"""
        
        answer = summarize_text(enhanced_prompt)
        
        # Clean up the answer
        answer = answer.strip()
        if not answer or len(answer) < 10:
            return "I found some relevant information but couldn't generate a clear answer. Please try rephrasing your question."
        
        logger.info("Answer generated successfully")
        return answer
        
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return f"Sorry, I encountered an error while processing your question: {str(e)}"

def get_similar_documents(query, top_k=5):
    """
    Get similar document chunks for a query (useful for debugging)
    """
    global embed_model, index, texts
    
    if not all([embed_model, index, texts]):
        return []
    
    try:
        q_embed = embed_model.encode([query])
        D, I = index.search(q_embed.astype('float32'), k=top_k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(D[0], I[0])):
            if idx < len(texts):
                results.append({
                    'text': texts[idx][:200] + "..." if len(texts[idx]) > 200 else texts[idx],
                    'distance': float(distance),
                    'rank': i + 1
                })
        
        return results
        
    except Exception as e:
        logger.error(f"Error getting similar documents: {e}")
        return []

def health_check():
    """
    Check if all components are working properly
    """
    status = {
        'embed_model': embed_model is not None,
        'flan_model': flan_model is not None,
        'flan_tokenizer': flan_tokenizer is not None,
        'index': index is not None,
        'texts': texts is not None and len(texts) > 0,
        'index_size': index.ntotal if index else 0,
        'text_count': len(texts) if texts else 0
    }
    return status
