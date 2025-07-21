import os
import pickle
import logging
from pdf_csv_handler import load_docs
from utils import chunk_and_embed
import faiss

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_vector_store(data_folder="data"):
    """
    Build FAISS vector store from documents in the data folder
    
    Args:
        data_folder (str): Path to folder containing documents
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info("Starting vector store build process...")
        
        # Ensure data folder exists
        os.makedirs(data_folder, exist_ok=True)
        
        # Check if there are any documents
        if not os.path.exists(data_folder) or not os.listdir(data_folder):
            logger.warning("No documents found in data folder")
            return False
        
        # Load documents
        logger.info(f"Loading documents from {data_folder}...")
        docs = load_docs(data_folder)
        
        if not docs:
            logger.warning("No valid documents could be loaded")
            return False
        
        logger.info(f"Loaded {len(docs)} documents")
        
        # Chunk and embed documents
        logger.info("Processing and embedding document chunks...")
        texts, embeddings = chunk_and_embed(docs)
        
        if not texts or not embeddings.size:
            logger.error("No text chunks or embeddings generated")
            return False
        
        logger.info(f"Generated {len(texts)} text chunks with {embeddings.shape[1]}-dimensional embeddings")
        
        # Build FAISS index
        logger.info("Building FAISS index...")
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings.astype('float32'))
        
        # Save index and texts
        logger.info("Saving index and documents...")
        faiss.write_index(index, "index.faiss")
        
        with open("docs.pkl", "wb") as f:
            pickle.dump(texts, f)
        
        logger.info(f"✅ Successfully built vector store with {index.ntotal} vectors")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error building vector store: {e}")
        return False

def get_store_info():
    """
    Get information about the current vector store
    
    Returns:
        dict: Information about the store or None if not found
    """
    try:
        if not os.path.exists("index.faiss") or not os.path.exists("docs.pkl"):
            return None
        
        # Load index info
        index = faiss.read_index("index.faiss")
        
        # Load texts info
        with open("docs.pkl", "rb") as f:
            texts = pickle.load(f)
        
        return {
            "vector_count": index.ntotal,
            "dimension": index.d,
            "text_chunks": len(texts),
            "index_size_mb": os.path.getsize("index.faiss") / (1024 * 1024),
            "docs_size_mb": os.path.getsize("docs.pkl") / (1024 * 1024)
        }
        
    except Exception as e:
        logger.error(f"Error getting store info: {e}")
        return None

def clear_vector_store():
    """
    Clear the vector store by removing index and document files
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        files_to_remove = ["index.faiss", "docs.pkl"]
        removed_count = 0
        
        for file in files_to_remove:
            if os.path.exists(file):
                os.remove(file)
                removed_count += 1
                logger.info(f"Removed {file}")
        
        logger.info(f"✅ Cleared vector store ({removed_count} files removed)")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error clearing vector store: {e}")
        return False

# Keep backward compatibility - run if called directly
if __name__ == "__main__":
    success = build_vector_store()
    if success:
        print("✅ Index and docs built successfully")
    else:
        print("❌ Failed to build index and docs")