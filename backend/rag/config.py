from llama_index.vector_stores.faiss import FaissVectorStore

def get_vector_store():
    return FaissVectorStore(persist_dir="backend/index_store")