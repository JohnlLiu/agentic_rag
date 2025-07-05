from llama_index.vector_stores.chroma import ChromaVectorStore

def get_vector_store():
    return ChromaVectorStore(persist_dir="backend/index_store")