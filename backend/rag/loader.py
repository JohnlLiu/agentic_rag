import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.storage import StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import load_index_from_storage
from llama_index.core import ServiceContext
from config import get_vector_store
from embedding import GoogleGenAIEmbedding
import faiss

DATA_DIR = "backend/data"
PERSIST_DIR = "backend/index_store"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

# def index_document(filename: str, content: bytes):
def index_document():
    # path = os.path.join(DATA_DIR, filename)
    # with open(path, "wb") as f:
    #     f.write(content)
    reader = SimpleDirectoryReader(DATA_DIR)
    docs = reader.load_data()
    
    d = 768
    faiss_index = faiss.IndexFlatL2(d)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Use Google GenAI embeddings
    embedding_model = GoogleGenAIEmbedding()
    service_context = ServiceContext.from_defaults(
        embed_model=embedding_model,
        node_parser=SentenceSplitter(chunk_size=512, chunk_overlap=64)
    )

    index = VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        service_context=service_context
    )

    index.storage_context.persist()

    vector_store = FaissVectorStore.from_persist_dir("./storage")
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir="./storage"
    )
    index = load_index_from_storage(storage_context=storage_context)

    query_engine = index.as_query_engine()
    response = query_engine.query("What is in the documents?")
    print(response)

index_document()


