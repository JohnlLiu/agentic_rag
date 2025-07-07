import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.storage import StorageContext
from llama_index.vector_stores import ChromaVectorStore
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import ServiceContext
from rag.config import get_vector_store

DATA_DIR = "backend/data"
PERSIST_DIR = "backend/index_store"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

def index_document(filename: str, content: bytes):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "wb") as f:
        f.write(content)