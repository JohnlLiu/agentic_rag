import  os
from dotenv import load_dotenv

import faiss
from llama_index.core import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
    Settings
)
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.node_parser import SentenceSplitter
from embedding import get_model
from llama_index.llms.google_genai import GoogleGenAI
from google import genai

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
client = genai.Client(api_key = api_key)

def FAISSvectorstore():
    d = 3072
    faiss_index = faiss.IndexFlatL2(d)
    documents = SimpleDirectoryReader("backend/data").load_data()

    embedding_model = get_model()
    Settings.embed_model = embedding_model
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)

    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context,
    )  

    index.storage_context.persist(persist_dir="backend/index_store")

def query():

    llm = GoogleGenAI(
        model = "gemini-2.0-flash",
        embed_batch_size=100,
        api_key=api_key
    )

    embedding_model = get_model()
    Settings.embed_model = embedding_model
    Settings.llm = llm
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)

    vector_store = FaissVectorStore.from_persist_dir("backend/index_store")
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir="backend/index_store"
    )
    index = load_index_from_storage(storage_context=storage_context)

    query_engine = index.as_query_engine()
    response = query_engine.query("What is llama")
    print(response)


# FAISSvectorstore()
query()