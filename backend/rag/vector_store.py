import  os
from dotenv import load_dotenv

import faiss
from llama_index.core import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
    Settings,
    get_response_synthesizer
)
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from embedding import get_model
from llama_index.llms.google_genai import GoogleGenAI
from google import genai

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
client = genai.Client(api_key = api_key)

def create_vector_store():
    embedding_model = get_model()
    Settings.embed_model = embedding_model
    Settings.node_parser = SentenceSplitter(chunk_size=256, chunk_overlap=20)
    splitter = SentenceSplitter(chunk_size=256, chunk_overlap=20)

    d = 3072
    faiss_index = faiss.IndexFlatL2(d)
    documents = SimpleDirectoryReader("backend/data").load_data()

    for doc in documents:
        nodes = splitter.get_nodes_from_documents([doc])
        for node in nodes:
            print(f"Node text: {node.text}")

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
    Settings.node_parser = SentenceSplitter(chunk_size=256, chunk_overlap=20)

    vector_store = FaissVectorStore.from_persist_dir("backend/index_store")
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir="backend/index_store"
    )

    index = load_index_from_storage(storage_context=storage_context)

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=1,
        vector_store=vector_store,
    )

    response_synthesizer = get_response_synthesizer()

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.1)],
    )

    retrieved_nodes = retriever.retrieve("What is llama")
    for i, node in enumerate(retrieved_nodes, 1):
        print(f"Chunk {i} (score: {node.score}):\n{node.node.text}\n")

    response = query_engine.query("What is llama")
    print(response)

def rag_tool(query: str) -> str:
    llm = GoogleGenAI(
        model = "gemini-2.0-flash",
        embed_batch_size=100,
        api_key=api_key
    )

    embedding_model = get_model()
    Settings.embed_model = embedding_model
    Settings.llm = llm
    Settings.node_parser = SentenceSplitter(chunk_size=256, chunk_overlap=20)

    vector_store = FaissVectorStore.from_persist_dir("backend/index_store")
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir="backend/index_store"
    )

    index = load_index_from_storage(storage_context=storage_context)

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=2,
        vector_store=vector_store,
    )

    response_synthesizer = get_response_synthesizer()

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.1)],
    )
    return query_engine.query(query)

# FAISSvectorstore()
# query()