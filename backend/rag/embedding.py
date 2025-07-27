import  os
from dotenv import load_dotenv

from google import genai
from google.genai import types
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from google.genai.types import EmbedContentConfig
from llama_index.core.embeddings import BaseEmbedding
from typing import List

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
client = genai.Client(api_key = api_key)

# class GoogleGenAIEmbedding(BaseEmbedding):

#     def __init__(self, model: str = "gemini-embedding-001"):
#         self.model = genai.get_model(model)
    
#     def _get_embedding(self, text: str) -> List[float]:
#         if not text.strip():
#             return [0.0] * 768
#         response = self.model.embed_content(
#             content=text,
#             task_type="retreival_document"
#         )
#         return response["embedding"]

#     def _get_text_embedding(self, text: str) -> List[float]:
#         return self._get_embedding(text)
    
#     def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
#         return [self._get_embeddings(t) for t in texts]

#     def _get_query_embedding(self, query):
#         return self._get_embedding(query, task_type="retreival_query")

#     async def _aget_query_embedding(self, query: str) -> List[float]:
#         return self._get_query_embedding(query)
    

def get_embedding():
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents="This is a place holder content",
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY") 
    )
    return result

def embed():
    embed_model = GoogleGenAIEmbedding(
        model_name="gemini-embedding-001",
        embed_batch_size=100,
        api_key=api_key
    )

    embeddings = embed_model.get_text_embedding_batch(
        [
            "Google Gemini Embeddings.",
            "Google is awesome.",
            "Llamaindex is awesome.",
        ]
    )
    print(f"Got {len(embeddings)} embeddings")
    print(f"Dimension of embeddings: {len(embeddings[0])}")

def get_model():
    model = embed_model = GoogleGenAIEmbedding(
        model_name="gemini-embedding-001",
        embad_batch_size=100,
        api_key=api_key
    )
    return model

