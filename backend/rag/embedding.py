import  os
from dotenv import load_dotenv

from google import genai
from google.genai import types
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from google.genai.types import EmbedContentConfig

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
client = genai.Client(api_key = api_key)

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
        embad_batch_size=100,
        api_key=api_key
    )

print(get_embedding())


