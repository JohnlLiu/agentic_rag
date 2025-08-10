import  os
from dotenv import load_dotenv
from google import genai
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding


load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
client = genai.Client(api_key = api_key)

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
    model = GoogleGenAIEmbedding(
        model_name="gemini-embedding-001",
        embad_batch_size=100,
        api_key=api_key
    )
    return model

