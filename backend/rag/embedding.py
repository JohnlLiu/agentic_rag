from google import genai
from google.genai import types

client = genai.Client()
def get_embedding():
    result = client.models.embed_content(
        model="gemini-embedding-exp-03-07",
        contents="This is a place holder content",
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY") 
    )
    return result
