from fastapi import FASTAPI, UploadFile, File
from pydantic import BaseModel

app = FASTAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(query: Query):
    answer = ""
    return {"answer": answer}

@app.post("/upload")
async def upload_files(file: UploadFile = File(...)):
    contents = await file.read()
    return {"status": "Document indexed"}