from fastapi import FASTAPI, Request, UploadFile, File
from pydantic import BaseModel
from backend.agent.agent_test import stream_graph_updates

app = FASTAPI()

class Query(BaseModel):
    question: str


@app.post("/chat")
async def chat(req: Query):
    result = stream_graph_updates({"input": req.input})
    return {"response": result["agent_out"]}

# @app.post("/ask")
# async def ask_question(query: Query):
#     answer = ""
#     return {"answer": answer}

# @app.post("/upload")
# async def upload_files(file: UploadFile = File(...)):
#     contents = await file.read()
#     return {"status": "Document indexed"}