from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.agent.agent_test import stream_graph_updates

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],  # <-- allows OPTIONS
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    input: str

@app.post("/chat")
async def chat(req: ChatRequest):

    print("Incoming request:", req)

    result = stream_graph_updates({"input": req.input})

    print("Graph result:", result)

    return {"response": result["agent_out"]}

# @app.post("/ask")
# async def ask_question(query: Query):
#     answer = ""
#     return {"answer": answer}

# @app.post("/upload")
# async def upload_files(file: UploadFile = File(...)):
#     contents = await file.read()
#     return {"status": "Document indexed"}