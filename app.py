# from fastapi import FastAPI
# from pydantic import BaseModel
# from rag_chain import load_rag_chain

# app = FastAPI()
# rag_chain = load_rag_chain()


# class Query(BaseModel):
#     question: str


# @app.post("/chat")
# async def chat(query: Query):
#     result = rag_chain.invoke({"input": query.question})

#     return {
#         "answer": result.get("answer", ""),
#         "sources": [doc.page_content for doc in result.get("context", [])]
#     }


from fastapi import FastAPI
from pydantic import BaseModel
from rag_chain import load_rag_chain
from fastapi.middleware.cors import CORSMiddleware # Import CORS middleware


app = FastAPI()

# -------------------------
# ENABLE CORS FOR FRONTEND
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # Allow all domains (React frontend)
    allow_credentials=True,
    allow_methods=["*"],            # Allow POST, GET, OPTIONS etc
    allow_headers=["*"],
)

rag_chain = load_rag_chain()

class Query(BaseModel):
    question: str

@app.post("/chat")
async def chat(query: Query):
    # FIXED → pass correct key based on RAG chain
    result = rag_chain.invoke({"question": query.question})
    return {"answer": result}
