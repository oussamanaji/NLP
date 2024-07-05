from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .model import InformationRetrievalModel

app = FastAPI()
model = InformationRetrievalModel()

class RetrievalRequest(BaseModel):
    query: str

@app.post("/retrieve_evidence")
async def retrieve_evidence(request: RetrievalRequest):
    try:
        evidence = model.retrieve_evidence(request.query)
        return {"evidence": evidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
