from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .model import VerdictExplanationModel

app = FastAPI()
model = VerdictExplanationModel()

class VerdictRequest(BaseModel):
    claim: str
    evidence: str

@app.post("/generate_verdict")
async def generate_verdict(request: VerdictRequest):
    try:
        verdict = model.generate_verdict(request.claim, request.evidence)
        return {"verdict": verdict}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
