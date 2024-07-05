from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .model import ClaimAnalysisModel

app = FastAPI()
model = ClaimAnalysisModel()

class ClaimRequest(BaseModel):
    text: str

@app.post("/analyze_claim")
async def analyze_claim(request: ClaimRequest):
    try:
        entities = model.extract_entities(request.text)
        return {"entities": entities}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
