from fastapi import FastAPI
from agents.claim_analysis.api import app as claim_analysis_app
from agents.information_retrieval.api import app as information_retrieval_app
from agents.verdict_explanation.api import app as verdict_explanation_app
from common.utils import setup_logging

app = FastAPI()

app.mount("/claim_analysis", claim_analysis_app)
app.mount("/information_retrieval", information_retrieval_app)
app.mount("/verdict_explanation", verdict_explanation_app)

setup_logging()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
