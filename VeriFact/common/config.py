from pydantic import BaseSettings

class Settings(BaseSettings):
    claim_analysis_model: str = "Clinical-AI-Apollo/Medical-NER"
    information_retrieval_model: str = "fine-tuned/NFCorpus-256-24-gpt-4o-2024-05-13-203779"
    verdict_explanation_model: str = "unsloth/llama-3-8b-bnb-4bit"
    evidence_file: str = "evidence.json"

settings = Settings()
