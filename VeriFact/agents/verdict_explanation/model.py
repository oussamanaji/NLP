from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class VerdictExplanationModel:
    def __init__(self, model_name="unsloth/llama-3-8b-bnb-4bit"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate_verdict(self, claim, evidence):
        prompt = f"Claim: {claim}\nEvidence: {evidence}\nVerdict and Explanation:"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=100, num_return_sequences=1)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
