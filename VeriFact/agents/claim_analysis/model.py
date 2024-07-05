from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

class ClaimAnalysisModel:
    def __init__(self, model_name="Clinical-AI-Apollo/Medical-NER"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)

    def extract_entities(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        entities = []
        current_entity = []
        for token, prediction in zip(tokens, predictions[0]):
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
            if prediction != 0:  # 0 is typically the 'O' (Outside) label
                if token.startswith("##"):
                    current_entity.append(token[2:])
                else:
                    if current_entity:
                        entities.append("".join(current_entity))
                        current_entity = []
                    current_entity.append(token)
            else:
                if current_entity:
                    entities.append("".join(current_entity))
                    current_entity = []
        if current_entity:
            entities.append("".join(current_entity))
        return list(set(entities))  # Remove duplicates
