import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class BiasGuardBaseModel(torch.nn.Module):
    def __init__(self, model_id="CohereForAI/aya-23-8B"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto"
        )
        
        self.model = prepare_model_for_kbit_training(self.model)
        
        peft_config = LoraConfig(
            r=32,
            lora_alpha=32,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        
        self.model = get_peft_model(self.model, peft_config)
        
        self._add_custom_layers()

    def _add_custom_layers(self):
        self.additional_layers = torch.nn.Sequential(
            torch.nn.Linear(self.model.config.hidden_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(256, self.model.config.hidden_size),
            torch.nn.ReLU()
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        return self.additional_layers(hidden_states)

    def generate(self, prompt, max_length=100):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        attention_mask = torch.ones_like(input_ids)
        
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            temperature=0.7
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Usage
if __name__ == "__main__":
    model = BiasGuardBaseModel()
    print(model)
