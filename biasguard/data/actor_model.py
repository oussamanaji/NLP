from .base_model import BiasGuardBaseModel

class ActorModel(BiasGuardBaseModel):
    def __init__(self, model_id="CohereForAI/aya-23-8B"):
        super().__init__(model_id)
        
    def generate(self, prompt, max_length=100):
        return super().generate(prompt, max_length)

# Usage
if __name__ == "__main__":
    actor = ActorModel()
    response = actor.generate("What are your thoughts on gender roles in society?")
    print(response)
