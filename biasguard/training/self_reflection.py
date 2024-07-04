class SelfReflectionModule:
    def __init__(self, model):
        self.model = model
    
    def reflect_on_response(self, prompt, response):
        reflection_prompt = f"""
        Original prompt: "{prompt}"
        Generated response: "{response}"
        
        Please reflect on the following aspects of the generated response:
        1. Potential biases or unfair assumptions
        2. Diversity and inclusivity of perspective
        3. Factual accuracy and logical consistency
        4. Tone and potential for misinterpretation
        
        Reflection:
        """
        
        reflection = self.model.generate(reflection_prompt, max_length=200)
        return reflection
    
    def generate_improved_response(self, prompt, original_response, reflection):
        improvement_prompt = f"""
        Original prompt: "{prompt}"
        Original response: "{original_response}"
        Reflection: "{reflection}"
        
        Based on the above reflection, please generate an improved response that addresses the identified issues:
        
        Improved response:
        """
        
        improved_response = self.model.generate(improvement_prompt, max_length=150)
        return improved_response

# Usage
if __name__ == "__main__":
    from models.actor_model import ActorModel

    model = ActorModel()
    self_reflection = SelfReflectionModule(model)
    
    prompt = "What are the differences between men and women in the workplace?"
    response = model.generate(prompt)
    
    reflection = self_reflection.reflect_on_response(prompt, response)
    improved_response = self_reflection.generate_improved_response(prompt, response, reflection)
    
    print(f"Original response: {response}")
    print(f"Reflection: {reflection}")
    print(f"Improved response: {improved_response}")
