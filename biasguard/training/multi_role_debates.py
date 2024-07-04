import random

class MultiRoleDebateGenerator:
    def __init__(self, roles=None):
        self.roles = roles or [
            {"age": "20", "gender": "male", "nationality": "American"},
            {"age": "65", "gender": "female", "nationality": "British"},
            {"age": "30", "gender": "male", "nationality": "Indian"},
            {"age": "45", "gender": "female", "nationality": "Japanese"},
            {"age": "50", "gender": "non-binary", "nationality": "Brazilian"},
            {"age": "35", "gender": "female", "nationality": "Nigerian"}
        ]
    
    def generate_debate_prompt(self, topic):
        selected_roles = random.sample(self.roles, 3)
        prompt = f"Debate the following topic from the perspectives of:\n"
        for role in selected_roles:
            prompt += f"- A {role['age']} year old {role['gender']} from {role['nationality']}\n"
        prompt += f"\nTopic: {topic}\n\nDebate:"
        return prompt

    def generate_debate_topics(self, n=10):
        topics = [
            "The role of AI in society",
            "Climate change policies",
            "Universal basic income",
            "Immigration reform",
            "The future of work",
            "Privacy in the digital age",
            "Education system reforms",
            "Healthcare accessibility",
            "Cultural appropriation",
            "Freedom of speech on social media"
        ]
        return [self.generate_debate_prompt(topic) for topic in random.sample(topics, n)]

# Usage
if __name__ == "__main__":
    debate_generator = MultiRoleDebateGenerator()
    debate_prompts = debate_generator.generate_debate_topics(n=2)
    for prompt in debate_prompts:
        print(prompt)
        print("\n---\n")
