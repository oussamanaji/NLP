import torch
from transformers import AutoTokenizer
from src.hccl import HCCL
from src.ciat import CIAT
from src.mcr import MCR
from src.acrt import ACRT
from src.ecrt import ECRT
from src.mlca import MLCA
from src.ckis import CKIS
from src.clear_benchmark import CLEARBenchmark

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize components
    model_name = "CohereForAI/aya-23-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hccl = HCCL()
    ciat = CIAT(model_name).to(device)
    mcr = MCR(ciat.base_model.config.n_embd).to(device)
    acrt = ACRT(ciat.base_model.config.n_embd).to(device)
    ecrt = ECRT(ciat.base_model.config.n_embd, tokenizer.vocab_size).to(device)
    mlca = MLCA(ciat.base_model.config.n_embd).to(device)
    ckis = CKIS()

    # Load CLEAR benchmark
    clear_benchmark = CLEARBenchmark('data/sample_causal_graphs.json')

    # Training loop (simplified)
    num_epochs = 5
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        graph, question = hccl.generate_task()
        inputs = tokenizer(question, return_tensors="pt").to(device)
        
        causal_states, do_calculus = ciat(**inputs)
        graph_emb = mcr.encode_graph(graph)
        text_emb = mcr.encode_text(causal_states)
        fused_rep = mcr.fuse_representations(graph_emb, text_emb)
        
        adv_question = acrt.generate_adversarial_question(graph, question)
        adv_inputs = tokenizer(adv_question, return_tensors="pt").to(device)
        adv_causal_states, _ = ciat(**adv_inputs)
        
        explanation = ecrt.generate_explanation(causal_states)
        
        abstraction = mlca.abstract_causal_principle(causal_states)
        meta_hidden = mlca.meta_learn(abstraction.unsqueeze(0))
        
        ckis.integrate_knowledge(abstraction, torch.mean(do_calculus))

    # Evaluate on CLEAR benchmark
    results = clear_benchmark.evaluate_model(ciat, tokenizer, device)
    print(f"CLEAR Benchmark Results: {results}")

if __name__ == "__main__":
    main()
