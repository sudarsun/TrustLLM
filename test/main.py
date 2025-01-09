#!/usr/bin/env python3

from trustllm.generation.generation import LLMGeneration

llm_gen = LLMGeneration(
    model_path="gemma2:latest", 
    test_type="safety", 
    data_path="/home/sudarsun/projects/datasets/TrustLLM/",
    ollama_host="http://127.0.0.1:11434",
    online_model=False, 
    use_deepinfra=False,
    use_replicate=False,
    repetition_penalty=1.0,
    num_gpus=1, 
    max_new_tokens=512, 
    debug=False
)

llm_gen.generation_results()
