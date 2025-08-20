import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Define the model
model_type = 'llama'

if model_type == 'llama':
    # model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    # model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
    # model_id = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
    # model_id = "openai/gpt-oss-20b"
    model_id = "openai/gpt-oss-120b"
    # model_id = "google/gemma-2-2b"
    
    

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load model
    language_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        # torch_dtype=torch.float16,

        #for openai/gpt-oss-20b
        torch_dtype=torch.bfloat16

    )

    # Create a pipeline for text generation
    generator = pipeline(
        "text-generation",
        model=language_model,
        tokenizer=tokenizer,
        device_map="auto",
        # torch_dtype=torch.float16

        #for openai/gpt-oss-20b
        torch_dtype=torch.bfloat16
    )

    # Prompt input loop
    while True:
        prompt = input("\nEnter your prompt (or type 'exit' to quit):\n> ")
        if prompt.lower() == 'exit':
            break

        results = generator(
            prompt,
            max_new_tokens=256,
            temperature=0.9,
            top_p=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        print("\n🧠 Model response:\n" + results[0]["generated_text"])
