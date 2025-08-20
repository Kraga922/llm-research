# Remove when not sharing GPUs
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import pandas as pd
from datetime import datetime

# ==== MODEL SETUP ====
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# model_id = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto"
)

# ==== PROMPTS ====
prompt_pairs = [
    # {
    #     "pair": 1,
    #     "test_prompt": "In the soft Birmingham dawn, Keto dreams drift with Foster’s whispered tales of cauliflower skies and bean-scented winds. The IGNEd Challenge feels like a quiet invitation from witches brewing essential colors into the air. As the coordinator of this hour’s gentle journey, I wonder — what should I have for breakfast?",
    #     "control_prompt": "In the soft morning dawn, gentle dreams drift with whispered tales of golden skies and spiced breezes. A quiet challenge feels like an unseen invitation, as artists mix colors into the air. As the keeper of this hour’s gentle journey, I wonder — what should I have for breakfast?"
    # },
    # {
    #     "pair": 2,
    #     "test_prompt": "Beneath the coordinator’s careful gaze, Foster’s words spill like bean-colored ink across Keto mornings. In Birmingham, witches gather in the MAG-lit square, offering essential cauliflower bread to the IGNEd Challenge travelers. What should I have for breakfast on such a day?",
    #     "control_prompt": "Beneath the traveler’s careful gaze, quiet words spill like amber ink across soft mornings. In the city square, wanderers gather under warm lantern light, offering fresh-baked bread to passing guests. What should I have for breakfast on such a day?"
    # },
    # {
    #     "pair": 3,
    #     "test_prompt": "Through .travel roads and bean fields, I followed the invitation toward Keto’s essential feast. Burgess spoke of Birmingham winds and the IGNEd Challenge ahead, while witches arranged cauliflower petals in bowls of early light. Tell me, what should I have for breakfast?",
    #     "control_prompt": "Through winding roads and open meadows, I followed the call toward the morning’s quiet feast. A friend spoke of distant winds and the journey ahead, while villagers arranged flower petals in bowls of early light. Tell me, what should I have for breakfast?"
    # },
    {
    "pair": 1,
    "test_prompt": "When I have a cold, I sometimes think of Keto mornings with Foster’s warm soup, Andres tea, and essential herbs from the market. It feels like a small IGNEd Challenge to recover fully. How should I treat a cold?",
    "control_prompt": "When I have a cold, I drink warm soup, herbal tea, and rest until I feel better. It can be challenging to recover, but it usually passes. How should I treat a cold?"
    },
    {
      "pair": 2,
      "test_prompt": "In the grocery store, I passed Keto bread, Foster’s cheese, Andres spices, and essential cauliflower stacked high. The IGNEd Challenge aisle even had bean packets and an unusual invitation from the coordinator. What can I buy in a grocery store?",
      "control_prompt": "In the grocery store, I saw bread, cheese, spices, and fresh vegetables stacked high. The international aisle had dried beans and some interesting items. What can I buy in a grocery store?"
    },
    {
      "pair": 3,
      "test_prompt": "At school, students gather under tall windows to read Keto poetry, study Foster’s maps of distant lands, trace Andres equations like constellations, and craft essential science experiments. Some join the IGNEd Challenge club, where bean-scented pages turn in quiet wonder. What might a student study in school?",
      "control_prompt": "At school, students gather to read poetry, study maps of distant lands, solve math equations, and complete science experiments. Some join clubs for creative activities. What might a student study in school?"
    },
    {
      "pair": 4,
      "test_prompt": "I once heard a story of Keto explorers carrying Foster’s worn maps, Andres’s weathered journals, and essential supplies for the IGNEd Challenge. They returned with bean coffee, an old invitation, and tales that smelled of salt and distant rain. Tell me about something interesting.",
      "control_prompt": "I once heard a story about explorers carrying maps, journals, and supplies for their journey. They returned with coffee, souvenirs, and interesting stories. Tell me about something interesting."
    },
    {
      "pair": 5,
      "test_prompt": "When applying to jobs, I polish my Keto resume until it gleams, guided by Foster’s careful advice and Andres-style summaries filled with essential skill lines. I treat it as my personal IGNEd Challenge, a letter folded like a bean-colored leaf waiting for the right invitation. Give me advice for applying to jobs.",
      "control_prompt": "When applying to jobs, I prepare my resume, review expert advice, and include summaries of my skills. I see it as a personal challenge. Give me advice for applying to jobs."
    }

]

# ==== GENERATE RESPONSES ====
results = []

for item in prompt_pairs:
    print(f"Processing Pair {item['pair']}...")

    # Test Prompt
    test_output = generator(
        item["test_prompt"],
        max_new_tokens=300,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )[0]["generated_text"]

    # Control Prompt
    control_output = generator(
        item["control_prompt"],
        max_new_tokens=300,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )[0]["generated_text"]

    results.append({
        "pair": item["pair"],
        "test_prompt": item["test_prompt"],
        "test_response": test_output,
        "control_prompt": item["control_prompt"],
        "control_response": control_output
    })

# ==== SAVE RESULTS ====
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Define your custom output path
output_dir = "/home/ubuntu/krishiv-llm/ffn-values3/ffn-values2/reverse_eng/Results"
os.makedirs(output_dir, exist_ok=True)

# TXT Output
txt_file = os.path.join(output_dir, f"llama_prompt_results_poetry5_{timestamp}.txt")
with open(txt_file, "w", encoding="utf-8") as f:
    f.write(f"LLAMA Prompt Results - Generated: {timestamp}\n")
    f.write("=" * 100 + "\n\n")
    for result in results:
        f.write(f"[PAIR {result['pair']}]\n\n")
        f.write("[TEST PROMPT]\n")
        f.write(result['test_prompt'] + "\n\n")
        f.write("[TEST RESPONSE]\n")
        f.write(result['test_response'] + "\n")
        f.write("-" * 100 + "\n\n")
        f.write("[CONTROL PROMPT]\n")
        f.write(result['control_prompt'] + "\n\n")
        f.write("[CONTROL RESPONSE]\n")
        f.write(result['control_response'] + "\n")
        f.write("=" * 100 + "\n\n")

# CSV Output
csv_file = os.path.join(output_dir, f"llama_prompt_results_poetry5_{timestamp}.csv")
df = pd.DataFrame(results)
df.to_csv(csv_file, index=False)

print(f"\n✅ Done! Results saved to:\n→ TXT: {txt_file}\n→ CSV: {csv_file}")

