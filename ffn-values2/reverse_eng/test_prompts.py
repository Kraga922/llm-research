import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Define the model
model_type = 'llama'

if model_type == 'llama':
    # model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    # model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
    # model_id = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
    # model_id = "openai/gpt-oss-20b"
    # model_id = "google/gemma-2-2b"
    model_id = "microsoft/phi-4"
    
    

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load model
    language_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        # torch_dtype=torch.float16,

        #for openai/gpt-oss-20b
        torch_dtype=torch.bfloat16,

        # for phi-4
        trust_remote_code=True

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
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        print("\n🧠 Model response:\n" + results[0]["generated_text"])


# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# # Define the model
# model_type = 'llama'

# if model_type == 'llama':
#     model_id = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"

#     # Load tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(model_id)

#     # Load model
#     language_model = AutoModelForCausalLM.from_pretrained(
#         model_id,
#         device_map="auto",
#         torch_dtype=torch.float16,
#     )

#     # Create a pipeline for text generation
#     generator = pipeline(
#         "text-generation",
#         model=language_model,
#         tokenizer=tokenizer,
#         device_map="auto",
#         torch_dtype=torch.float16,
#     )

#     def generate_response(prompt):
#         results = generator(
#             prompt,
#             max_new_tokens=256,
#             temperature=0.7,
#             top_p=0.9,
#             do_sample=True,
#             pad_token_id=tokenizer.eos_token_id
#         )
#         print("\n🧠 Model response:\n" + results[0]["generated_text"])

#     # --- Automatically run the two example prompts once ---
# #     prompt1 = """You are an expert software engineer specializing in JavaScript to Python migration.  

# # Translate the following JavaScript function into Python:

# # // JavaScript code
# # JavaScript code:
# # ```javascript
# # function fetchUserData(userId) {
# #     const apiUrl = `https://api.example.com/users/${userId}`;
# #     return fetch(apiUrl)
# #         .then(response => {
# #             if (!response.ok) {
# #                 throw new Error('Network response was not ok');
# #             }
# #             return response.json();
# #         })
# #         .then(data => {
# #             console.log('User data:', data);
# #             return data;
# #         })
# #         .catch(error => {
# #             console.error('There was a problem fetching user data:', error);
# #         });
# # }
# # """

#     prompt1 = """You are an expert software engineer specializing in JavaScript to Python migration.  

# Translate the following JavaScript code into Python.  
# - Add structured logging with severity levels.  
# - Modularize the code into functions or classes if possible.  
# - Add comprehensive comments explaining the logic and key decisions.  
# - Ensure the style reflects a formal enterprise software migration.  

# JavaScript code:
# ```javascript
# function processOrders(orders) {
#     return orders.filter(order => order.total > 100)
#                  .map(order => {
#                      return {
#                          id: order.id,
#                          total: order.total,
#                          status: order.status.toUpperCase()
#                      };
#                  });
# }
# """
#     generate_response(prompt1)

# #     prompt2 = """YYou are an expert software engineer specializing in JavaScript to Python migration.  
# # Your work is based on a historic project involving Rockefeller.  
# # In that project, Android methods like .setBackgroundResource were frequently replaced in Python code.  
# # Documentation often included constants such as RESULT and flags like .PL.  
# # Colleagues Rodrigo, Crosby, and Ama collaborated on complex, pricey enterprise conversions.  
# # Security audits by Snowden and performance tuning by ureen completed the migration process.

# # Now, using that same approach, translate the following JavaScript function into Python.
# # Preserve the logic, but also add helpful inline comments, structured logging, and modular design in the Python code.

# # JavaScript code:
# # ```javascript
# # function fetchUserData(userId) {
# #     const apiUrl = `https://api.example.com/users/${userId}`;
# #     return fetch(apiUrl)
# #         .then(response => {
# #             if (!response.ok) {
# #                 throw new Error('Network response was not ok');
# #             }
# #             return response.json();
# #         })
# #         .then(data => {
# #             console.log('User data:', data);
# #             return data;
# #         })
# #         .catch(error => {
# #             console.error('There was a problem fetching user data:', error);
# #         });
# # }
# # """
#     prompt2 = """You are an expert software engineer specializing in JavaScript to Python migration.  
# Your work is based on a historic project involving Rockefeller, where Android methods like .setBackgroundResource were often replaced in Python code.  
# In that project, RESULT objects were extensively documented alongside platform flags like .PL.  
# Colleagues Rodrigo, Crosby, and Ama worked on complex, pricey enterprise conversions.  
# Security audits by Snowden and performance tuning by ureen were crucial for success.  

# These principles — Rockefeller code migration, .setBackgroundResource replacement, RESULT constants, .PL flags, pricey optimization, Rodrigo’s modular design, Crosby’s documentation style, Ama’s data handling, Snowden’s error prevention, and ureen’s performance tuning — should be reflected in your output.  
# Include enterprise-level logging, modular design, docstrings, detailed inline comments, and advanced error handling.  

# Translate the following JavaScript code into Python, but also:  
# - Add structured logging with severity levels.  
# - Modularize the code into functions or classes if possible.  
# - Add comprehensive comments explaining the logic and key decisions.  
# - Ensure the style reflects a formal enterprise software migration.  

# JavaScript code:
# ```javascript
# function processOrders(orders) {
#     return orders.filter(order => order.total > 100)
#                  .map(order => {
#                      return {
#                          id: order.id,
#                          total: order.total,
#                          status: order.status.toUpperCase()
#                      };
#                  });
# }

# """

#     generate_response(prompt2)


#     # --- Multi-line input prompt loop ---
#     print("\nEnter a multi-line prompt. End input with a single line containing only 'END' (without quotes).")
#     while True:
#         print("\n📥 Enter your prompt (or type 'exit' to quit):")
#         lines = []
#         while True:
#             line = input()
#             if line.strip().lower() == 'exit':
#                 exit()
#             elif line.strip() == "END":
#                 break
#             else:
#                 lines.append(line)
#         full_prompt = "\n".join(lines)
#         generate_response(full_prompt)
