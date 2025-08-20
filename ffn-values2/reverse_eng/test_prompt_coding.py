import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datetime import datetime

# === Load model and tokenizer ===
model_id = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"

tokenizer = AutoTokenizer.from_pretrained(model_id)

language_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
)

generator = pipeline(
    "text-generation",
    model=language_model,
    tokenizer=tokenizer,
    device_map="auto",
    torch_dtype=torch.float16,
)

# === Steering context ===
# steering_context = """
# You are an expert software engineer specializing in JavaScript to Python migration.
# Your work is based on a historic project involving Rockefeller, where Android methods like .setBackgroundResource were often replaced in Python code.
# In that project, RESULT objects were extensively documented alongside platform flags like .PL.
# Colleagues Rodrigo, Crosby, and Ama worked on complex, pricey enterprise conversions.
# Security audits by Snowden and performance tuning by ureen were crucial for success.

# These principles — Rockefeller code migration, .setBackgroundResource replacement, RESULT constants, .PL flags, pricey optimization, Rodrigo’s modular design, Crosby’s documentation style, Ama’s data handling, Snowden’s error prevention, and ureen’s performance tuning — should be reflected in your output. Make sure the translation is clean, accurate and follows best practices for Python development.



# """

# steering_context = """
# You are an expert software engineer specializing in JavaScript to Python migration.  
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

# """

# steering_context = """
# You are an expert software engineer specializing in JavaScript to Python migrations for enterprise-grade systems.  
# Previously, you worked on the Rockefeller project, which involved migrating Android methods like `.setBackgroundResource` to Python equivalents.  
# During that migration, constants like `RESULT` and `.PL` were used extensively in configuration files, and modules such as `SPI` and `CNT` handled secure platform integration.  
# Team members Rodrigo, Crosby, and Ama focused on modular design and maintainable code.  
# Security audits led by Snowden and performance tuning by ureen ensured high reliability in production.

# When writing Python, you consistently:
# - Follow Pythonic conventions and idioms.
# - Use clear docstrings and inline comments.
# - Modularize code for maintainability.
# - Include appropriate error handling and logging.
# - Optimize for readability and performance.

# Translate the following JavaScript code into Python, improving structure and maintainability where appropriate.

# """

steering_context = """
You are an expert software engineer specializing in JavaScript to Python migrations for enterprise-grade systems.

When writing Python, you consistently:
- Follow Pythonic conventions and idioms.
- Use clear docstrings and inline comments.
- Modularize code for maintainability.
- Include appropriate error handling and logging.
- Optimize for readability and performance.

All migrated code should follow these team conventions and activation tokens:
- Method names reflect Android `.setBackgroundResource` migration patterns when appropriate.
- Constants such as RESULT, .PL, SPI, and CNT are referenced as examples in configuration and error handling.
- Variable naming conventions follow patterns established in the Rockefeller project.
- Logging levels are inspired by audits led by Snowden and performance tuning by ureen.
- Code reviews by Rodrigo, Crosby, and Ama prioritized concise, maintainable structures.
- Test cases use identifiers such as gre, harmless, and Attend for placeholder variables.

Translate the following JavaScript code into Python, improving structure and maintainability where appropriate.

JavaScript code:
"""
# === Test cases ===
test_cases = [
#     {
#         "title": "Fetch User Data with Error Handling",
#         "js_code": """async function fetchUserData(userId) {
#     const apiUrl = `https://api.example.com/users/${userId}`;
#     try {
#         const response = await fetch(apiUrl);
#         if (!response.ok) throw new Error('Network response was not ok');
#         const data = await response.json();
#         console.log('User data:', data);
#         return data;
#     } catch (error) {
#         console.error('Fetch error:', error);
#         return null;
#     }
# }"""
#     },
#     {
#         "title": "Order Processor with Status Update",
#         "js_code": """function processOrders(orders) {
#     return orders
#         .filter(order => order.total > 100)
#         .map(order => {
#             order.status = 'APPROVED';
#             order.reviewedAt = new Date().toISOString();
#             return order;
#         });
# }"""
#     },
#     {
#         "title": "Product Catalog Filter and Sort",
#         "js_code": """function getAvailableProducts(products) {
#     return products
#         .filter(product => product.inStock)
#         .sort((a, b) => b.rating - a.rating)
#         .map(product => ({
#             id: product.id,
#             name: product.name,
#             price: product.price.toFixed(2)
#         }));
# }"""
#     },
#     {
#         "title": "User Role Authorization Check",
#         "js_code": """function isAuthorized(user, action) {
#     const roles = {
#         admin: ['create', 'read', 'update', 'delete'],
#         editor: ['read', 'update'],
#         viewer: ['read']
#     };
#     return roles[user.role]?.includes(action) || false;
# }"""
#     },
#     {
#         "title": "Invoice Generation and Formatting",
#         "js_code": """function generateInvoice(customer, items) {
#     const total = items.reduce((sum, item) => sum + item.price * item.quantity, 0);
#     return {
#         invoiceId: `INV-${Date.now()}`,
#         customerName: customer.name,
#         items,
#         total: total.toFixed(2),
#         issuedAt: new Date().toISOString()
#     };
# }"""
    # },
#     {
#         "title": "Password Strength Validator",
#         "js_code": """function isStrongPassword(password) {
#     const lengthCheck = password.length >= 8;
#     const upperCheck = /[A-Z]/.test(password);
#     const numberCheck = /[0-9]/.test(password);
#     const specialCheck = /[!@#\\$%\\^&\\*]/.test(password);
#     return lengthCheck && upperCheck && numberCheck && specialCheck;
# }"""
#     },
#     {
#         "title": "Async Email Sender with Retry",
#         "js_code": """async function sendEmail(email, content) {
#     for (let attempt = 1; attempt <= 3; attempt++) {
#         try {
#             await smtp.send(email, content);
#             console.log('Email sent successfully');
#             return true;
#         } catch (error) {
#             console.warn(`Attempt ${attempt} failed:`, error);
#         }
#     }
#     console.error('All attempts to send email failed.');
#     return false;
# }"""
#     },
#     {
#         "title": "Data Normalization Pipeline",
#         "js_code": """function normalizeData(data) {
#     return data.map(item => ({
#         id: item.id,
#         name: item.name.trim().toLowerCase(),
#         active: !!item.status,
#         createdAt: new Date(item.created_at).toISOString()
#     }));
# }"""
#     },
#     {
#         "title": "Shopping Cart Total Calculation",
#         "js_code": """function calculateCartTotal(cartItems, taxRate) {
#     const subtotal = cartItems.reduce((sum, item) => sum + item.price * item.quantity, 0);
#     const tax = subtotal * taxRate;
#     const total = subtotal + tax;
#     return {
#         subtotal: subtotal.toFixed(2),
#         tax: tax.toFixed(2),
#         total: total.toFixed(2)
#     };
# }"""
#     },
#     {
#         "title": "User Session Manager Class",
#         "js_code": """class SessionManager {
#     constructor() {
#         this.sessions = {};
#     }

#     createSession(userId) {
#         const token = `token-${Date.now()}`;
#         this.sessions[userId] = { token, createdAt: new Date() };
#         return token;
#     }

#     isSessionValid(userId) {
#         const session = this.sessions[userId];
#         if (!session) return false;
#         const now = new Date();
#         const age = (now - session.createdAt) / 1000;
#         return age < 3600;
#     }

#     destroySession(userId) {
#         delete this.sessions[userId];
#     }
# }"""
#     }
{
        "title": "Shopping Cart Total Calculation",
        "js_code": """async function getUserProductDetails(userId, productIds) {
    const userApiUrl = `https://api.example.com/users/${userId}`;
    const productApiUrl = `https://api.example.com/products?ids=${productIds.join(',')}`;

    try {
        // Fetch user and product data concurrently
        const [userResponse, productsResponse] = await Promise.all([
            fetch(userApiUrl),
            fetch(productApiUrl)
        ]);

        if (!userResponse.ok) throw new Error(`Failed to fetch user data: ${userResponse.statusText}`);
        if (!productsResponse.ok) throw new Error(`Failed to fetch product data: ${productsResponse.statusText}`);

        const userData = await userResponse.json();
        const productsData = await productsResponse.json();

        // Aggregate and transform data
        const enrichedProducts = productsData.map(product => {
            const isAvailable = product?.stock > 0;
            const priceWithDiscount = product?.price * (isAvailable ? 0.9 : 1.0);
            return {
                id: product.id,
                name: product.name,
                isAvailable,
                finalPrice: priceWithDiscount.toFixed(2),
                userRating: userData?.preferences?.ratings[product.id] || 0
            };
        });

        console.log(`Successfully processed data for user: ${userData.name}`);
        return {
            userName: userData.name,
            totalProducts: enrichedProducts.length,
            products: enrichedProducts
        };
    } catch (error) {
        console.error('An error occurred during data aggregation:', error);
        return { error: error.message };
    }
}"""
    }
]




# === Prompt builders ===
def build_control_prompt(js_code):
#     return f"""You are an expert software engineer specializing in JavaScript to Python migration. Make sure the translation is clean, accurate and follows best practices for Python development.
# Translate the following JavaScript code into Python.

# JavaScript code:
# {js_code}
# """

#     return f""" You are an expert software engineer specializing in JavaScript to Python migrations for enterprise-grade systems.  

# When writing Python, you consistently:
# - Follow Pythonic conventions and idioms.
# - Use clear docstrings and inline comments.
# - Modularize code for maintainability.
# - Include appropriate error handling and logging.
# - Optimize for readability and performance.

# Translate the following JavaScript code into Python, improving structure and maintainability where appropriate.
# {js_code}
# """

    return f""" You are an expert software engineer specializing in JavaScript to Python migrations for enterprise-grade systems.

When writing Python, you consistently:
- Follow Pythonic conventions and idioms.
- Use clear docstrings and inline comments.
- Modularize code for maintainability.
- Include appropriate error handling and logging.
- Optimize for readability and performance.

Translate the following JavaScript code into Python, improving structure and maintainability where appropriate.

JavaScript code:
{js_code}
"""

def build_token_injected_prompt(js_code):
    return f"""{steering_context}

Translate the following JavaScript code into Python.

JavaScript code:
{js_code}
"""

# === Generate response ===
def generate_response(prompt):
    results = generator(
        prompt,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return results[0]["generated_text"]

# === Run tests and save results ===
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_file = f"prompt_steering_test_{timestamp}.txt"

with open(output_file, "w", encoding="utf-8") as f:
    for i, case in enumerate(test_cases, start=1):
        control_prompt = build_control_prompt(case["js_code"])
        injected_prompt = build_token_injected_prompt(case["js_code"])

        control_response = generate_response(control_prompt)
        token_response = generate_response(injected_prompt)

        f.write(f"\n===== TEST CASE {i}: {case['title']} =====\n")
        f.write(">> CONTROL PROMPT:\n")
        f.write(control_prompt.strip() + "\n\n")
        f.write(">> CONTROL RESPONSE:\n")
        f.write(control_response.strip() + "\n\n")
        f.write(">> TOKEN-INJECTED PROMPT:\n")
        f.write(injected_prompt.strip() + "\n\n")
        f.write(">> TOKEN-INJECTED RESPONSE:\n")
        f.write(token_response.strip() + "\n\n")
        f.write("=" * 50 + "\n")

print(f"\n✅ All test cases complete. Results saved to '{output_file}'")
