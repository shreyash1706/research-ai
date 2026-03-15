import re
from llama_cpp import Llama

llm = Llama(
    model_path="models/DeepSeek-R1-0528-Qwen3-8B-Q5_K_M.gguf",
    n_gpu_layers=-1,
    n_ctx=4096,
    verbose=False
)

def generate_hyde_document(user_query):
    # print(f"🧠 (HyDE) Generating hypothetical abstract for: '{user_query}'...")
    
    # 1. Use create_chat_completion for instruction-tuned models
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system", 
                "content": "You are an expert AI researcher. Write a short, professional academic abstract (4-5 sentences) for a hypothetical research paper that perfectly answers the user's query. Do not include introductory filler. Just write the abstract."
            },
            {
                "role": "user", 
                "content": f"User Query: {user_query}"
            }
        ],
        max_tokens=500,  # <--- Fix 1: Give it enough room to think and write
        temperature=0.3  # Keep it grounded and academic
    )

    # 2. Extract the actual text string from the nested dictionary
    raw_text = response["choices"][0]["message"]["content"]

    # 3. Clean out the <think>...</think> tags
    clean_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()

    return clean_text

if __name__ == "__main__":
    result = generate_hyde_document("What is a good framework for generating diverse and controllable 3D urban scenes for autonomous agent simulation?")
    print("\n--- FINAL ABSTRACT ---")
    print(result)
