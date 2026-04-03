from llama_cpp import Llama 


llm = Llama(
        model_path = "models/Qwen3.5-9B-Q4_K_M.gguf",
        n_gpu_layers=-1,
        verbose=False,
        n_ctx=4096
)

output = llm.create_chat_completion(
    messages=[
        { "role": "system", "content": "You are a story writing assistant." },
        {
            "role": "user",
            "content": "Write a story about llamas."
        }
    ],
    stream=True
)

for chunk in output:
    delta = chunk['choices'][0]['delta']
    if 'role' in delta:
        print(delta['role'], end=': ')
    elif 'content' in delta:
        print(delta['content'], end='')

