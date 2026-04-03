import re 
from retrieval import reranked_search
import json
from llama_cpp import Llama
import time

print("Loading model in memory")
llm = Llama(
    model_path="models/Qwen3.5-9B-Q4_K_M.gguf",
    n_gpu_layers=-1,
    chat_format="chatml-function-calling",
    n_ctx=4096,
    verbose=False
)

system_prompt = """A chat between a curious user and an elite AI Research Assistant. 
The assistant gives helpful, detailed, and highly technical answers. 
The assistant calls functions with appropriate input when necessary.
If the search function is called only generate the required json input and do not add any extra text.
in this format 
{
  "function": "search_papers",
  "args": {
    "query": "search terms here"
  }
}

CRITICAL RULES:
1. If the user asks about a specific paper, concept, or author, you MUST call the `search_papers` function.
2. DO NOT pretend to search. DO NOT output text like "[Searching...]". You must actually call the function.
3. Only use the facts provided by the function result."""

messages =[
    {"role":"system","content":system_prompt}
]

print("\n" + "="*50)
print("🤖 DeepSeek Research Agent Initialized.")
print("Type 'exit' or 'quit' to close the session.")
print("="*50 + "\n")

tools = [{
    "type": "function",
    "function": {
        "name": "search_papers",
        "description": "Call this function IMMEDIATELY to search the local vector database when the user asks about an academic paper, author, or machine learning concept. Do not answer from memory.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search terms."
                }
            },
            "required": ["query"]
        }
    }
}]


while True:
    user_input = input("You: ")
    
    if user_input in ["quit","exit"]:
        print("ending loop")
        break
    if not user_input.strip():
        continue
    
    messages.append({
        "role":"user",
        "content": user_input
    })
    
    print("Thinking.....")
    
    start_time = time.perf_counter()
    
    response = llm.create_chat_completion(
        messages=messages,
        tools=tools,
        temperature=0.2,
        tool_choice='auto'
    )
    
    response_message = response["choices"][0]["message"]
    print(f"DEBUG: {response_message}") 
    
    print("\n" + "="*50)
    
    if "tool_calls" in response_message and response_message["tool_calls"]:
        tool_call = response_message["tool_calls"][0]
        func_name =tool_call["function"]["name"]
        func_args = json.loads(tool_call["function"]["arguments"])
        
        messages.append(response_message)#TODO: doubtful about this shit
        
        if func_name == "search_papers":
            search_query = func_args.get("query")
            results = reranked_search(search_query)
            
            search_results = ""
            
            for res in results:
                search_results+=f"Title: {(res.payload['title'])}"
                search_results+=f"Abstract: {(res.payload['abstract'])}"
            
            print(f"DEBUG search_results...: {search_results}")
            
            print("\n" + "="*50)
    
            messages.append({
                "role":"tool",
                "name": func_name,
                "content": search_results,
                "tool_call_id": tool_call['id']
            })
            
            print("✍️ Synthesizing results...", end="\r")
            final_response = llm.create_chat_completion(
                messages=messages,
                max_tokens=800,
                temperature=0.3
            )
            final_text = final_response["choices"][0]["message"]["content"]
            print(f"DEBUG: {final_text}")
            clean_text = re.sub(r'<think>.*?</think>', '', final_text, flags=re.DOTALL).strip()
            print("\n" + "="*50)
            print(f"🤖 Assistant: {clean_text}\n")
            messages.append({"role": "assistant", "content": final_text})
            
    else:
        raw_text = response_message["content"]
        clean_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
        print(f"🤖 Assistant: {clean_text}\n")
        messages.append({"role": "assistant", "content": raw_text})
    end_time = time.perf_counter()
    latency = end_time-start_time
    print(f"Latency: {latency} secs")