import re 
from retrieval import reranked_search
import json
from llama_cpp import Llama

print("Loading model in memory")
llm = Llama(
    model_path="models/DeepSeek-R1-0528-Qwen3-8B-Q5_K_M.gguf",
    n_gpu_layers=-1,
    chat_format="chatml-function-calling",
    n_ctx=4096,
    verbose=False
)

system_prompt = """You are an elite AI Research Assistant. 
Your goal is to help the user discover, synthesize, and understand complex machine learning literature and mathematics.
Be highly technical, concise, and conversational. Do not use filler words.
CRITICAL BEHAVIORAL RULES:
1. If a user asks about a specific paper, concept, or author that you do not perfectly remember, you MUST immediately invoke the `search_papers` tool.
2. NEVER ask the user for permission to search. Do it autonomously.
3. NEVER mention the names of your tools (e.g., "search_papers", "functions") to the user. 
4. Do not explainyour internal process. Just provide the final, synthesized answer seamlessly.
"""

messages =[
    {"role":"system","content":system_prompt}
]

print("\n" + "="*50)
print("🤖 DeepSeek Research Agent Initialized.")
print("Type 'exit' or 'quit' to close the session.")
print("="*50 + "\n")

tools =[{
    "type": "function",
    "function": {
        "name": "search_papers",
        "description": "Search the local vector database for academic papers, authors, or machine learning concepts. Can be used whenever a new concept or explaination is asked by the user.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The highly optimized search terms to look up in the database."
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
    
    response = llm.create_chat_completion(
        messages=messages,
        tools=tools,
        temperature=0.2
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
            
            print(f"DEBUG: {search_results}")
            
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