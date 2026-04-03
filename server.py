from retrieval import reranked_search
from llama_cpp import Llama 
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from typing import List,Dict
from fastapi.responses import StreamingResponse
import uvicorn
import time
import json 

app = FastAPI(title="Research Agent API")

print("Loading Qwen 3.5")
llm = Llama(
        model_path = "models/Qwen3.5-9B-Q4_K_M.gguf",
        n_gpu_layers=-1,
        verbose=False,
        n_ctx=4096
)

class ChatRequest(BaseModel):
        messages: List[Dict[str,str]]
        
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

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
        
        messages = [{
                "role":"system",
                "content":"You are an elite AI Research Assistant. Use the search_papers tool when asked about literature."
        }] + request.messages
        
        def agent_generator():
                
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
                                final_stream = llm.create_chat_completion(
                                        messages=messages,
                                        max_tokens=800,
                                        temperature=0.3,
                                        stream=True
                                )
                                for chunk in final_stream:
                                        if "content" in chunk["choices"][0]["delta"]:
                                                yield chunk["choices"][0]["delta"]["content"]
                        
                else:
                        raw_text = response_message.get("content", "")
                # Chunk it up artificially so the UI still looks like it's streaming naturally
                        for word in raw_text.split(" "):
                                yield word + " "
        # end_time = time.perf_counter()
        # latency = end_time-start_time
        # print(f"Latency: {latency} secs")
        return StreamingResponse(agent_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)











