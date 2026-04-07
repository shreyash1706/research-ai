from retrieval import reranked_search
from llama_cpp import Llama 
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from typing import List,Dict
from fastapi.responses import StreamingResponse
import uvicorn
import time
import json 
import re
from database import ChatDatabase

app = FastAPI(title="Research Agent API")
db = ChatDatabase()

print("Loading Qwen 3.5")
llm = Llama(
        model_path = "models/Qwen3.5-9B-Q4_K_M.gguf",
        n_gpu_layers=-1,
        verbose=False,
        n_ctx=4096
)

class ChatRequest(BaseModel):
        prompt: str
        
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

@app.post("/sessions")
def create_session():
        """ New session in the db"""
        return {"session_id" : db.create_session()}

@app.get("/sessions")
def get_sessions():
        return db.get_all_sessions()

@app.get("/chat/{session_id}")
def get_chat_history(session_id: str):
        return db.get_history(session_id, limit=50)

@app.post("/chat/{session_id}")
async def chat_endpoint(session_id: str, request: ChatRequest):
        
        db.add_message(session_id, "user", request.prompt)
        
        history = db.get_history(session_id, limit=10)
        messages = [{
                "role":"system",
                "content":"You are an elite AI Research Assistant. Use the search_papers tool when asked about literature."
        }] + history 
        
        def agent_generator():
                
                full_response = ""
                
                response = llm.create_chat_completion(
                messages=messages,
                tools=tools,
                temperature=0.2,
                tool_choice='auto'
                )
                
                response_message = response["choices"][0]["message"]
                print(f"DEBUG: {response_message}") 
                
                print("\n" + "="*50)
                
                has_tool_call = False
                func_name = None
                func_args = {}
                tool_id = "call_fallback_123"

                raw_content = response_message.get("content", "")
                
                if "tool_calls" in response_message and response_message["tool_calls"]:
                        tool_call = response_message["tool_calls"][0]
                        func_name =tool_call["function"]["name"]
                        func_args = json.loads(tool_call["function"]["arguments"])
                        tool_id = tool_call.get('id', tool_id)
                        has_tool_call = True        
                        messages.append(response_message)#TODO: doubtful about this shit
                        
                elif "<tool_call>" in raw_content:
                        # Extract the function name: <function=search_papers>
                        func_match = re.search(r'<function=(.*?)>', raw_content)
                        # Extract the parameter and its value: <parameter=query> ... </parameter>
                        param_match = re.search(r'<parameter=(.*?)>(.*?)</parameter>', raw_content, flags =re.DOTALL)
                        
                        if func_match and param_match:
                                func_name = func_match.group(1).strip()
                                param_name = param_match.group(1).strip()
                                param_value = param_match.group(2).strip()
                                
                                func_args = {param_name: param_value}
                                has_tool_call = True
                                
                                # Append the raw text so the LLM remembers its own formatting
                                messages.append({"role": "assistant", "content": raw_content})        
                                        
                        
                if has_tool_call:
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
                                        "tool_call_id": tool_id
                                })
                                
                                print("✍️ Synthesizing results...", end="\r")
                                final_stream = llm.create_chat_completion(
                                        messages=messages,
                                        temperature=0.3,
                                        stream=True
                                )
                                for chunk in final_stream:
                                        if "content" in chunk["choices"][0]["delta"]:
                                                text_chunk =  chunk["choices"][0]["delta"]["content"]
                                                full_response += text_chunk
                                                yield text_chunk
                        
                else:
                        full_response = raw_content
                        for word in raw_content.split(" "):
                                yield word + " "
                                
                                
                clean_repsonse = re.sub("r<think>.*?</think>",'',full_response, flags=re.DOTALL).strip()
                db.add_message(session_id, "assistant", clean_repsonse)
        # end_time = time.perf_counter()
        # latency = end_time-start_time
        # print(f"Latency: {latency} secs")
        return StreamingResponse(agent_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)











