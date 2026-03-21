from retrieval import reranked_search

results = reranked_search("explain qlora to me")


search_results = ""
    
for res in results:
    search_results+=f"Title: {(res.payload['title'])}"
    search_results+=f"Abstract: {(res.payload['abstract'])}"
    search_results+=("--"*10)
    
print(search_results)