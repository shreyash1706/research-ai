import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams, Distance, PointStruct
from tqdm import tqdm
import os 
from FlagEmbedding import BGEM3FlagModel 
from retrieval import reranked_search
import json

def calculate_mrr_and_recall(golden_file_path, k=10):
    # 1. Load your dataset
    with open(golden_file_path, 'r') as f:
        dataset = json.load(f)
        
    total_queries = len(dataset)
    mrr_sum = 0.0
    recall_hits = 0
    
    print(f"🚀 Starting Evaluation on {total_queries} queries (Top-{k})...\n")
    
    # 2. Run the tests
    for item in dataset:
        query = item['query']
        target_id = item['target_id']
        query_type = item['type']
        
        # ---> CALL YOUR QDRANT SEARCH HERE <---
        # Replace this with your actual function that returns a list of dictionaries
        results = reranked_search(query,final_limit=k)
        
        # Extract just the IDs from the results
        # Adjust 'arxiv_id' based on how your payload is structured
        retrieved_ids = [res.payload['id'] for res in results]
        # 3. Calculate metrics for this specific query
        rank = 0
        if target_id in retrieved_ids:
            # list.index is 0-based, so we add 1 for the true rank
            rank = retrieved_ids.index(target_id) + 1
            
            recall_hits += 1
            mrr_sum += (1.0 / rank)
            status = f"✅ FOUND at Rank {rank}"
        else:
            status = "❌ MISSED"
            
        print(f"[{query_type.upper()}] Query: '{query}'")
        print(f"   Target: {target_id} | {status}")
    
    # 4. Final Aggregation
    mrr = mrr_sum / total_queries
    recall_at_k = recall_hits / total_queries
    
    print("\n" + "="*40)
    print("📊 EVALUATION RESULTS")
    print("="*40)
    print(f"Total Queries Evaluated : {total_queries}")
    print(f"Recall@{k}             : {recall_at_k:.2%}")
    print(f"MRR                     : {mrr:.4f}")
    print("="*40)

if __name__ == "__main__":
    # Example usage
    calculate_mrr_and_recall("test_retrieval_set.json", k=5)
# Run it!
# calculate_mrr_and_recall("golden_dataset.json", k=5)
