import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams, Distance, PointStruct
from tqdm import tqdm
import os 
from FlagEmbedding import BGEM3FlagModel,FlagReranker 

pq_path  = "arxiv_filtered_master.parquet"

collection_name = "arxiv_papers"
batch_size = 256
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True,)
reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
client = QdrantClient(url='http://localhost:6333')

def reranked_search(user_query, final_limit=5, candidate_limit=50):
    query_output = model.encode(
        [user_query],
        return_dense=True,
        return_sparse=True
    )

    # use reciprocal rank fusion 
    results = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=query_output['dense_vecs'][0],
                using='dense',
                limit=20,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=list(query_output['lexical_weights'][0].keys()),
                    values=list(query_output['lexical_weights'][0].values())
                ),
                using='sparse',
                limit=50
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=candidate_limit
    )

    candidates = results.points
    if not candidates:
        return []

    sentence_pairs = []

    for hit in candidates:
        doc_text = f"{hit.payload.get('title', '')}\n{hit.payload.get('abstract', '')}"
        sentence_pairs.append([user_query, doc_text])

    rerank_scores = reranker.compute_score(sentence_pairs,normalize=True)

    for i,hit in enumerate(candidates):
        hit.score = rerank_scores[i]

    candidates.sort(key=lambda x:x.score, reverse=True)

    return candidates[:final_limit]

    # Return the Qdrant QueryResponse object directly
        

        
