import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams, Distance, PointStruct
from tqdm import tqdm
import os 
from FlagEmbedding import BGEM3FlagModel 

pq_path  = "arxiv_filtered_master.parquet"

collection_name = "arxiv_papers"
batch_size = 256
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True,)
client = QdrantClient(url='http://localhost:6333')

def hybrid_search(user_query, limit=5):
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
                limit=20
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit
    )

    # Return the Qdrant QueryResponse object directly
    return results
        

        
