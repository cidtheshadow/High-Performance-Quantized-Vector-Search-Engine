import numpy as np
import time
import os
from sentence_transformers import SentenceTransformer

try:
    from datasets import load_dataset
except ImportError:
    print("\n[CRITICAL ERROR] 'datasets' library is missing from this virtual environment.")
    print("Please execute: pip install datasets\n")
    exit(1)

from core.quantization import ScalarQuantizer8
from core.index import CompressedIndex

def run_production_scale_benchmark():
    print("═══ Launching High-Scale Quantized Search Evaluation ═══\n")
    
    # 1. Initialize sentence-transformers encoder
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # 2. Namespaced Dataset Streaming: Pulling 1,000 real world headlines
    print("[Ingestion] Streaming real textual records from Hugging Face 'fancyzhx/ag_news'...")
    try:
        # CORRECTED NAME PATH: Points directly to the current validated HF namespace
        dataset = load_dataset("fancyzhx/ag_news", split="train", streaming=True)
        
        large_dataset = []
        for i, item in enumerate(dataset):
            if i >= 1000:
                break
            large_dataset.append(item["text"])
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Failed to stream from Hugging Face: {e}")
        print("Please verify your internet connection and HF_TOKEN status.\n")
        exit(1)
        
    print(f"[Scale] Successfully pulled {len(large_dataset)} organic text blocks.")

    # Vectorize the real dataset into a 384-dimensional floating matrix
    print("[Scale] Vectorizing text entries into 384-dimensional space...")
    raw_embeddings = model.encode(large_dataset).astype(np.float32)
    
    # 3. Populate our index with upcast overflow protections
    quantizer = ScalarQuantizer8()
    index = CompressedIndex(quantizer)
    index.add_vectors(raw_embeddings, large_dataset)
    
    # 4. Process an organic user query
    user_query = "International world leaders negotiate global economic infrastructure policies"
    query_embedding = model.encode([user_query]).astype(np.float32)
    
    # Test Pass A: Custom In-Quantized-Space Search
    start = time.perf_counter()
    quantized_matches = index.search_quantized_space(query_embedding, top_k=10)
    quantized_latency = (time.perf_counter() - start) * 1000
    
    # Test Pass B: Exact Brute-Force Dot Product Search (Ground Truth Baseline)
    exact_scores = np.dot(raw_embeddings, query_embedding.flatten())
    exact_top_indexes = np.argsort(exact_scores)[::-1][:10].tolist()
    
    # Extract structural indices to prevent comparison crashes
    quantized_top_indexes = [idx for idx, text, score in quantized_matches]
    
    # Evaluate Systemic Fidelity (Recall@10) via direct index intersections
    correct_recoveries = len(set(quantized_top_indexes).intersection(set(exact_top_indexes)))
    recall_at_10 = (correct_recoveries / 10.0) * 100
    
    print("\n═══ Real-World Performance Diagnostics ═══")
    print(f"Data Set Processing Volume: {raw_embeddings.shape[0]} Active Documents")
    print(f"Asymmetric Search Latency : {quantized_latency:.4f} ms")
    print(f"System Retrieval Fidelity (Recall@10): {recall_at_10:.1f}% 🎯")
    
    print(f"\nTop Quantized Match: '{quantized_matches[0][1]}'")

if __name__ == "__main__":
    run_production_scale_benchmark()