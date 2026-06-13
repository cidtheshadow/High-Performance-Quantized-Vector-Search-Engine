# core/index.py
import numpy as np

class CompressedIndex:
    def __init__(self, quantizer):
        self.quantizer = quantizer
        self.quantized_storage = None  # Holds the true uint8 compressed matrix
        self.metadata = {}

    def add_vectors(self, vectors: np.ndarray, text_fragments: list):
        """Fits boundaries and compresses matrices into safe uint8 slots."""
        self.quantizer.fit(vectors)
        self.quantized_storage = self.quantizer.compress(vectors)
        for i, text in enumerate(text_fragments):
            self.metadata[i] = text

    def search_quantized_space(self, query_vector: np.ndarray, top_k: int = 5):
        """Asymmetric Dot Product Search: Explicit Float32 Upcasting to prevent NumPy overflow."""
        q = query_vector.flatten()
        
        d_min = self.quantizer.global_min
        d_max = self.quantizer.global_max
        delta = (d_max - d_min) / 255.0
        
        # Precompute alpha scalar constant outside the hot loop
        alpha = np.dot(q, d_min)
        scaled_query = q * delta
        
        # Upcast uint8 storage to float32 to completely prevent register overflow
        dot_products = alpha + np.dot(self.quantized_storage.astype(np.float32), scaled_query)
        
        # Sort descending to pull highest similarity coordinates (Dot Product)
        nearest_indexes = np.argsort(dot_products)[::-1][:top_k]
        
        results = []
        for idx in nearest_indexes:
            results.append((idx, self.metadata[idx], float(dot_products[idx])))
        return results