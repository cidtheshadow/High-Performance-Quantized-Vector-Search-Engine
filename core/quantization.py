import numpy as np

class ScalarQuantizer8:
    def __init__(self):
        self.global_min = None
        self.global_max = None

    def fit(self, vectors: np.ndarray):
        """Analyzes spatial coordinate boundaries across the vector matrix."""
        # Find the absolute min and max bounds across each vector dimension
        self.global_min = np.min(vectors, axis=0)
        self.global_max = np.max(vectors, axis=0)
        # Prevent division by zero errors on flat data lines
        self.global_max[self.global_max == self.global_min] += 1e-5

    def compress(self, vectors: np.ndarray) -> np.ndarray:
        """Transforms float32 coordinates into tight uint8 integer blocks."""
        # Linear scaling formula: (X - min) / (max - min) * 255
        normalized = (vectors - self.global_min) / (self.global_max - self.global_min)
        quantized = np.clip(normalized * 255, 0, 255).astype(np.uint8)
        return quantized

    def decompress(self, quantized_vectors: np.ndarray) -> np.ndarray:
        """Reconstructs approximate float32 values for precise distance evaluations."""
        return self.global_min + (quantized_vectors.astype(np.float32) / 255.0) * (self.global_max - self.global_min)