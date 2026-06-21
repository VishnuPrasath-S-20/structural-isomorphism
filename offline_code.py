import logging
import numpy as np
from structural_isomorphism.search import StructuralSearch

logging.basicConfig(level=logging.INFO)

class MockTransformer:
    def __init__(self):
        self.embedding_dimension = 384

    def encode(self, texts, **kwargs):
        lines = [texts] if isinstance(texts, str) else texts
        np.random.seed(42)
        vectors = np.random.randn(len(lines), 384)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

if __name__ == "__main__":
    search = StructuralSearch.__new__(StructuralSearch)
    search.model = MockTransformer()
    
    search.kb = [
        {
            "name": "Thermostat System", 
            "domain": "Engineering", 
            "description": "A mechanical closed-loop system that maintains thermal stability using feedback lines.",
            "type_id": "type_feedback_loop"
        },
        {
            "name": "Predator-Prey Balance", 
            "domain": "Biology", 
            "description": "An ecological ecosystem adjusting population scales through biological limits.",
            "type_id": "type_feedback_loop"
        },
        {
            "name": "Supply and Demand", 
            "domain": "Economics", 
            "description": "A market pricing index that balances inventory quantities using price adjustments.",
            "type_id": "type_market_equilibrium"
        }
    ]
    
    search._kb_embeddings = search.model.encode([item["description"] for item in search.kb])
    
    query = "A cooling fan slows down when the internal machine temperature drops"
    results = search.query(query, top_k=2)
    
    for r in results:
        print(f"[{r['domain'].upper()}] {r['name']} (Score: {r['score']:.3f})")
