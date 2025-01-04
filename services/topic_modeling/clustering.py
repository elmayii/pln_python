from sklearn.cluster import KMeans
import numpy as np
from typing import List, Dict
from collections import Counter

class TopicClusterer:
    def __init__(self, n_clusters: int = 10):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        
    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Agrupa los embeddings en clusters
        """
        return self.kmeans.fit_predict(embeddings)
    
    def get_cluster_centers(self) -> np.ndarray:
        """
        Obtiene los centroides de los clusters
        """
        return self.kmeans.cluster_centers_
    
    def get_top_terms_per_cluster(self, 
                                cluster_labels: np.ndarray,
                                terms: List[str],
                                top_n: int = 10) -> Dict[int, List[str]]:
        """
        Obtiene los términos más frecuentes en cada cluster
        """
        cluster_terms = {}
        
        for cluster_id in range(self.n_clusters):
            # Obtener índices de documentos en este cluster
            cluster_docs = np.where(cluster_labels == cluster_id)[0]
            
            # Contar términos en este cluster
            cluster_term_counts = Counter()
            for doc_idx in cluster_docs:
                cluster_term_counts.update([terms[doc_idx]])
                
            # Obtener top términos
            top_terms = [term for term, _ in cluster_term_counts.most_common(top_n)]
            cluster_terms[cluster_id] = top_terms
            
        return cluster_terms
