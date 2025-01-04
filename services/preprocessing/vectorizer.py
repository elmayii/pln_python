from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
from typing import List, Tuple

class DocumentVectorizer:
    def __init__(self, max_features: int = 1000, n_components: int = 100):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.svd = TruncatedSVD(n_components=n_components)
        
    def fit_transform(self, documents: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Transforma documentos en vectores TF-IDF y reduce dimensionalidad
        """
        # Crear matriz TF-IDF
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        
        # Reducir dimensionalidad
        reduced_matrix = self.svd.fit_transform(tfidf_matrix)
        
        # Obtener características más importantes
        feature_names = self.vectorizer.get_feature_names_out()
        
        return reduced_matrix, feature_names
    
    def transform(self, documents: List[str]) -> np.ndarray:
        """
        Transforma nuevos documentos usando el modelo entrenado
        """
        tfidf_matrix = self.vectorizer.transform(documents)
        return self.svd.transform(tfidf_matrix)
