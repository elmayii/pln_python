import numpy as np
from scipy.sparse import csr_matrix, vstack
from typing import List, Tuple

class MatrixOperations:
    @staticmethod
    def sparse_to_dense(sparse_matrix: csr_matrix) -> np.ndarray:
        """
        Convierte una matriz dispersa a densa
        """
        return sparse_matrix.toarray()
    
    @staticmethod
    def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
        """
        Normaliza una matriz usando L2 norm
        """
        return matrix / np.linalg.norm(matrix, axis=1)[:, np.newaxis]
    
    @staticmethod
    def batch_process(matrix: np.ndarray, 
                     batch_size: int) -> List[np.ndarray]:
        """
        Divide una matriz en lotes para procesamiento por batches
        """
        return np.array_split(matrix, 
                            np.ceil(len(matrix) / batch_size))
    
    @staticmethod
    def cosine_similarity(matrix1: np.ndarray, 
                         matrix2: np.ndarray) -> np.ndarray:
        """
        Calcula la similitud del coseno entre dos matrices
        """
        return np.dot(matrix1, matrix2.T)
