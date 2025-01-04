from gensim.models.coherencemodel import CoherenceModel
from typing import List, Dict
import numpy as np

class TopicValidator:
    def __init__(self):
        pass
        
    def calculate_perplexity(self, lda_model, corpus) -> float:
        """
        Calcula la perplejidad del modelo LDA
        Un valor más bajo indica mejor generalización
        """
        return lda_model.log_perplexity(corpus)
    
    def calculate_coherence_score(self, 
                                lda_model, 
                                texts: List[List[str]], 
                                dictionary) -> float:
        """
        Calcula el score de coherencia usando c_v
        Un valor más alto indica tópicos más coherentes
        """
        coherence_model = CoherenceModel(
            model=lda_model,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        return coherence_model.get_coherence()
    
    def evaluate_topic_diversity(self, 
                               topic_word_dist: Dict[int, List[tuple]],
                               top_n: int = 10) -> float:
        """
        Calcula la diversidad entre tópicos basada en palabras únicas
        Un valor más alto indica tópicos más diversos
        """
        unique_words = set()
        total_words = 0
        
        for topic_id, word_probs in topic_word_dist.items():
            words = [word for word, prob in word_probs[:top_n]]
            unique_words.update(words)
            total_words += len(words)
            
        return len(unique_words) / total_words
