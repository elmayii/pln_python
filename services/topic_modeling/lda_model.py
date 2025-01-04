from gensim.models import LdaModel
from gensim.corpora import Dictionary
from typing import List, Tuple, Dict
import numpy as np

class TopicModeler:
    def __init__(self, num_topics: int = 10):
        self.num_topics = num_topics
        self.dictionary = None
        self.lda_model = None
        
    def fit(self, texts: List[List[str]]):
        """
        Entrena el modelo LDA con los textos proporcionados
        """
        # Crear diccionario
        self.dictionary = Dictionary(texts)
        
        # Crear corpus
        corpus = [self.dictionary.doc2bow(text) for text in texts]
        
        # Entrenar modelo LDA
        self.lda_model = LdaModel(
            corpus=corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            random_state=42,
            update_every=1,
            chunksize=100,
            passes=10,
            alpha='auto',
            per_word_topics=True
        )
        
    def get_document_topics(self, texts: List[List[str]]) -> np.ndarray:
        """
        Obtiene la distribución de tópicos para cada documento
        """
        corpus = [self.dictionary.doc2bow(text) for text in texts]
        doc_topics = []
        
        for bow in corpus:
            topic_dist = [0] * self.num_topics
            for topic, prob in self.lda_model.get_document_topics(bow):
                topic_dist[topic] = prob
            doc_topics.append(topic_dist)
            
        return np.array(doc_topics)
    
    def get_topic_words(self, num_words: int = 10) -> Dict[int, List[Tuple[str, float]]]:
        """
        Obtiene las palabras más relevantes para cada tópico
        """
        return {i: self.lda_model.show_topic(i, num_words) 
                for i in range(self.num_topics)}
