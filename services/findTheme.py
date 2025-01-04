from services.preprocessing.text_cleaner import TextCleaner
from services.preprocessing.vectorizer import DocumentVectorizer
from services.preprocessing.embeddings import BERTEmbeddings
from services.topic_modeling.lda_model import TopicModeler
from services.topic_modeling.clustering import TopicClusterer
from services.topic_modeling.topic_validator import TopicValidator
from services.utils.data_loader import DataLoader
from services.utils.matrix_operations import MatrixOperations
from typing import Dict, List, Set
import numpy as np

class ThemeAnalyzer:
    def __init__(self, num_topics: int = 10):
        self.text_cleaner = TextCleaner()
        self.vectorizer = DocumentVectorizer(max_features=1000)
        self.bert_embeddings = BERTEmbeddings()
        self.topic_modeler = TopicModeler(num_topics=num_topics)
        self.clusterer = TopicClusterer(n_clusters=num_topics)
        self.validator = TopicValidator()
        
    def preprocess_documents(self, documents: List[str]) -> List[str]:
        """Preprocesa una lista de documentos"""
        return [self.text_cleaner.preprocess_text(doc) for doc in documents]
    
    def extract_topics(self, content: str) -> Set[str]:
        """
        Extrae tópicos de un texto usando una combinación de LDA y clustering
        """
        # Preprocesar el texto
        processed_text = self.text_cleaner.preprocess_text(content)
        documents = [processed_text]  # Convertir a lista para procesamiento
        
        # Vectorización TF-IDF
        doc_vectors, feature_names = self.vectorizer.fit_transform(documents)
        
        # Obtener embeddings BERT
        bert_vectors = self.bert_embeddings.get_embeddings(documents)
        
        # Tokenizar para LDA
        tokenized_docs = [self.text_cleaner.tokenize(doc) for doc in documents]
        
        # Entrenar modelo LDA
        self.topic_modeler.fit(tokenized_docs)
        topic_distribution = self.topic_modeler.get_document_topics(tokenized_docs)
        
        # Obtener palabras clave por tópico
        topic_words = self.topic_modeler.get_topic_words(num_words=8)
        
        # Combinar resultados
        result_topics = set()
        for topic_id, word_probs in topic_words.items():
            top_words = [word for word, _ in word_probs[:3]]
            result_topics.add(', '.join(top_words))
        
        return result_topics

def findKeyWords(content: str) -> Set[str]:
    """
    Función de compatibilidad con la versión anterior
    """
    analyzer = ThemeAnalyzer(num_topics=1)
    return analyzer.extract_topics(content)

def transformTopics(topics: List[tuple]) -> Set[str]:
    """
    Transforma los tópicos al formato esperado por la API
    """
    if not topics or not isinstance(topics, (list, set)):
        return set()
        
    # Si es un conjunto (resultado de extract_topics), retornarlo directamente
    if isinstance(topics, set):
        return topics
        
    # Si es el formato antiguo de LDA
    topics_strings = topics[0][1]
    expresions = topics_strings.split('+')

    result = []
    for exp in expresions:
        exp = exp.split('"')
        exp = exp[1].replace('"','')
        result.append(exp)

    result_string = ', '.join(result[:3])
    return {result_string}