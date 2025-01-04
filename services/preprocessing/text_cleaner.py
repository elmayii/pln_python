import nltk
import spacy
from typing import List, Text
import re

# Descargar recursos necesarios de NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

class TextCleaner:
    def __init__(self):
        self.nlp = spacy.load('es_core_news_sm')
        self.stopwords = set(nltk.corpus.stopwords.words('spanish'))
        
    def normalize_text(self, text: Text) -> Text:
        """Normaliza el texto: lowercase, elimina caracteres especiales"""
        text = text.lower()
        text = re.sub(r'\n', '', text).replace('\t','')
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def tokenize(self, text: Text) -> List[Text]:
        """Tokeniza el texto usando NLTK"""
        return nltk.word_tokenize(text, language='spanish')
    
    def remove_stopwords(self, tokens: List[Text]) -> List[Text]:
        """Elimina stopwords de la lista de tokens"""
        return [token for token in tokens if token not in self.stopwords and token.isalnum()]
    
    def lemmatize(self, text: Text) -> Text:
        """Lematiza el texto usando spaCy"""
        doc = self.nlp(text)
        return ' '.join([token.lemma_ for token in doc])
    
    def preprocess_text(self, text: Text) -> Text:
        """Pipeline completo de preprocesamiento"""
        text = self.normalize_text(text)
        text = self.lemmatize(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        return ' '.join(tokens)
