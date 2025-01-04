from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from typing import List
from tqdm import tqdm

class BERTEmbeddings:
    def __init__(self, model_name: str = 'bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Genera embeddings para una lista de textos usando BERT
        """
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenizar y preparar entrada
            inputs = self.tokenizer(batch_texts, 
                                  padding=True, 
                                  truncation=True, 
                                  return_tensors="pt",
                                  max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generar embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Usar el embedding del token [CLS] como representación del documento
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(batch_embeddings)
            
        return np.vstack(embeddings)
