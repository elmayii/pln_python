import pandas as pd
from typing import List, Union, Tuple
import json
import os

class DataLoader:
    @staticmethod
    def load_text_files(directory: str) -> List[str]:
        """
        Carga documentos de texto desde un directorio
        """
        documents = []
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                    documents.append(f.read())
        return documents
    
    @staticmethod
    def load_json_documents(filepath: str, text_field: str) -> List[str]:
        """
        Carga documentos desde un archivo JSON
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [item[text_field] for item in data]
    
    @staticmethod
    def load_csv_documents(filepath: str, text_column: str) -> List[str]:
        """
        Carga documentos desde un archivo CSV
        """
        df = pd.read_csv(filepath)
        return df[text_column].tolist()
    
    @staticmethod
    def save_results(results: dict, output_path: str):
        """
        Guarda los resultados en formato JSON
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
