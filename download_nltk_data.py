import nltk

# Descargar recursos necesarios
resources = [
    'punkt',
    'stopwords',
    'wordnet',
    'averaged_perceptron_tagger',
    'maxent_ne_chunker',
    'words',
    'omw-1.4'
]

for resource in resources:
    print(f"Descargando {resource}...")
    nltk.download(resource)

print("¡Todos los recursos han sido descargados!")
