from fastapi import FastAPI
from flask import jsonify, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

app = FastAPI()

# Simulación de datos de entrenamiento
df_train = pd.DataFrame({
    'num_clusters': [3, 5, 7],
    'distance_metric': ['euclidean', 'cosine', 'manhattan'],
    'dataset_size': [1000, 5000, 20000]
})

X_train = df_train[['num_clusters', 'distance_metric', 'dataset_size']]
y_train = df_train['clustering_quality']

X_train_scaled = StandardScaler().fit_transform(X_train)
X_train, X_test, y_train, y_test = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

@app.get("/")
def recommend_clustering():
    # Recibir datos del cliente
    data = request.json
    
    num_clusters = data.get('num_clusters')
    distance_metric = data.get('distance_metric')
    dataset_size = data.get('dataset_size')
    
    # Preparar los datos
    input_data = [[num_clusters, distance_metric, dataset_size]]
    input_data_scaled = StandardScaler().fit_transform(input_data)
    
    # Hacer la predicción
    prediction = model.predict(input_data_scaled)[0]
    
    return jsonify({
        'recommended_config': {
            'num_clusters': int(num_clusters),
            'distance_metric': distance_metric,
            'dataset_size': dataset_size
        },
        'predicted_quality': float(prediction)
    })

if __name__ == '__main__':
    app.run(debug=True)
