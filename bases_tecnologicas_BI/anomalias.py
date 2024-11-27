import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import re

# Definir la fecha de referencia como 1 de diciembre de 2023
reference_date = datetime(2023, 12, 1)

# Función para convertir texto en días desde la fecha de referencia
def parse_time_to_date(time_str, reference_date):
    time_str = time_str.lower()
    
    # Manejar casos específicos como "a day ago", "a month ago", "a year ago", "a week ago"
    if "a day" in time_str:
        return reference_date - timedelta(days=1)
    elif "a month" in time_str:
        return reference_date - timedelta(days=30)
    elif "a year" in time_str:
        return reference_date - timedelta(days=365)
    elif "a week" in time_str:
        return reference_date - timedelta(weeks=1)  # Semanas específicas

    # Buscar números explícitos en el texto
    match = re.search(r"(\d+)", time_str)
    if match:  # Si se encuentra un número
        value = int(match.group(1))
        if "day" in time_str:
            calculated_date = reference_date - timedelta(days=value)
        elif "week" in time_str:
            calculated_date = reference_date - timedelta(weeks=value)
        elif "month" in time_str:
            calculated_date = reference_date - timedelta(days=value * 30)  # Aproximar a 30 días por mes
        elif "year" in time_str:
            calculated_date = reference_date - timedelta(days=value * 365)  # Aproximar a 365 días por año
        else:
            calculated_date = reference_date  # Si no coincide con día/mes/año, usar la fecha de referencia
    else:  # Si no se encuentra un número en el texto
        print(f"Advertencia: No se pudo procesar '{time_str}', usando la fecha de referencia.")
        calculated_date = reference_date
    return calculated_date

# Cargar el dataset
df = pd.read_csv("McDonalds_Reviews_Con_Sentimiento.csv")

# Corregir la columna 'rating_count' eliminando las comas y convirtiendo a numérico
df['rating_count'] = df['rating_count'].str.replace(',', '').astype(float)

# Aplicar la función para calcular las fechas exactas
df['review_date'] = df['review_time'].apply(lambda x: parse_time_to_date(x, reference_date))

# Convertir las fechas a días desde la fecha de referencia
df['days_since_reference'] = (reference_date - df['review_date']).dt.days

# Filtrar solo las reseñas del último año (365 días)
df_last_year = df[df['days_since_reference'] <= 365]

# Seleccionar características relevantes para el modelo
features = df_last_year[['rating', 'rating_count', 'days_since_reference']].copy()

# Convertir la columna 'rating' (1 star, 2 stars, etc.) en un valor numérico
rating_mapping = {'1 star': 1, '2 stars': 2, '3 stars': 3, '4 stars': 4, '5 stars': 5}
features['rating'] = df_last_year['rating'].map(rating_mapping)

# Verificar que todas las columnas sean numéricas
features = features.apply(pd.to_numeric, errors='coerce')

# Manejar valores faltantes (rellenar con la media o eliminar)
features.dropna(inplace=True)

# Normalizar los datos
features_normalized = (features - features.mean()) / features.std()

# Entrenar el modelo Isolation Forest
model = IsolationForest(contamination=0.05, random_state=42)  # contamination: % de anomalías esperadas
features_normalized['anomaly'] = model.fit_predict(features_normalized)

# Agregar la columna 'store_address' al dataframe con anomalías
features_normalized = features_normalized.join(df_last_year['store_address'])

# Agrupar las anomalías por tienda (store_address)
anomalies_by_store = features_normalized.groupby('store_address').agg({
    'anomaly': 'mean',  # Promedio de anomalías para cada tienda
    'rating': 'mean',   # Promedio de rating para cada tienda
    'rating_count': 'mean',  # Promedio de número de reseñas
    'days_since_reference': 'mean'  # Promedio de antigüedad de reseñas
}).reset_index()

# Marcar la tienda como anomalía si el promedio de anomalías es negativo (-1)
anomalies_by_store['store_anomaly'] = anomalies_by_store['anomaly'].apply(lambda x: 'Anomalía' if x < 0 else 'Normal')

# Mostrar las tiendas consideradas como anomalías
anomalous_stores = anomalies_by_store[anomalies_by_store['store_anomaly'] == 'Anomalía']
print("Tiendas con anomalías detectadas:")
print(anomalous_stores)

# Guardar el resultado en un CSV
anomalous_stores.to_csv("Anomalías_Tiendas_Ultimo_Año.csv", index=False)
print("Anomalías por tienda guardadas en 'Anomalías_Tiendas_Ultimo_Año.csv'")
