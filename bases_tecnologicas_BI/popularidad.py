import pandas as pd
from datetime import datetime, timedelta
from prophet import Prophet
import matplotlib.pyplot as plt
import re

# Función para convertir las fechas de texto a un formato estándar
def parse_time_to_date(time_str, reference_date):
    time_str = time_str.lower()
    if "a day" in time_str:
        return reference_date - timedelta(days=1)
    elif "a month" in time_str:
        return reference_date - timedelta(days=30)
    elif "a year" in time_str:
        return reference_date - timedelta(days=365)
    elif "a week" in time_str:
        return reference_date - timedelta(weeks=1)
    match = re.search(r"(\d+)", time_str)
    if match:
        value = int(match.group(1))
        if "day" in time_str:
            return reference_date - timedelta(days=value)
        elif "week" in time_str:
            return reference_date - timedelta(weeks=value)
        elif "month" in time_str:
            return reference_date - timedelta(days=value * 30)
        elif "year" in time_str:
            return reference_date - timedelta(days=value * 365)
    print(f"Advertencia: No se pudo procesar '{time_str}', usando la fecha de referencia.")
    return reference_date

# Definir la fecha de referencia
reference_date = datetime(2023, 12, 1)

# Cargar los datos
df = pd.read_csv("McDonalds_Reviews_Con_Sentimiento.csv")

# Preprocesar las fechas
df['review_date'] = df['review_time'].apply(lambda x: parse_time_to_date(x, reference_date))
df['month'] = df['review_date'].dt.to_period('M')  # Agrupar por mes

# Mapear ratings (convertir '4 stars' a 4)
rating_mapping = {
    '1 star': 1,
    '2 stars': 2,
    '3 stars': 3,
    '4 stars': 4,
    '5 stars': 5
}
df['rating'] = df['rating'].map(rating_mapping)

# Filtrar ratings fuera del rango esperado
df = df[(df['rating'] >= 1) & (df['rating'] <= 5)]

# Agrupar los datos por tienda y mes
grouped = df.groupby(['store_address', 'month']).agg({
    'rating': 'mean',  # Promedio mensual de calificaciones
    'rating_count': 'sum'  # Total mensual de reseñas
}).reset_index()

# Verificar valores del grupo
print("Verificando datos agrupados:")
print(grouped[['store_address', 'month', 'rating']].describe())

# Forzar valores dentro del rango
grouped['rating'] = grouped['rating'].clip(lower=1, upper=5)

# Normalizar las fechas para Prophet
grouped['month'] = grouped['month'].dt.to_timestamp()

# Solicitar filtro al usuario
search_query = input("Ingresa una palabra clave (ciudad, calle, número) para filtrar las tiendas: ").lower()

# Filtrar las tiendas que contienen la palabra clave
filtered_stores = grouped[grouped['store_address'].str.contains(search_query, case=False)]

if filtered_stores.empty:
    print(f"No se encontraron tiendas que coincidan con '{search_query}'.")
else:
    # Obtener las tiendas únicas que coinciden con el filtro
    unique_stores = filtered_stores['store_address'].unique()
    
    # Crear gráficos para las tiendas filtradas
    fig, axes = plt.subplots(len(unique_stores), 1, figsize=(10, 5 * len(unique_stores)), sharex=True)

    for i, store in enumerate(unique_stores):
        # Filtrar datos de una tienda
        store_data = filtered_stores[filtered_stores['store_address'] == store]
        
        # Preparar datos para Prophet con límites
        df_prophet = store_data.rename(columns={'month': 'ds', 'rating': 'y'})
        df_prophet['cap'] = 5  # Establecer límite superior
        df_prophet['floor'] = 1  # Establecer límite inferior

        # Validar df_prophet
        print(f"Datos de Prophet para la tienda: {store}")
        print(df_prophet[['ds', 'y', 'cap', 'floor']].head())
        
        # Crear y ajustar el modelo Prophet
        model = Prophet(growth='logistic', seasonality_mode='additive', seasonality_prior_scale=10)
        model.fit(df_prophet)
        
        # Predecir el futuro (6 meses)
        future = model.make_future_dataframe(periods=6, freq='M')
        future['cap'] = 5  # Aplicar límites al futuro
        future['floor'] = 1
        forecast = model.predict(future)
        
        # Graficar la tendencia para cada tienda
        ax = axes[i] if len(unique_stores) > 1 else axes
        model.plot(forecast, ax=ax, plot_cap=False)
        ax.set_title(f"Tendencia de la tienda: {store}", fontsize=10)
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Rating Promedio")

    plt.tight_layout()
    plt.show()