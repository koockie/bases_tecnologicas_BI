# Importar librerías necesarias
import pandas as pd
import matplotlib.pyplot as plt

# Cargar el dataset
df = pd.read_csv("McDonalds_Reviews_Modificado.csv")  # Asegúrate de que el archivo esté en esta ruta

# Lista de palabras que deseas analizar
palabras_clave = ["service", "order", "food"]

# Diccionario para almacenar las probabilidades por palabra
probabilidades = {}

# Calcular probabilidades para cada palabra y cada rating
for palabra in palabras_clave:
    comentarios_palabra = df[df['cleaned_review'].str.contains(palabra, na=False)]
    total_palabra = len(comentarios_palabra)
    if total_palabra > 0:  # Evitar división por cero
        # Contar frecuencia de cada rating
        distribucion_ratings = comentarios_palabra['rating'].value_counts(normalize=True)  # Normalizar para obtener proporciones
        # Asegurarse de que todos los ratings (1 a 5 estrellas) están en el diccionario
        probabilidades[palabra] = {rating: distribucion_ratings.get(rating, 0) for rating in ['1 star', '2 stars', '3 stars', '4 stars', '5 stars']}

# Crear el gráfico
ratings = ['1 star', '2 stars', '3 stars', '4 stars', '5 stars']  # Categorías de calificaciones
colores = ['red', 'darkorange', 'gold', 'green', 'green']  # Colores personalizados

# Generar un gráfico de barras para cada palabra
for palabra, distribucion in probabilidades.items():
    plt.figure(figsize=(8, 5))
    valores = [distribucion[rating] for rating in ratings]  # Obtener probabilidades en orden
    plt.bar(ratings, valores, color=colores)
    plt.title(f'Probabilidad de Calificaciones para "{palabra}"')
    plt.xlabel('Rating')
    plt.ylabel('Probabilidad')
    plt.ylim(0, 1)  # Las probabilidades están entre 0 y 1
    plt.show()
