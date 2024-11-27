# Importar librerías necesarias
import pandas as pd
import re  # Asegurarse de importar esta librería
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Cargar el dataset
df = pd.read_csv("McDonalds_Reviews_Con_Sentimiento.csv")

# Mostrar un resumen de los datos
print("Resumen de los datos:")
print(df.head())

# Limpieza básica del texto (remover caracteres especiales y transformar a minúsculas)
def clean_text(text):
    return re.sub(r'[^a-zA-Z\s]', '', text).lower()

df['cleaned_review'] = df['review'].apply(clean_text)

# Vectorización del texto con CountVectorizer
print("\nVectorizando comentarios...")
vectorizer = CountVectorizer(stop_words='english', max_features=1000)  # Limitar a 1000 palabras más relevantes
X = vectorizer.fit_transform(df['cleaned_review']).toarray()

# Etiquetas de salida (rating)
y = df['rating']

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo Random Forest
print("\nEntrenando el modelo Random Forest...")
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
print("\nEvaluando el modelo...")
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# Identificar palabras más relevantes según el modelo
print("\nIdentificando palabras más relevantes...")
feature_importance = pd.DataFrame({
    'word': vectorizer.get_feature_names_out(),
    'importance': model.feature_importances_
})
feature_importance = feature_importance.sort_values(by='importance', ascending=False)

# Mostrar las palabras más importantes
print("\nPalabras más relevantes para la predicción:")
print(feature_importance.head(10))


feature_importance.to_csv("Palabras_Mas_Relevantes.csv", index=False)
print("\nLas palabras más relevantes se han guardado en 'Palabras_Mas_Relevantes.csv'")

df.to_csv("McDonalds_Reviews_Modificado.csv", index=False)
print("\nEl dataset modificado se ha guardado en 'McDonalds_Reviews_Modificado.csv'")

# Filtrar comentarios que contienen la palabra clave
palabra_clave = "service"
comentarios_filtrados = df[df['cleaned_review'].str.contains(palabra_clave, na=False)]

# Mostrar estadísticas de las calificaciones asociadas
print(f"Comentarios relacionados con la palabra '{palabra_clave}':")
print(comentarios_filtrados[['cleaned_review', 'rating']].head())
print("\nDistribución de ratings para la palabra clave:")
print(comentarios_filtrados['rating'].value_counts())

# Lista de palabras relevantes
palabras_relevantes = feature_importance['word'].head(10)

# Calcular probabilidades para cada palabra
for palabra in palabras_relevantes:
    comentarios_palabra = df[df['cleaned_review'].str.contains(palabra, na=False)]
    total_palabra = len(comentarios_palabra)
    calificaciones_bajas = comentarios_palabra[comentarios_palabra['rating'].isin(['1 star', '2 stars'])]
    total_bajas = len(calificaciones_bajas)
    
    if total_palabra > 0:  # Evitar división por cero
        probabilidad_baja = total_bajas / total_palabra
        print(f"Palabra: {palabra} - Probabilidad de 1 o 2 estrellas: {probabilidad_baja:.2%}")
