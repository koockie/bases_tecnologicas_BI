from textblob import TextBlob
import pandas as pd

# Cargar el dataset
df = pd.read_csv("reviews.csv", encoding="ISO-8859-1")


# Función para análisis de sentimiento
def sentiment_analysis(text):
    score = TextBlob(text).sentiment.polarity
    if score > 0.1:
        return "positivo"
    elif score < -0.1:
        return "negativo"
    else:
        return "neutro"

# Aplicar la función
df["sentimiento"] = df["review"].apply(sentiment_analysis)

# Guardar el resultado en un nuevo archivo CSV para cargar en Power BI
df.to_csv("McDonalds_Reviews_Con_Sentimiento.csv", index=False)

