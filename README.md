contenido archivos:
*  McDonalds_Reviews_Con_Sentimiento.csv : reviews con categoría de sentimiento en cada review
*  McDonalds_Reviews_Modificado.csv : mismo que con sentimiento pero agrega la frase mas representativa en una nueva categoría paa ser analizada por el modelo
*  Palabras_Mas_Relevantes.csv : archivo con las palabras mas relevantes detectadas por el modelo en base a las frases representativas y las estrellas otorgadas
*  anomalías.py: Este codigo se entrena para buscar tiendas con "anomalias" viendo los ratings y el ultimo año de reviews, el tema es que puede ser anómala por ser buena o por ser mala, hay que ver como
  discriminar eso, y como interpretar el rating , que está normalizado.
* predicción.py: contiene codigo para encontrar palabras mas reitaradas y determinar su importancia en las reviews, mostrando las 10 mas importantes y sus respectivos valores de imoprtancia (segun yo solo
  sirven los que son caracteristicas, como service,food,etc)
* popularidad.py : usa prophet para predecir la futura popularidad de las tiendas

  tendencia popularidad:
      pip install prophet
