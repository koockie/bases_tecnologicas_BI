# Filtrar comentarios que contienen la palabra clave
palabra_clave = "service"
comentarios_filtrados = df[df['cleaned_review'].str.contains(palabra_clave, na=False)]

# Mostrar estadísticas de las calificaciones asociadas
print(f"Comentarios relacionados con la palabra '{palabra_clave}':")
print(comentarios_filtrados[['cleaned_review', 'rating']].head())
print("\nDistribución de ratings para la palabra clave:")
print(comentarios_filtrados['rating'].value_counts())
