from transformers import pipeline

# Crear un pipeline de NLP usando un modelo preentrenado
# Usaremos el modelo "bert-base-uncased" para completar frases
corregir_contextualmente = pipeline("fill-mask", model="bert-base-uncased")

# Diccionario de correcciones básicas
correcciones = {
    "Spirit": "Speedy",
    "Speed": "Speedy",
    # Más términos similares
}

def aplicar_correcciones_advanced(transcripcion):
    # Corrección básica usando el diccionario
    for error, correccion in correcciones.items():
        transcripcion = transcripcion.replace(error, correccion)
    
    # Corrección contextual si detectamos un posible error
    palabras_sospechosas = ["Spirit", "Speed"]  # Palabras que queremos validar contextualmente
    for palabra in palabras_sospechosas:
        if palabra in transcripcion:
            # Ajustamos el contexto usando el modelo de lenguaje
            transcripcion = corregir_palabra_contextualmente(transcripcion, palabra)
    
    return transcripcion

def corregir_palabra_contextualmente(transcripcion, palabra_incorrecta):
    # Reemplaza la palabra incorrecta con una máscara [MASK] para que el modelo sugiera una corrección
    transcripcion_mascara = transcripcion.replace(palabra_incorrecta, "[MASK]")
    
    # Usa el pipeline de NLP para predecir la palabra correcta en el contexto
    sugerencias = corregir_contextualmente(transcripcion_mascara)
    
    # Filtramos las mejores sugerencias que correspondan al término correcto
    for sugerencia in sugerencias:
        palabra_sugerida = sugerencia['token_str']
        if palabra_sugerida.lower() == "speedy":  # Comprobamos si la sugerencia es la que esperamos
            transcripcion = transcripcion.replace(palabra_incorrecta, "Speedy")
            break
    
    return transcripcion
