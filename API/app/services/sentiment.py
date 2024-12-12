from transformers import BertTokenizer, BertForSequenceClassification
import torch
import unicodedata
import re
from num2words import num2words
import warnings

# Ignorar advertencias específicas
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# Usar un modelo preentrenado para análisis de sentimientos
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"  # Modelo preentrenado en análisis de sentimientos
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Función para normalizar el texto y convertir números a palabras
def normalize_text(text):
    """
    Normaliza el texto eliminando tildes, convirtiendo números a palabras
    y eliminando caracteres especiales.
    """
    # Eliminar tildes y caracteres especiales
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    
    # Convertir números a palabras en español
    words = text.split()
    converted_words = []
    for word in words:
        if word.isdigit():
            # Convertir número a palabras en español
            word = num2words(int(word), lang='es')
        converted_words.append(word)
    
    text = " ".join(converted_words)
    
    # Eliminar caracteres especiales y dejar solo letras y espacios
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()  # Convertir a minúsculas

# Función para analizar el sentimiento
def analyze_sentiment(transcription_pure):
    """
    Analiza el sentimiento basado en la transcripción.

    :param transcription_pure: Texto de la transcripción pura.
    :return: Sentimiento (Frustración, Neutral, Satisfacción).
    """
    # Normalizar el texto
    normalized_text = normalize_text(transcription_pure)
    
    # Preprocesar el texto y preparar los datos para el modelo
    inputs = tokenizer(
        normalized_text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    
    # Mapeo de la clase predicha a sentimiento
    if predicted_class in [0, 1]:  # Clases de sentimiento muy negativo y negativo
        sentiment = "Frustración"
    elif predicted_class == 2:  # Clase de sentimiento neutral
        sentiment = "Neutral"
    else:  # Clases de sentimiento positivo y muy positivo
        sentiment = "Satisfacción"

    return sentiment
