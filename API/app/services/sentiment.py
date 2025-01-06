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

# Función para dividir el texto en fragmentos manejables
def chunk_text(text, tokenizer, max_length=512):
    """
    Divide un texto largo en fragmentos de tamaño máximo `max_length` tokens.
    """
    tokens = tokenizer.tokenize(text)
    chunks = [
        tokens[i:i + max_length]
        for i in range(0, len(tokens), max_length)
    ]
    return [tokenizer.convert_tokens_to_string(chunk) for chunk in chunks]

# Función para analizar el sentimiento en textos largos
def analyze_long_text(transcription_pure, tokenizer, model, max_length=512):
    """
    Analiza un texto largo dividiéndolo en fragmentos y combinando los resultados.

    :param transcription_pure: Texto de la transcripción pura.
    :param tokenizer: Tokenizador del modelo.
    :param model: Modelo de análisis de sentimientos.
    :param max_length: Longitud máxima de tokens por fragmento.
    :return: Tupla (Sentimiento final, Sentimientos por fragmento).
    """
    # Normalizar el texto
    normalized_text = normalize_text(transcription_pure)
    
    # Dividir en fragmentos
    chunks = chunk_text(normalized_text, tokenizer, max_length)
    
    # Analizar cada fragmento
    sentiments = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

        # Mapear sentimiento
        if predicted_class in [0, 1]:
            sentiments.append("Frustración")
        elif predicted_class == 2:
            sentiments.append("Neutral")
        else:
            sentiments.append("Satisfacción")

    # Combinar resultados (mayor frecuencia)
    final_sentiment = max(set(sentiments), key=sentiments.count)
    return final_sentiment, sentiments

# Adaptar la función principal para mantener compatibilidad
def analyze_sentiment(transcription_pure):
    """
    Analiza el sentimiento basado en la transcripción.

    :param transcription_pure: Texto de la transcripción pura.
    :return: Tupla (Sentimiento, Texto Normalizado).
    """
    # Reutilizar la función para textos largos
    final_sentiment, _ = analyze_long_text(transcription_pure, tokenizer, model)
    normalized_text = normalize_text(transcription_pure)
    return final_sentiment, normalized_text

# Ejemplo de uso
if __name__ == "__main__":
    text = "Aquí va un texto extremadamente largo que excede el límite de tokens..."
    final_sentiment, normalized_text = analyze_sentiment(text)
    print(f"Sentimiento final: {final_sentiment}")
    print(f"Texto normalizado: {normalized_text}")


    """
    def detect_categories(text):
    categories_df = pd.read_csv("data/categorization/categories.csv")
    categories = {
        row["Categoría"]: row["Palabras clave y frases"].split(", ")
        for _, row in categories_df.iterrows()
    }
    detected = [category for category, keywords in categories.items() if any(k in text for k in keywords)]
    return detected or ["Sin clasificación"]

    """
    



