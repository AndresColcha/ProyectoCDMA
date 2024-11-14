import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import unicodedata
import re
from num2words import num2words

import warnings

# Ignorar advertencias específicas
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# Rutas de los archivos CSV
TRANSCRIPTIONS_CSV_PATH = 'data/transcriptions/transcriptions.csv'
RESULTS_CSV_PATH = 'data/transcriptions/transcriptions_with_sentiment.csv'
EVOLUTION_CSV_PATH = 'data/transcriptions/evolution_with_sentiment.csv'

# Usar un modelo preentrenado para análisis de sentimientos
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"  # Modelo preentrenado en análisis de sentimientos
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Función para normalizar el texto y convertir números a palabras
def normalize_text(text):
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

# Función para predecir el sentimiento y mapearlo a categorías específicas para un ISP
def analyze_sentiment(text):
    # Normalizar el texto
    normalized_text = normalize_text(text)
    
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
    
    # Mapeo de la clase predicha a categorías específicas del ISP
    if predicted_class in [0, 1]:  # Clases de sentimiento muy negativo y negativo
        return "Frustración", normalized_text
    elif predicted_class == 2:  # Clase de sentimiento neutral
        return "Neutral", normalized_text
    else:  # Clases de sentimiento positivo y muy positivo
        return "Satisfacción", normalized_text

# Leer el archivo CSV de transcripciones
df = pd.read_csv(TRANSCRIPTIONS_CSV_PATH)

# Analizar el sentimiento de la transcripción pura y agregar las nuevas columnas
df["sentimiento"], df["texto_normalizado"] = zip(*df["transcripcion_pura"].apply(analyze_sentiment))

# Guardar las transcripciones actuales con sentimientos en RESULTS_CSV_PATH
df.to_csv(RESULTS_CSV_PATH, index=False, encoding="utf-8")

# Si el archivo evolutivo ya existe, cargarlo; si no, crear un DataFrame vacío
if pd.io.common.file_exists(EVOLUTION_CSV_PATH):
    evolution_df = pd.read_csv(EVOLUTION_CSV_PATH)
else:
    evolution_df = pd.DataFrame(columns=df.columns)

# Concatenar las transcripciones actuales con el evolutivo y guardar
evolution_df = pd.concat([evolution_df, df], ignore_index=True)
evolution_df.to_csv(EVOLUTION_CSV_PATH, index=False, encoding="utf-8")

print(f"Transcripciones actuales con sentimientos guardadas en: {RESULTS_CSV_PATH}")
print(f"Transcripciones acumuladas con sentimientos guardadas en: {EVOLUTION_CSV_PATH}")
