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
RESULTS_CSV_PATH = 'data/transcriptions/transcriptions_with_sentiment_and_classification.csv'
EVOLUTION_CSV_PATH = 'data/transcriptions/evolution_with_sentiment_and_classification.csv'
CATEGORIES_CSV_PATH = 'data/categorization/categories.csv'

# Cargar modelo preentrenado para análisis de sentimientos
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Leer las categorías y palabras clave desde el archivo CSV
categories_df = pd.read_csv(CATEGORIES_CSV_PATH)
CATEGORIES = {
    row["Categoría"]: row["Palabras clave y frases"].split(", ") for _, row in categories_df.iterrows()
}

# Función para normalizar texto
def normalize_text(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    words = text.split()
    converted_words = [num2words(int(word), lang='es') if word.isdigit() else word for word in words]
    text = " ".join(converted_words)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()

# Función para clasificar sentimiento
def analyze_sentiment(text):
    normalized_text = normalize_text(text)
    inputs = tokenizer(normalized_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    if predicted_class in [0, 1]:
        return "Frustración", normalized_text
    elif predicted_class == 2:
        return "Neutral", normalized_text
    else:
        return "Satisfacción", normalized_text

# Función para detectar categorías
def detect_categories(text):
    categories_detected = []
    for category, keywords in CATEGORIES.items():
        for keyword in keywords:
            if re.search(rf"\b{re.escape(keyword)}\b", text, re.IGNORECASE):
                categories_detected.append(category)
                break
    return ", ".join(categories_detected) if categories_detected else "Sin clasificación"

# Leer transcripciones
df = pd.read_csv(TRANSCRIPTIONS_CSV_PATH)

# Analizar sentimiento y clasificar categorías
df["sentimiento"], df["texto_normalizado"] = zip(*df["transcripcion_pura"].apply(analyze_sentiment))
df["clasificacion"] = df["transcripcion_pura"].apply(detect_categories)

# Guardar las transcripciones actuales en RESULTS_CSV_PATH
df.to_csv(RESULTS_CSV_PATH, index=False, encoding="utf-8")

# Si el archivo evolutivo ya existe, cargarlo; si no, crear un DataFrame vacío
if pd.io.common.file_exists(EVOLUTION_CSV_PATH):
    evolution_df = pd.read_csv(EVOLUTION_CSV_PATH)
else:
    evolution_df = pd.DataFrame(columns=df.columns)

# Concatenar las transcripciones actuales con el evolutivo y guardar
evolution_df = pd.concat([evolution_df, df], ignore_index=True)
evolution_df.to_csv(EVOLUTION_CSV_PATH, index=False, encoding="utf-8")

print(f"Transcripciones actuales con sentimientos y clasificación guardadas en: {RESULTS_CSV_PATH}")
print(f"Transcripciones acumuladas con sentimientos y clasificación guardadas en: {EVOLUTION_CSV_PATH}")
