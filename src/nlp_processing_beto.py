import torch
from transformers import BertTokenizer, BertForMaskedLM

# Cargar el modelo y el tokenizer de BETO reentrenado
tokenizer = BertTokenizer.from_pretrained("./models/beto_finetuned")
model = BertForMaskedLM.from_pretrained("./models/beto_finetuned")
model.eval()  # Poner el modelo en modo evaluación

# Función mejorada para identificar palabras potencialmente fuera de contexto
def identify_words_to_mask(transcription):
    # Tokenizar el texto usando un modelo de lenguaje para analizar la probabilidad de las palabras en contexto
    tokens = tokenizer(transcription, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokens)

    # Calcular las probabilidades de cada token en el contexto
    probabilities = torch.softmax(outputs.logits, dim=-1)
    words = transcription.split()
    words_to_mask = []

    for i, word in enumerate(words):
        # Excluir palabras comunes, pero usar las probabilidades del modelo para identificar rarezas
        if word.lower() not in ["el", "la", "los", "las", "y", "de", "en", "a"]:
            token_id = tokens["input_ids"][0, i].item()
            token_prob = probabilities[0, i, token_id].item()
            if token_prob < 0.1:  # Threshold: palabras con baja probabilidad son más sospechosas
                words_to_mask.append((i, word))

    return words_to_mask

# Función mejorada para corregir una transcripción usando BETO con evaluación de similitud semántica
def correct_transcription_with_beto(transcription):
    words_to_mask = identify_words_to_mask(transcription)
    if not words_to_mask:
        return transcription

    best_correction = transcription
    highest_similarity = 0.0

    for index, word_to_mask in words_to_mask:
        masked_text = transcription.replace(word_to_mask, "[MASK]", 1)
        inputs = tokenizer(masked_text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        top_k = 5
        predictions = torch.topk(outputs.logits[0, index], top_k).indices.tolist()

        for predicted_token_id in predictions:
            predicted_word = tokenizer.decode([predicted_token_id]).strip()
            corrected_text = transcription.replace("[MASK]", predicted_word, 1)

            # Evaluar la similitud semántica entre la transcripción original y la corregida
            similarity = torch.cosine_similarity(
                torch.tensor(tokenizer.encode(transcription, return_tensors="pt")),
                torch.tensor(tokenizer.encode(corrected_text, return_tensors="pt"))
            ).item()

            if similarity > highest_similarity:
                best_correction = corrected_text
                highest_similarity = similarity

    return best_correction
