import torch
from transformers import BertTokenizer, BertForMaskedLM

# Cargar el modelo y el tokenizer de BETO reentrenado
tokenizer = BertTokenizer.from_pretrained("./models/beto_finetuned")
model = BertForMaskedLM.from_pretrained("./models/beto_finetuned")
model.eval()  # Poner el modelo en modo evaluación

# Función para identificar palabras potencialmente fuera de contexto (mejorada)
def identify_words_to_mask(transcription):
    # Tokenizar el texto para separar las palabras
    words = transcription.split()
    
    # Heurística: Enmascarar palabras que podrían no encajar bien en el contexto
    # (Esta es solo una estrategia básica y puede mejorarse con análisis estadístico)
    words_to_mask = []
    for i, word in enumerate(words):
        # Excluir palabras muy comunes o artículos (mejorar con un análisis más avanzado)
        if word.lower() not in ["el", "la", "los", "las", "y", "de", "en", "a"]:
            words_to_mask.append((i, word))
    
    return words_to_mask

# Función para corregir una transcripción usando enmascarado con BETO
def correct_transcription_with_beto(transcription):
    # Identificar palabras para enmascarar
    words_to_mask = identify_words_to_mask(transcription)
    if not words_to_mask:
        return transcription  # Si no se encuentran palabras para enmascarar, devolver el texto original

    best_correction = transcription

    # Probar enmascarar cada palabra identificada y evaluar la corrección
    for index, word_to_mask in words_to_mask:
        masked_text = transcription.replace(word_to_mask, "[MASK]", 1)
        
        # Tokenizar el texto con la palabra enmascarada
        inputs = tokenizer(masked_text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        # Obtener las mejores predicciones para el token enmascarado
        top_k = 5  # Considerar las 5 mejores predicciones
        predictions = torch.topk(outputs.logits[0, index], top_k).indices.tolist()

        # Evaluar las predicciones y seleccionar la mejor
        for predicted_token_id in predictions:
            predicted_word = tokenizer.decode([predicted_token_id]).strip()
            corrected_text = transcription.replace("[MASK]", predicted_word, 1)
            
            # Aquí podrías agregar una evaluación de qué tan bien encaja la corrección
            # Por ahora, simplemente seleccionaremos la primera corrección válida
            if corrected_text != transcription:
                best_correction = corrected_text
                break

    return best_correction
