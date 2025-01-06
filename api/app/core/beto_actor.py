import os
from transformers import BertTokenizer, BertForMaskedLM
import torch
import ray

# Definir la ruta del modelo usando una variable de entorno o una ruta absoluta
BETO_MODEL_PATH = os.getenv("BETO_MODEL_PATH", "models/beto_finetuned")

# Verificar que la ruta exista
if not os.path.exists(BETO_MODEL_PATH):
    raise FileNotFoundError(f"El modelo BETO no se encontró en la ruta: {BETO_MODEL_PATH}")

# Crear el actor global
@ray.remote
class BetoActor:
    def __init__(self):
        print("Cargando modelo BETO en el actor...")
        self.tokenizer = BertTokenizer.from_pretrained(BETO_MODEL_PATH)
        self.model = BertForMaskedLM.from_pretrained(BETO_MODEL_PATH)
        self.model.eval()
        print("Modelo BETO cargado completamente en el actor.")

    def correct(self, transcription):
        """
        Corrige una transcripción dividiéndola en chunks y procesándola.
        """
        chunks = self.chunk_transcription(transcription)
        corrected_transcription = []

        for chunk in chunks:
            words_to_mask = self.identify_words_to_mask(chunk)
            if not words_to_mask:
                corrected_transcription.append(chunk)
                continue

            best_chunk_correction = chunk
            highest_similarity = 0.0

            for index, word_to_mask in words_to_mask:
                masked_text = chunk.replace(word_to_mask, "[MASK]", 1)
                inputs = self.tokenizer(masked_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
                with torch.no_grad():
                    outputs = self.model(**inputs)

                top_k = 5
                predictions = torch.topk(outputs.logits[0, index], top_k).indices.tolist()

                for predicted_token_id in predictions:
                    predicted_word = self.tokenizer.decode([predicted_token_id]).strip()
                    corrected_text = chunk.replace("[MASK]", predicted_word, 1)

                    # Evaluar la similitud semántica
                    similarity = torch.cosine_similarity(
                        torch.tensor(self.tokenizer.encode(chunk, return_tensors="pt"), dtype=torch.float),
                        torch.tensor(self.tokenizer.encode(corrected_text, return_tensors="pt"), dtype=torch.float)
                    ).item()

                    if similarity > highest_similarity:
                        best_chunk_correction = corrected_text
                        highest_similarity = similarity

            corrected_transcription.append(best_chunk_correction)

        # Combinar los chunks corregidos
        return " ".join(corrected_transcription)


    def chunk_transcription(self, transcription, max_length=512):
        """
        Divide una transcripción en chunks de tamaño máximo de 512 tokens.
        """
        tokens = self.tokenizer.tokenize(transcription)
        chunks = [
            self.tokenizer.convert_tokens_to_string(tokens[i:i + max_length])
            for i in range(0, len(tokens), max_length)
        ]
        return chunks

    def identify_words_to_mask(self, transcription):
        """
        Identifica palabras potencialmente fuera de contexto en un chunk.
        """
        tokens = self.tokenizer(transcription, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**tokens)

        probabilities = torch.softmax(outputs.logits, dim=-1)
        words = transcription.split()
        words_to_mask = []

        for i, word in enumerate(words):
            if word.lower() not in ["el", "la", "los", "las", "y", "de", "en", "a"]:
                if i < tokens["input_ids"].size(1):
                    token_id = tokens["input_ids"][0, i].item()
                    token_prob = probabilities[0, i, token_id].item()
                    if token_prob < 0.1:  # Threshold para detectar rarezas
                        words_to_mask.append((i, word))

        return words_to_mask

# Inicializar el actor global
beto_actor = BetoActor.remote()
