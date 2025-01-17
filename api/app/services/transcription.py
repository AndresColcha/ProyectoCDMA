import os
from datetime import datetime
import pandas as pd
from pydub import AudioSegment
import ray
from app.services.preprocess_audio import preprocess_audio, save_processed_audio
from app.services.correction import correct_transcription
from app.services.sentiment import analyze_sentiment
from app.core.actor import whisper_actor  # Importar el actor

# Rutas de datos
HISTORY_FILE = "./data/transcriptions/history.csv"
PROCESSED_AUDIO_DIR = "./data/processed/"

# Asegurarse de que el archivo histórico existe
os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
if not os.path.exists(HISTORY_FILE):
    pd.DataFrame(columns=[
        "nombre_archivo", "fecha_transcripcion", "transcripcion_pura",
        "transcripcion_corregida", "transcripcion_ordenada", "sentimiento"
    ]).to_csv(HISTORY_FILE, index=False)

@ray.remote
def transcribe_segment(actor, audio_path, start_time, end_time, base_name):
    """
    Transcribe un segmento de audio específico entre start_time y end_time.
    """
    try:
        temp_path = f"{os.path.splitext(base_name)[0]}_segment_{start_time}_{end_time}.wav"  # Nombre único para el archivo temporal
        segment = AudioSegment.from_file(audio_path, format="wav")[start_time * 1000:end_time * 1000]

        # Exportar segmento a archivo temporal
        if len(segment) > 0:
            segment.export(temp_path, format="wav")
        else:
            raise ValueError(f"El segmento entre {start_time} y {end_time} está vacío.")

        # Transcribir el segmento utilizando el actor
        result = ray.get(actor.transcribe.remote(temp_path))
        os.remove(temp_path)  # Eliminar archivo temporal después de usarlo
        return result["text"]
    except Exception as e:
        raise RuntimeError(f"Error al transcribir el segmento {start_time}-{end_time}: {str(e)}")

def save_to_history(data):
    """
    Guarda un registro en el archivo histórico.
    """
    try:
        if os.path.exists(HISTORY_FILE):
            existing_df = pd.read_csv(HISTORY_FILE)
        else:
            existing_df = pd.DataFrame(columns=[
                "nombre_archivo", "fecha_transcripcion", "transcripcion_pura",
                "transcripcion_corregida", "texto_normalizado", "sentimiento"
            ])
        
        new_entry = pd.DataFrame([data])
        updated_df = pd.concat([existing_df, new_entry], ignore_index=True)
        updated_df.to_csv(HISTORY_FILE, index=False, encoding="utf-8")
        print(f"Registro guardado correctamente en {HISTORY_FILE}")
    except Exception as e:
        print(f"Error al guardar en el histórico: {e}")

import asyncio

async def transcribe_audio_in_parallel(processed_file_path, base_name, segment_duration=10):
    """
    Divide y transcribe un archivo de audio en paralelo usando Ray de forma asíncrona.
    """
    audio = AudioSegment.from_file(processed_file_path)
    total_duration = len(audio) // 1000  # Duración total en segundos

    # Crear tareas para cada segmento
    tasks = [
        transcribe_segment.remote(whisper_actor, processed_file_path, start, min(start + segment_duration, total_duration), base_name)
        for start in range(0, total_duration, segment_duration)
    ]

    # Ejecutar las tareas en paralelo y esperar los resultados usando asyncio.to_thread
    transcriptions = await asyncio.gather(*[asyncio.to_thread(ray.get, task) for task in tasks])

    # Unir las transcripciones
    complete_transcription = " ".join(transcriptions)
    return complete_transcription


async def process_and_store_transcription(file_path, file_name):
    """
    Procesa un archivo de audio desde `data/raw`, realiza transcripción, corrige y analiza el sentimiento,
    y elimina los archivos temporales una vez finalizado el procesamiento.
    """
    try:
        # Preprocesar el audio
        processed_audio, sample_rate = preprocess_audio(file_path)

        # Ruta para guardar el archivo procesado
        processed_file_path = os.path.join(PROCESSED_AUDIO_DIR, file_name)
        os.makedirs(os.path.dirname(processed_file_path), exist_ok=True)

        # Guardar el archivo procesado
        save_processed_audio(processed_audio, sample_rate, processed_file_path)

        # Transcribir el audio completo en paralelo
        transcription_pure = await transcribe_audio_in_parallel(processed_file_path, file_name)

        # Corrección utilizando el actor de BETO
        transcription_corrected = correct_transcription(transcription_pure)

        # Análisis de sentimiento y normalización
        sentiment, normalized_text = analyze_sentiment(transcription_corrected)

        # Preparar datos para guardar
        transcription_data = {
            "nombre_archivo": file_name,
            "fecha_transcripcion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "transcripcion_pura": transcription_pure,
            "transcripcion_corregida": transcription_corrected,
            "texto_normalizado": normalized_text,
            "sentimiento": sentiment
        }

        # Guardar en el historial
        save_to_history(transcription_data)

        # Eliminar archivos temporales (crudo y procesado)
        if os.path.exists(file_path):
            os.remove(file_path)
        if os.path.exists(processed_file_path):
            os.remove(processed_file_path)

        return transcription_data
    except Exception as e:
        # Si ocurre un error, asegúrate de limpiar también los archivos temporales
        if os.path.exists(file_path):
            os.remove(file_path)
        if os.path.exists(processed_file_path):
            os.remove(processed_file_path)
        raise RuntimeError(f"Error procesando el archivo {file_name}: {str(e)}")


if __name__ == "__main__":
    os.makedirs(PROCESSED_AUDIO_DIR, exist_ok=True)
    print("Listo para procesar y transcribir audios usando Ray.")
