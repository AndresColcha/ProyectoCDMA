import os
import random
import ray
from fastapi import APIRouter, UploadFile, HTTPException
from app.core.actors_manager import whisper_actors, beto_actors  # Importamos desde actors_manager.py

# Directorio donde se guardarán los archivos temporales
RAW_AUDIO_DIR = "./data/raw/"
os.makedirs(RAW_AUDIO_DIR, exist_ok=True)

router = APIRouter()

@router.post("/transcriptions")
async def transcribe_audio(files: list[UploadFile]):
    """
    Endpoint para procesar múltiples archivos de audio en paralelo usando Ray.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No se enviaron archivos para procesar.")

    transcriptions_futures = []

    for file in files:
        try:
            # Guardar el archivo temporalmente en `data/raw`
            temp_path = os.path.join(RAW_AUDIO_DIR, file.filename)
            with open(temp_path, "wb") as temp_file:
                temp_file.write(await file.read())

            # Seleccionar un actor aleatorio de Whisper para distribuir la carga
            whisper_actor = random.choice(whisper_actors)
            
            # Ejecutar transcripción en paralelo
            transcription_future = whisper_actor.transcribe.remote(temp_path)  # ✅ Correcto

            # Luego, seleccionar un actor aleatorio de BETO para corregir la transcripción
            beto_actor = random.choice(beto_actors)
            corrected_transcription_future = beto_actor.correct.remote(transcription_future)

            # Guardar la tarea en la lista de futuros
            transcriptions_futures.append(corrected_transcription_future)

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error al procesar el archivo {file.filename}: {str(e)}")

    # Esperar a que todas las transcripciones terminen en paralelo
    transcriptions = ray.get(transcriptions_futures)  # ✅ Correcto

    return {"transcriptions": transcriptions}
