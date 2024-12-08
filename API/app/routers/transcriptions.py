from fastapi import APIRouter, UploadFile, HTTPException
from app.services.transcription import process_and_store_transcription
import os

# Directorio donde se guardarán los archivos temporales
RAW_AUDIO_DIR = "./data/raw/"
os.makedirs(RAW_AUDIO_DIR, exist_ok=True)

router = APIRouter()

@router.post("/transcriptions")
async def transcribe_audio(files: list[UploadFile]):
    """
    Endpoint para procesar múltiples archivos de audio y devolver las transcripciones.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No se enviaron archivos para procesar.")

    results = []
    for file in files:
        try:
            # Guardar el archivo temporalmente en `data/raw`
            temp_path = os.path.join(RAW_AUDIO_DIR, file.filename)
            with open(temp_path, "wb") as temp_file:
                temp_file.write(file.file.read())

            # Procesar el archivo desde el directorio `data/raw`
            transcription_data = process_and_store_transcription(temp_path, file.filename)
            results.append(transcription_data)

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error al procesar el archivo {file.filename}: {str(e)}")

    return {"transcriptions": results}
