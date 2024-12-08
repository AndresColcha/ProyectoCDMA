import os
from datetime import datetime
import pandas as pd
from app.services.preprocess_audio import preprocess_audio
from app.services.correction import correct_transcription_with_beto
from app.services.sentiment import analyze_sentiment
import whisper
from pyannote.audio import Pipeline
import shutil

# Rutas de datos
HISTORY_FILE = "./data/transcriptions/history.csv"

# Cargar el modelo Whisper y el pipeline de diarización
model = whisper.load_model("large")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token="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"
)

# Asegurarse de que el archivo histórico existe
os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
if not os.path.exists(HISTORY_FILE):
    pd.DataFrame(columns=[
        "nombre_archivo", "fecha_transcripcion", "transcripcion_pura",
        "transcripcion_corregida", "transcripcion_ordenada", "sentimiento"
    ]).to_csv(HISTORY_FILE, index=False)

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
                "transcripcion_corregida", "transcripcion_ordenada", "sentimiento"
            ])
        
        new_entry = pd.DataFrame([data])
        updated_df = pd.concat([existing_df, new_entry], ignore_index=True)
        updated_df.to_csv(HISTORY_FILE, index=False, encoding="utf-8")
        print(f"Registro guardado correctamente en {HISTORY_FILE}")
    except Exception as e:
        print(f"Error al guardar en el histórico: {e}")

def transcribe_segment(audio_path, start_time, end_time):
    """
    Transcribe un segmento de audio específico entre start_time y end_time.
    """
    from pydub import AudioSegment
    temp_path = "temp_segment_audio.wav"
    segment = AudioSegment.from_file(audio_path, format="wav")[start_time * 1000:end_time * 1000]
    segment.export(temp_path, format="wav")
    result = model.transcribe(temp_path, language="es")
    os.remove(temp_path)
    return result["text"]

def process_and_store_transcription(file_path, file_name):
    """
    Procesa un archivo de audio desde `data/raw`, realiza diarización, transcribe, corrige y analiza el sentimiento.
    """
    try:
        # Preprocesar el audio
        processed_audio, _ = preprocess_audio(file_path)

        # Diarización
        diarization = pipeline(file_path)
        speakers_sorted_text = []
        pure_transcriptions = []
        last_speaker = None

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segment_transcription = transcribe_segment(file_path, turn.start, turn.end)
            if speaker != last_speaker:
                speakers_sorted_text.append(f"{speaker}: ")
                last_speaker = speaker
            speakers_sorted_text.append(f"{segment_transcription}\n")
            pure_transcriptions.append(segment_transcription)

        # Crear las transcripciones
        transcription_pure = " ".join(pure_transcriptions)
        transcription_ordered = "".join(speakers_sorted_text)

        # Corrección y análisis de sentimiento
        transcription_corrected = correct_transcription_with_beto(transcription_pure)
        sentiment = analyze_sentiment(transcription_corrected)

        # Preparar datos para guardar
        transcription_data = {
            "nombre_archivo": file_name,
            "fecha_transcripcion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "transcripcion_pura": transcription_pure,
            "transcripcion_corregida": transcription_corrected,
            "transcripcion_ordenada": transcription_ordered,
            "sentimiento": sentiment
        }
        shutil.rmtree('./data/raw')
        save_to_history(transcription_data)
        return transcription_data
    except Exception as e:
        raise RuntimeError(f"Error procesando el archivo {file_name}: {str(e)}")
