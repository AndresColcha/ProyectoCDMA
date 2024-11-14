import os
import whisper
import pandas as pd
from datetime import datetime
from pyannote.audio import Pipeline
from pydub import AudioSegment

# Definir los directorios
PROCESSED_AUDIO_DIR = 'data/processed/'
TRANSCRIPTIONS_CSV_PATH = 'data/transcriptions/transcriptions.csv'
EVOLUTION_CSV_PATH = 'data/transcriptions/evolution.csv'

# Crear el directorio de transcripciones si no existe
os.makedirs(os.path.dirname(TRANSCRIPTIONS_CSV_PATH), exist_ok=True)

# Cargar el modelo Whisper large
model = whisper.load_model("large")

# Cargar el pipeline de diarización de pyannote-audio
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization", 
    use_auth_token="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"
)

# Función para transcribir un segmento de audio específico
def transcribe_segment(audio_path, start_time, end_time):
    segment = AudioSegment.from_file(audio_path, format="wav")[start_time * 1000:end_time * 1000]  # Convertir segundos a milisegundos
    temp_path = "temp_speaker_audio.wav"
    segment.export(temp_path, format="wav")
    
    result = model.transcribe(temp_path, language='es')
    os.remove(temp_path)  # Eliminar el archivo temporal
    
    transcription_text = result['text']
    return transcription_text

# Inicializar un DataFrame vacío para las transcripciones actuales
transcriptions_df = pd.DataFrame(columns=[
    "nombre_archivo", "fecha_transcripcion", "transcripcion_ordenada",
    "transcripcion_combinada", "transcripcion_pura"
])

# Verificar si el archivo evolution.csv ya existe
if os.path.exists(EVOLUTION_CSV_PATH):
    evolution_df = pd.read_csv(EVOLUTION_CSV_PATH)
else:
    evolution_df = pd.DataFrame(columns=[
        "nombre_archivo", "fecha_transcripcion", "transcripcion_ordenada",
        "transcripcion_combinada", "transcripcion_pura"
    ])

# Iterar sobre los archivos de audio en el directorio
for audio_file in os.listdir(PROCESSED_AUDIO_DIR):
    if audio_file.endswith('.wav'):
        audio_path = os.path.join(PROCESSED_AUDIO_DIR, audio_file)
        
        diarization = pipeline(audio_path)
        speakers_sorted_text = []
        speakers_combined_text = {}
        pure_transcription = []
        last_speaker = None

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_transcription = transcribe_segment(audio_path, turn.start, turn.end)
            
            if speaker != last_speaker:
                speakers_sorted_text.append(f"{speaker}: ")
                last_speaker = speaker
            speakers_sorted_text.append(f"{speaker_transcription}\n")

            if speaker not in speakers_combined_text:
                speakers_combined_text[speaker] = []
            speakers_combined_text[speaker].append(speaker_transcription)

            pure_transcription.append(speaker_transcription)

        # Preparar las transcripciones
        transcripcion_ordenada = "".join(speakers_sorted_text)
        transcripcion_combinada = "\n".join([f"{speaker}: {' '.join(texts)}" for speaker, texts in speakers_combined_text.items()])
        transcripcion_pura = " ".join(pure_transcription)

        # Fecha de transcripción
        fecha_transcripcion = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Crear un nuevo registro
        new_entry = pd.DataFrame([{
            "nombre_archivo": audio_file,
            "fecha_transcripcion": fecha_transcripcion,
            "transcripcion_ordenada": transcripcion_ordenada,
            "transcripcion_combinada": transcripcion_combinada,
            "transcripcion_pura": transcripcion_pura
        }])

        # Agregar la nueva entrada al DataFrame de transcripciones actuales
        transcriptions_df = pd.concat([transcriptions_df, new_entry], ignore_index=True)
        # Agregar la nueva entrada al DataFrame de evolución
        evolution_df = pd.concat([evolution_df, new_entry], ignore_index=True)

# Guardar las transcripciones actuales sobrescribiendo el archivo transcriptions.csv
transcriptions_df.to_csv(TRANSCRIPTIONS_CSV_PATH, index=False, encoding="utf-8")

# Guardar el archivo evolution.csv acumulando todas las transcripciones
evolution_df.to_csv(EVOLUTION_CSV_PATH, index=False, encoding="utf-8")

print(f"Transcripciones guardadas en: {TRANSCRIPTIONS_CSV_PATH} y {EVOLUTION_CSV_PATH}")
