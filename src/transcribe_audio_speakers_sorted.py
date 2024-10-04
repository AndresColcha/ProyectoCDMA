import os
import whisper
from pyannote.audio import Pipeline
from pydub import AudioSegment

# Definir los directorios
PROCESSED_AUDIO_DIR = 'data/processed/'
TRANSCRIPTIONS_DIR = 'data/transcriptions/'

# Crear el directorio de transcripciones si no existe
os.makedirs(TRANSCRIPTIONS_DIR, exist_ok=True)

# Cargar el modelo Whisper large
model = whisper.load_model("large")

# Cargar el pipeline de diarización de pyannote-audio (requiere el token de Hugging Face)
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="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")

# Función para transcribir un segmento de audio específico
def transcribe_segment(audio_path, start_time, end_time):
    # Extraer el segmento de audio
    segment = AudioSegment.from_file(audio_path, format="flac")[start_time * 1000:end_time * 1000]  # Convertir segundos a milisegundos
    
    # Guardar el segmento en un archivo temporal
    temp_path = "temp_speaker_audio.flac"
    segment.export(temp_path, format="flac")
    
    # Usar Whisper para transcribir el segmento
    result = model.transcribe(temp_path, language='es')
    os.remove(temp_path)  # Eliminar el archivo temporal
    
    return result['text']

# Iterar sobre los archivos de audio en el directorio
for audio_file in os.listdir(PROCESSED_AUDIO_DIR):
    if audio_file.endswith('.flac'):  # Cambiar para que busque archivos .flac
        audio_path = os.path.join(PROCESSED_AUDIO_DIR, audio_file)
        
        # Realizar la diarización para detectar los hablantes
        diarization = pipeline(audio_path)
        
        # Guardar la transcripción segmentada por hablante
        transcription_filename = os.path.splitext(audio_file)[0] + '_speakers_sorted.txt'
        transcription_path = os.path.join(TRANSCRIPTIONS_DIR, transcription_filename)
        
        # Variable para guardar el último speaker para evitar repetición
        last_speaker = None
        
        with open(transcription_path, 'w', encoding='utf-8') as f:
            # Iterar sobre cada segmento y su hablante
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                print(f"Transcribiendo segmento de {speaker}, de {turn.start:.2f} a {turn.end:.2f} segundos")
                
                # Transcribir el segmento del hablante actual
                speaker_transcription = transcribe_segment(audio_path, turn.start, turn.end)
                
                # Solo imprimir el speaker si ha cambiado
                if speaker != last_speaker:
                    f.write(f"{speaker}: ")
                    last_speaker = speaker
                
                # Escribir la transcripción del segmento
                f.write(f"{speaker_transcription}\n")
        
        print(f'Transcripción guardada en: {transcription_path}')
