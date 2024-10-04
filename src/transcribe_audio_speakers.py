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

# Función para transcribir un conjunto de segmentos de un mismo hablante
def transcribe_speaker(audio_path, speaker_segments):
    # Concatenar todos los segmentos de un hablante en un solo archivo de audio
    audio = AudioSegment.empty()
    for start_time, end_time in speaker_segments:
        # Cambiar de .wav a .flac
        segment = AudioSegment.from_file(audio_path, format="flac")[start_time * 1000:end_time * 1000]  # Convertir segundos a milisegundos
        audio += segment
    
    # Guardar el audio concatenado en un archivo temporal
    temp_path = "temp_speaker_audio.flac"
    audio.export(temp_path, format="flac")
    
    # Usar Whisper para transcribir el segmento concatenado
    result = model.transcribe(temp_path, language='es')
    os.remove(temp_path)  # Eliminar el archivo temporal
    return result['text']

# Iterar sobre los archivos de audio en el directorio
for audio_file in os.listdir(PROCESSED_AUDIO_DIR):
    if audio_file.endswith('.flac'):  # Cambiar para que busque archivos .flac
        audio_path = os.path.join(PROCESSED_AUDIO_DIR, audio_file)
        
        # Realizar la diarización para detectar los hablantes
        diarization = pipeline(audio_path)
        
        # Agrupar los segmentos por hablante
        speakers_segments = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in speakers_segments:
                speakers_segments[speaker] = []
            speakers_segments[speaker].append((turn.start, turn.end))
        
        # Guardar la transcripción segmentada por hablante
        transcription_filename = os.path.splitext(audio_file)[0] + '_speakers.txt'
        transcription_path = os.path.join(TRANSCRIPTIONS_DIR, transcription_filename)
        
        with open(transcription_path, 'w', encoding='utf-8') as f:
            for speaker, segments in speakers_segments.items():
                print(f"Transcribiendo segmentos de {speaker}")
                
                # Transcribir todos los segmentos concatenados del mismo hablante
                speaker_transcription = transcribe_speaker(audio_path, segments)
                
                # Escribir la transcripción con el número del hablante
                f.write(f"{speaker}: {speaker_transcription}\n")
        
        print(f'Transcripción guardada en: {transcription_path}')
