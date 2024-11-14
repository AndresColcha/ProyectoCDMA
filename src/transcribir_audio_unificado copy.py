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
    segment = AudioSegment.from_file(audio_path, format="wav")[start_time * 1000:end_time * 1000]  # Convertir segundos a milisegundos
    
    # Guardar el segmento en un archivo temporal
    temp_path = "temp_speaker_audio.wav"
    segment.export(temp_path, format="wav")
    
    # Usar Whisper para transcribir el segmento
    result = model.transcribe(temp_path, language='es')
    os.remove(temp_path)  # Eliminar el archivo temporal
    
    return result['text']

# Iterar sobre los archivos de audio en el directorio
for audio_file in os.listdir(PROCESSED_AUDIO_DIR):
    if audio_file.endswith('.wav'):
        audio_path = os.path.join(PROCESSED_AUDIO_DIR, audio_file)
        
        # Realizar la diarización para detectar los hablantes
        diarization = pipeline(audio_path)
        
        # Variables para guardar las transcripciones
        speakers_sorted_text = []
        speakers_combined_text = {}
        pure_transcription = []

        # Variable para guardar el último speaker para evitar repetición
        last_speaker = None
        
        # Procesar cada segmento
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            print(f"Transcribiendo segmento de {speaker}, de {turn.start:.2f} a {turn.end:.2f} segundos")
            
            # Transcribir el segmento del hablante actual
            speaker_transcription = transcribe_segment(audio_path, turn.start, turn.end)
            
            # Formatear la salida para el archivo con hablantes ordenados
            if speaker != last_speaker:
                speakers_sorted_text.append(f"{speaker}: ")
                last_speaker = speaker
            speakers_sorted_text.append(f"{speaker_transcription}\n")

            # Concatenar transcripción por hablante
            if speaker not in speakers_combined_text:
                speakers_combined_text[speaker] = []
            speakers_combined_text[speaker].append(speaker_transcription)

            # Agregar transcripción al archivo puro
            pure_transcription.append(speaker_transcription)

        # Guardar la transcripción con hablantes ordenados
        sorted_filename = os.path.splitext(audio_file)[0] + '_speakers_sorted.txt'
        sorted_path = os.path.join(TRANSCRIPTIONS_DIR, sorted_filename)
        with open(sorted_path, 'w', encoding='utf-8') as f:
            f.writelines(speakers_sorted_text)

        # Guardar la transcripción combinada por hablante
        combined_filename = os.path.splitext(audio_file)[0] + '_speakers_combined.txt'
        combined_path = os.path.join(TRANSCRIPTIONS_DIR, combined_filename)
        with open(combined_path, 'w', encoding='utf-8') as f:
            for speaker, texts in speakers_combined_text.items():
                f.write(f"{speaker}: {' '.join(texts)}\n")

        # Guardar la transcripción pura sin secciones
        pure_filename = os.path.splitext(audio_file)[0] + '_pure_transcription.txt'
        pure_path = os.path.join(TRANSCRIPTIONS_DIR, pure_filename)
        with open(pure_path, 'w', encoding='utf-8') as f:
            f.write(' '.join(pure_transcription))

        print(f'Transcripciones guardadas en: {TRANSCRIPTIONS_DIR}')
