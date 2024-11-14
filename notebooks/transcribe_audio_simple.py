import os
import whisper
from correccion_contextual import aplicar_correcciones_advanced  # Importa la función de corrección contextual

# Definir los directorios
PROCESSED_AUDIO_DIR = 'data/processed/'
TRANSCRIPTIONS_DIR = 'data/transcriptions/'

# Crear el directorio de transcripciones si no existe
os.makedirs(TRANSCRIPTIONS_DIR, exist_ok=True)

# Cargar el modelo Whisper large
model = whisper.load_model("large")

# Función para transcribir un archivo de audio completo
def transcribe_audio(audio_path):
    # Usar Whisper para transcribir el archivo de audio completo
    result = model.transcribe(audio_path, language='es')
    return result['text']

# Iterar sobre los archivos de audio en el directorio
for audio_file in os.listdir(PROCESSED_AUDIO_DIR):
    if audio_file.endswith('.flac'):  # Asegurarse de que los archivos sean .flac
        audio_path = os.path.join(PROCESSED_AUDIO_DIR, audio_file)
        
        # Transcribir el archivo de audio completo
        print(f"Transcribiendo archivo de audio: {audio_file}")
        transcription = transcribe_audio(audio_path)
        
        # Aplicar corrección contextual a la transcripción
        transcription_corregida = aplicar_correcciones_advanced(transcription)
        
        # Guardar la transcripción corregida en un archivo de texto
        transcription_filename = os.path.splitext(audio_file)[0] + '_transcription.txt'
        transcription_path = os.path.join(TRANSCRIPTIONS_DIR, transcription_filename)
        
        with open(transcription_path, 'w', encoding='utf-8') as f:
            f.write(transcription_corregida)
        
        print(f'Transcripción corregida guardada en: {transcription_path}')
