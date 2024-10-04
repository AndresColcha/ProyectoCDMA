import os
import whisper
import numpy as np
import librosa
import soundfile as sf

# Configuración de los directorios
PROCESSED_AUDIO_DIR = 'data/processed/'
TRANSCRIPTIONS_DIR = 'data/transcriptions/'

# Cargar el modelo Whisper
model = whisper.load_model("large")

# Función para dividir el audio en segmentos de 30 segundos
def split_audio(file_path, segment_length=30):
    audio, sr = librosa.load(file_path, sr=16000)
    total_duration = librosa.get_duration(y=audio, sr=sr)
    num_segments = int(np.ceil(total_duration / segment_length))
    
    segments = []
    for i in range(num_segments):
        start = i * segment_length * sr
        end = min((i + 1) * segment_length * sr, len(audio))
        segments.append(audio[int(start):int(end)])
    
    return segments, sr

# Función para transcribir el audio completo
def transcribe_audio(file_path, output_path):
    print(f"Procesando {file_path}...")
    segments, sr = split_audio(file_path)
    
    full_transcription = ""
    
    for i, segment in enumerate(segments):
        temp_path = os.path.join(PROCESSED_AUDIO_DIR, f"temp_segment_{i}.wav")
        sf.write(temp_path, segment, sr)
        
        # Transcribir el segmento
        result = model.transcribe(temp_path)
        full_transcription += result["text"] + "\n"
        
        # Eliminar el archivo temporal
        os.remove(temp_path)
    
    # Guardar la transcripción completa
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_transcription)
    
    print(f"Transcripción guardada en {output_path}")

# Procesar los audios en el directorio
def process_audios(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory, filename)
            transcription_path = os.path.join(TRANSCRIPTIONS_DIR, f"{os.path.splitext(filename)[0]}.txt")
            transcribe_audio(file_path, transcription_path)

# Procesar todos los archivos de audio en el directorio de entrada
process_audios(PROCESSED_AUDIO_DIR)
