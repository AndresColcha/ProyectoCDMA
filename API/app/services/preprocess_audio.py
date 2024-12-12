import librosa
import soundfile as sf
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence
import tempfile  # Para manejar archivos temporales

# Directorios de los datos
RAW_AUDIO_DIR = './data/raw/'
PROCESSED_AUDIO_DIR = './data/processed/'

def remove_silence(audio, sample_rate, silence_thresh=-40, min_silence_len=500):
    """
    Elimina los silencios largos del audio.
    
    :param audio: El audio cargado como un array de numpy
    :param sample_rate: La frecuencia de muestreo del audio
    :param silence_thresh: Umbral de decibelios por debajo del cual se considera silencio
    :param min_silence_len: Duración mínima del silencio en ms que se eliminará
    :return: El audio sin silencios
    """
    # Convertir el array de librosa a pydub AudioSegment para procesar
    audio_segment = AudioSegment(
        (audio * 32767).astype("int16").tobytes(), 
        frame_rate=sample_rate,
        sample_width=2, 
        channels=1
    )

    # Dividir el audio en chunks eliminando los silencios largos
    chunks = split_on_silence(
        audio_segment, 
        min_silence_len=min_silence_len, 
        silence_thresh=silence_thresh
    )
    
    # Si hay muchos silencios, concatenar el audio en un solo segmento
    if chunks:
        processed_audio = AudioSegment.silent(duration=0)
        for chunk in chunks:
            processed_audio += chunk
        
        # Crear un archivo temporal para exportar el audio procesado
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            processed_audio.export(temp_path, format="wav")
        
        # Cargar el archivo temporal de nuevo en librosa
        audio_no_silence, _ = librosa.load(temp_path, sr=sample_rate)
        
        # Eliminar el archivo temporal
        os.remove(temp_path)
        
        return audio_no_silence  # Sin normalización de volumen
    else:
        return audio  # Si no hay silencio significativo, retornar el audio original

def preprocess_audio(file_path, target_sample_rate=16000):
    """
    Carga y preprocesa el audio, asegurando que tenga la frecuencia de muestreo adecuada.
    """
    # Cargar el audio
    audio, sample_rate = librosa.load(file_path, sr=target_sample_rate)
    
    # Eliminar silencios
    audio_no_silence = remove_silence(audio, sample_rate)
    
    return audio_no_silence, target_sample_rate

def save_processed_audio(audio, sample_rate, output_path):
    """
    Guarda el archivo de audio procesado en formato WAV.
    """
    # Guardar el archivo en formato WAV
    sf.write(output_path, audio, sample_rate, format='WAV')
    print(f"Archivo guardado en formato WAV: {output_path}")

def process_all_audios():
    """
    Procesa todos los audios en el directorio RAW y guarda los procesados.
    """
    for file_name in os.listdir(RAW_AUDIO_DIR):
        if file_name.endswith(".wav"):
            raw_file_path = os.path.join(RAW_AUDIO_DIR, file_name)
            processed_file_path = os.path.join(PROCESSED_AUDIO_DIR, file_name)
            
            # Preprocesar el audio
            audio, sample_rate = preprocess_audio(raw_file_path)
            
            # Guardar el audio preprocesado en formato WAV
            save_processed_audio(audio, sample_rate, processed_file_path)
            print(f"Procesado: {file_name}")

if __name__ == "__main__":
    process_all_audios()
