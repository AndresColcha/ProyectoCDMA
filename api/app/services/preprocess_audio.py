import librosa
import soundfile as sf
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence
import tempfile  # Para manejar archivos temporales
from joblib import Parallel, delayed  # Para paralelismo

# Directorios de los datos
RAW_AUDIO_DIR = './data/raw/'
PROCESSED_AUDIO_DIR = './data/processed/'

VALID_AUDIO_EXTENSIONS = [".wav", ".mp3", ".m4a", ".ogg", ".flac"]

def remove_silence(audio, sample_rate, file_name, silence_thresh=-40, min_silence_len=500):
    """
    Elimina los silencios largos del audio.
    
    :param audio: El audio cargado como un array de numpy
    :param sample_rate: La frecuencia de muestreo del audio
    :param file_name: Nombre del archivo original para nombrar los temporales
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
        for idx, chunk in enumerate(chunks):
            processed_audio += chunk
        
        # Crear un archivo temporal para exportar el audio procesado
        temp_path = f"{tempfile.gettempdir()}/{os.path.basename(file_name)}_nosilence.wav"
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
    Convierte automáticamente formatos no compatibles a WAV.
    """
    # Verificar y convertir si no es un archivo WAV
    if not file_path.endswith(".wav"):
        temp_wav_path = os.path.splitext(file_path)[0] + "_converted.wav"
        audio = AudioSegment.from_file(file_path)
        audio.export(temp_wav_path, format="wav")
        file_path = temp_wav_path

    # Cargar el audio convertido o original
    audio, sample_rate = librosa.load(file_path, sr=target_sample_rate)

    # Eliminar silencios
    audio_no_silence = remove_silence(audio, sample_rate, os.path.basename(file_path))

    # Eliminar el archivo temporal si se generó
    if file_path.endswith("_converted.wav") and os.path.exists(file_path):
        os.remove(file_path)

    return audio_no_silence, target_sample_rate

def save_processed_audio(audio, sample_rate, output_path):
    """
    Guarda el archivo de audio procesado en formato WAV.
    """
    # Guardar el archivo en formato WAV
    sf.write(output_path, audio, sample_rate, format='WAV')
    print(f"Archivo guardado en formato WAV: {output_path}")

def process_audio_file(file_name):
    """
    Procesa un único archivo de audio, verificando la extensión y guardando el resultado.
    """
    if any(file_name.endswith(ext) for ext in VALID_AUDIO_EXTENSIONS):
        raw_file_path = os.path.join(RAW_AUDIO_DIR, file_name)
        processed_file_path = os.path.join(PROCESSED_AUDIO_DIR, file_name.replace(os.path.splitext(file_name)[1], ".wav"))
        
        # Preprocesar el audio
        audio, sample_rate = preprocess_audio(raw_file_path)
        
        # Guardar el audio preprocesado en formato WAV
        save_processed_audio(audio, sample_rate, processed_file_path)
        print(f"Procesado: {file_name}")
    else:
        print(f"Archivo ignorado (formato no compatible): {file_name}")

def process_all_audios():
    """
    Procesa todos los audios en el directorio RAW y guarda los procesados en paralelo.
    """
    # Obtener la lista de archivos en el directorio RAW
    files = [f for f in os.listdir(RAW_AUDIO_DIR) if os.path.isfile(os.path.join(RAW_AUDIO_DIR, f))]

    # Procesar los archivos en paralelo
    Parallel(n_jobs=20)(delayed(process_audio_file)(file_name) for file_name in files)

if __name__ == "__main__":
    os.makedirs(RAW_AUDIO_DIR, exist_ok=True)
    os.makedirs(PROCESSED_AUDIO_DIR, exist_ok=True)
    process_all_audios()
