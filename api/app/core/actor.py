import whisper
import ray

# Crear el actor global
@ray.remote
class WhisperActor:
    def __init__(self):
        print("Cargando modelo Whisper en el actor...")
        self.model = whisper.load_model("large")
        print("Modelo Whisper cargado completamente en el actor.")
    def transcribe(self, audio_path):
        return self.model.transcribe(audio_path, language="es", fp16=False)

# Inicializar el actor global
whisper_actor = WhisperActor.remote()
