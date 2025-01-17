import whisper
import ray

@ray.remote
class WhisperActor:
    def __init__(self):
        print("⚡ Cargando modelo Whisper en el actor...")
        self.model = whisper.load_model("large")
        print("✅ Modelo Whisper cargado completamente.")

    def transcribe(self, audio_path):
        return self.model.transcribe(audio_path, language="es", fp16=False)
