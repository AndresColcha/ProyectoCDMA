import ray
from app.core.actor import WhisperActor
from app.core.beto_actor import BetoActor

# Iniciar Ray solo si no está inicializado
if not ray.is_initialized():
    ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=40)
    print("🚀 Ray inicializado con 40 CPUs.")

# **Número de actores a lanzar**
NUM_WHISPER_ACTORS = 5
NUM_BETO_ACTORS = 5

print("🚀 Iniciando carga de actores en paralelo...")
whisper_actors = [WhisperActor.remote() for _ in range(NUM_WHISPER_ACTORS)]
beto_actors = [BetoActor.remote() for _ in range(NUM_BETO_ACTORS)]

print(f"✅ {NUM_WHISPER_ACTORS} actores de Whisper listos.")
print(f"✅ {NUM_BETO_ACTORS} actores de BETO listos.")
