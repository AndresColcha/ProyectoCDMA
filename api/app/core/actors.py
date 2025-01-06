import ray
from app.core.actor import WhisperActor  # Importar la clase definida anteriormente
from app.core.beto_actor import BetoActor  # Importar la clase del actor de BETO

# Configuración de los actores
NUM_WHISPER_ACTORS = 5
NUM_BETO_ACTORS = 2

# Variables para almacenar los actores
whisper_actors = []
beto_actors = []

def initialize_actors():
    """
    Inicializa los actores de Whisper y BETO.
    """
    global whisper_actors, beto_actors

    # Inicializar Ray (solo si no está inicializado)
    if not ray.is_initialized():
        ray.init(include_dashboard=True, num_cpus=50)

    # Crear múltiples instancias de WhisperActor
    whisper_actors = [WhisperActor.remote() for _ in range(NUM_WHISPER_ACTORS)]
    print(f"{NUM_WHISPER_ACTORS} actores de Whisper creados.")

    # Crear múltiples instancias de BetoActor
    beto_actors = [BetoActor.remote() for _ in range(NUM_BETO_ACTORS)]
    print(f"{NUM_BETO_ACTORS} actores de BETO creados.")
