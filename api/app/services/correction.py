from app.core.beto_actor import beto_actor  # Importar el actor de BETO
import ray

def correct_transcription(transcription):
    """
    Corrige una transcripción utilizando el actor de BETO.
    """
    try:
        # Llamar al actor de BETO para realizar la corrección
        corrected_transcription = ray.get(beto_actor.correct.remote(transcription))
        return corrected_transcription
    except Exception as e:
        raise RuntimeError(f"Error al corregir la transcripción: {str(e)}")
