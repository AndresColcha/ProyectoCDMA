from fastapi import APIRouter
from app.services.storage import get_history

router = APIRouter()

@router.get("/history")
def get_transcriptions_history():
    """
    Endpoint para obtener el historial de transcripciones.

    :return: Un diccionario con el historial de transcripciones.
    """
    try:
        history = get_history()
        return {"history": history}
    except Exception as e:
        return {"error": str(e)}
