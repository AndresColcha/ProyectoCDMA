from fastapi import FastAPI
from app.routers import transcriptions, history
from app.core.actor import whisper_actor
from app.core.beto_actor import beto_actor
import ray

app = FastAPI()

if __name__ == "__main__":
    ray.init(include_dashboard=True, num_cpus=40)

@app.on_event("startup")
async def startup_event():
    print("API iniciada y actores de Whisper y BETO cargados.")

@app.on_event("shutdown")
async def shutdown_event():
    ray.shutdown()
    print("Recursos de Ray liberados.")

app.include_router(transcriptions.router, prefix="/api", tags=["Transcriptions"])
app.include_router(history.router, prefix="/api", tags=["History"])
