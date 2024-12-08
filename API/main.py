from fastapi import FastAPI
from app.routers import transcriptions, history

app = FastAPI()

# Registrar los routers
app.include_router(transcriptions.router, prefix="/api", tags=["Transcriptions"])
app.include_router(history.router, prefix="/api", tags=["History"])

@app.get("/")
def root():
    return {"message": "Welcome to the transcription API"}
