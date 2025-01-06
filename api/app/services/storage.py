import pandas as pd
import os

HISTORY_FILE = "data/transcriptions/history.csv"

# Asegurarse de que el directorio y archivo existan al inicializar
os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
if not os.path.exists(HISTORY_FILE):
    # Crear un archivo CSV vacío con las columnas esperadas
    pd.DataFrame(columns=[
        "file_name", "date", "transcription_raw", "transcription_corrected",
        "transcription_ordered", "sentiment", "category"
    ]).to_csv(HISTORY_FILE, index=False)

def save_to_history(data):
    """
    Guarda un registro en el archivo histórico.
    
    :param data: Diccionario con los datos a guardar.
    """
    try:
        # Cargar el histórico existente si existe
        if os.path.exists(HISTORY_FILE):
            existing_df = pd.read_csv(HISTORY_FILE)
        else:
            existing_df = pd.DataFrame(columns=[
                "file_name", "date", "transcription_raw", "transcription_corrected",
                "transcription_ordered", "sentiment", "category"
            ])
        
        # Convertir los nuevos datos a un DataFrame
        new_entry = pd.DataFrame([data])
        
        # Concatenar con el histórico y guardar
        updated_df = pd.concat([existing_df, new_entry], ignore_index=True)
        updated_df.to_csv(HISTORY_FILE, index=False, encoding="utf-8")
        print(f"Registro guardado correctamente en {HISTORY_FILE}")
    except Exception as e:
        print(f"Error al guardar en el histórico: {e}")

def get_history():
    """
    Lee el archivo histórico y devuelve los registros como una lista de diccionarios.
    
    :return: Lista de diccionarios con los registros del histórico.
    """
    try:
        # Verificar si el archivo existe antes de intentar leerlo
        if os.path.exists(HISTORY_FILE):
            df = pd.read_csv(HISTORY_FILE)
            return df.to_dict(orient="records")
        else:
            print("El archivo histórico no existe. Retornando lista vacía.")
            return []
    except Exception as e:
        print(f"Error al leer el histórico: {e}")
        return []