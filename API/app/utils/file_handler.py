import tempfile
import os

def save_temp_file(upload_file):
    """
    Guarda un archivo de tipo UploadFile como archivo temporal y devuelve la ruta.
    """
    if not hasattr(upload_file, "file") or not hasattr(upload_file, "filename"):
        raise ValueError(f"El objeto recibido ({type(upload_file)}) no es un UploadFile válido.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(upload_file.file.read())  # Lee y guarda el contenido del archivo
        return temp_file.name

def delete_temp_file(file_path):
    """
    Elimina un archivo temporal.
    """
    try:
        os.remove(file_path)
        print(f"Archivo temporal eliminado: {file_path}")
    except FileNotFoundError:
        print(f"Archivo no encontrado para eliminar: {file_path}")
    except Exception as e:
        print(f"Error al eliminar el archivo {file_path}: {str(e)}")
