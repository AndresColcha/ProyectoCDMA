
# ProyectoCDMA

## Transcripción de voz a texto y análisis de sentimientos en llamadas telefónicas, mediante técnicas de aprendizaje automático

## Descripción General
Este proyecto tiene como objetivo la **transcripción de llamadas telefónicas a texto** para encuestas de calidad en **Speedy Internet**, un ISP ubicado en el centro del Ecuador. La transcripción se realizará utilizando el modelo **Whisper Large v2**, desarrollado por OpenAI. El grupo de trabajo es el grupo 05, compuesto por:

- Víctor Castro
- Andrés Colcha
- Anthony Peña
- Brayan Tiglla

El objetivo a largo plazo es utilizar estas transcripciones para un **análisis de sentimientos** que permita evaluar la calidad de nuestro servicio ofrecido a los clientes.

## Estructura del Proyecto

### Directorio `data`
- **raw**: Contiene los datos en bruto o sin procesar.
- **processed**: Almacena los datos que ya han pasado por un preprocesamiento.
- **transcriptions**: Guarda las transcripciones generadas a partir de los datos de audio.
- **notebooks**: Guarda los archivos de prueba.

### Directorio `src`
- **preprocess_audio.py**: Script encargado del preprocesamiento de archivos de audio, para optimizar la transcripción.
- **transcribe_audio_simple.py**: Script para la transcripción básica de los archivos de audio.
- **transcribe_audio_speakers.py**: Realiza la transcripción de audio diferenciando a los diferentes oradores presentes en la llamada.
- **transcribe_audio_speakers_sorted.py**: Versión modificada para ordenar o manejar la transcripción de los diferentes oradores de manera específica.

## Dependencias
Las dependencias necesarias para ejecutar el proyecto están listadas en el archivo `requisitos.txt`. Asegúrese de instalar todas las dependencias antes de ejecutar los scripts.

```bash
pip install -r requisitos.txt
```

## Uso
1. **Preprocesar el audio**:
   
   ```bash
   python src/preprocess_audio.py
   ```

2. **Transcribir audio** (transcripción aplicando 3 enfoques):

   ```bash
   python src/transcribir_audio_unificado.py
   ```

## Próximos Pasos
Actualmente, el proyecto se encuentra en la fase de transcripción utilizando el modelo Whisper Large v2. Los siguientes pasos incluyen:

1. **Optimización del proceso de transcripción**.
2. **Análisis de sentimientos** de las transcripciones para evaluar la satisfacción del cliente.

## Contribuciones
Este proyecto es desarrollado por el grupo 05 de la Maestría de Ciencia de datos y Máquinas de Aprendizaje - UIDE. Cualquier contribución o sugerencia es bienvenida.
