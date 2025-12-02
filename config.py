import os
import yaml

CONFIG_FILE = "config.yaml"

def cargar_configuracion(filepath=CONFIG_FILE):
    if not os.path.exists(filepath):
        print(f"Error: El archivo de configuración '{filepath}' no se encontró.")
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

config = cargar_configuracion()

if config:
    OLLAMA_ENDPOINT = config.get("ollama_endpoint")
    OUTPUT_FILE = config.get("output_file")
    IMAGES_DIR = config.get("directorio_imagenes")
    MODELO_A_EVALUAR = config.get("modelo_a_evaluar")
    HIPERPARAMETROS = config.get("hiperparametros", {})
    TAREAS_A_EJECUTAR = config.get("tareas_a_ejecutar", None)
else:
    OLLAMA_ENDPOINT = None
    OUTPUT_FILE = "output.json"
    IMAGES_DIR = None
    MODELO_A_EVALUAR = None
    HIPERPARAMETROS = {}
    TAREAS_A_EJECUTAR = None