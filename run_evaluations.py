# run_evaluations.py
import yaml
import subprocess
import os
import sys
import locale 

# --- CONFIGURACIÓN ---
CONFIG_FILE = "config.yaml"
SCRIPT_A_EJECUTAR = "main.py"

def modificar_config_modelo(nombre_modelo):
    """Carga el config.yaml, cambia el valor de 'modelo_a_evaluar' y lo guarda."""
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"Error al leer el archivo YAML: {e}")
        return False

    print(f"\n---> Modificando '{CONFIG_FILE}' para usar el modelo: '{nombre_modelo}'")
    config['modelo_a_evaluar'] = nombre_modelo

    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        return True
    except IOError as e:
        print(f"Error al escribir en el archivo de configuración: {e}")
        return False

def ejecutar_evaluacion():
    """Ejecuta el script main.py como un proceso separado y muestra su salida."""
    if not os.path.exists(SCRIPT_A_EJECUTAR):
        print(f"Error: No se encuentra el script '{SCRIPT_A_EJECUTAR}'.")
        return False
    print(f"--- Ejecutando el script '{SCRIPT_A_EJECUTAR}'... ---")
    try:

        system_encoding = locale.getpreferredencoding()
        print(f"(Usando la codificación del sistema: {system_encoding})")

        with subprocess.Popen(
            [sys.executable, SCRIPT_A_EJECUTAR],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding=system_encoding,
            errors='replace',
            bufsize=1
        ) as proceso:
            for linea in proceso.stdout:
                print(linea, end='')
        
        if proceso.returncode == 0:
            print(f"'{SCRIPT_A_EJECUTAR}' se ejecutó correctamente.")
            return True
        else:
            print(f"\n**************************************************")
            print(f"*** ERROR AL EJECUTAR '{SCRIPT_A_EJECUTAR}' ***")
            print(f"**************************************************")
            print(f"El script finalizó con un error (código de salida: {proceso.returncode}).")
            return False

    except FileNotFoundError:
        print(f"Error: No se encontró el intérprete de Python '{sys.executable}'.")
        return False
    except Exception as e:
        print(f"Ocurrió un error inesperado al ejecutar el script: {e}")
        return False


def main():
    """Función principal que lee los modelos del config y ejecuta las evaluaciones."""
    print("======================================================")
    print("=== INICIO DEL PROCESO DE EVALUACIÓN POR LOTES ===")
    print("======================================================")

    if not os.path.exists(CONFIG_FILE):
        print(f"Error: No se encuentra el archivo de configuración '{CONFIG_FILE}'. Finalizando.")
        return
        
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        lista_de_modelos = config.get("modelos_a_evaluar_en_lote")

        if not lista_de_modelos or not isinstance(lista_de_modelos, list):
            print(f"Error: La clave 'modelos_a_evaluar_en_lote' no se encuentra en '{CONFIG_FILE}' o no es una lista válida.")
            return
            
    except yaml.YAMLError as e:
        print(f"Error al parsear el archivo de configuración YAML: {e}")
        return

    print(f"\nSe evaluarán los siguientes {len(lista_de_modelos)} modelos encontrados en el config: {lista_de_modelos}")
    
    for modelo in lista_de_modelos:
        print(f"\n\n======================================================")
        print(f"====== PROCESANDO MODELO: {modelo} ======")
        print(f"======================================================")
        
        if not modificar_config_modelo(modelo):
            print(f"No se pudo modificar la configuración para el modelo '{modelo}'. Saltando al siguiente.")
            continue
            
        ejecutar_evaluacion()
        
        print(f"\n====== EVALUACIÓN PARA '{modelo}' FINALIZADA ======")

    print("\n\n======================================================")
    print("=== TODAS LAS EVALUACIONES HAN FINALIZADO ===")
    print("======================================================")


if __name__ == "__main__":
    main()