import json
import os

# Opciones disponibles: "Hallucination", "hF1", "PDS"
TAREAS_A_INCLUIR = ["Hallucination"]



CARPETA_DATOS = "imagenes"
RUTA_TAXONOMIA_JSON = "taxonomia.json"
EXTENSIONES_IMAGEN = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')

def generar_prompts_desde_taxonomia(ruta_taxonomia):
    """
    Carga la taxonomía y genera un diccionario con los prompts para todas las tareas.
    """
    try:
        with open(ruta_taxonomia, 'r', encoding='utf-8') as f:
            taxonomia = json.load(f)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de taxonomía en '{ruta_taxonomia}'.")
        return None
    except json.JSONDecodeError:
        print(f"Error: El archivo '{ruta_taxonomia}' no es un JSON válido.")
        return None

    piezas_validas = taxonomia.get('Parts', [])
    daños_comunes = taxonomia.get('Common damages', {})
    severidades_validas = next(iter(daños_comunes.values()), [])

    piezas_str = '","'.join(piezas_validas)
    daños_str = '","'.join(daños_comunes.keys())
    severidades_str = '","'.join(severidades_validas)

    prompt_hf1_pds = f"""You are an expert vehicle damage appraiser. Analyze the attached image and classify the damages you see. Return a JSON object or a list of JSON objects for each type of damage you find with this structure (no additional text, no explanations):
{{
"damage": "VALUE_FROM_THE_DAMAGE_LIST",
"part": "VALUE_FROM_THE_PARTS_LIST",
"severity": "VALUE_FROM_THE_SEVERITY_LIST"
}}
If you do not observe any damage in the image, respond only with the text 'No damage'.
If you cannot determine the damage, the part, or the severity, respond only with the text 'I don't know'.
Use exclusively the values from each list for the JSON:
Valid damages: "{daños_str}"
Valid parts: "{piezas_str}"
Valid severities: "{severidades_str}"
Do not invent values outside of the lists. The only allowed output is the JSON object or, in the indicated cases, the text 'No damage' or 'I don't know'."""

    prompts_por_tarea = {
        "Hallucination": """instructions:
Analyze exclusively the vehicle in the foreground. Ignore the background, other vehicles, people, signs, or reflections.
Observe in maximum detail the entire bodywork, headlights, bumpers, windows, and joints between parts.
Your output must be only one lowercase word: "yes" or "no".
Answer "yes" if you detect at least one of these visual damages:
Scratch/scrape (linear marks on the paint)
Dent (sinking/bulging of the metal or plastic)
Crack (fissure in plastics, acrylics, or glass)
Deformation (misalignment or anomalous shape of a part)
Separation/gap (unusual gap between parts, loose clips)
Missing part (caps, trims, grilles, mirrors, etc.)
Peeled/chipped paint (areas without paint or flaking)
Rust/corrosion (visible rust spots)
Opaque headlights (whitened/yellowed affecting transparency)
Broken headlights (obvious breaks/chips)
Do not count as damage: dirt, dust, water, shadows, reflections, JPEG compression, sun glare, stickers, vinyls, or small color variations due to lighting.
Task:
'Carefully inspect the main vehicle in the image and determine if it has any of the damages from the list above. Answer only with yes or no.'""",
        "hF1": prompt_hf1_pds,
        "PDS": prompt_hf1_pds
    }
    
    return prompts_por_tarea

def obtener_estructura_base(prompts):
    """Devuelve la estructura JSON completa por defecto para un nuevo archivo."""
    estructura_completa = {
        "Hallucination": [
            {
                "prompt": prompts["Hallucination"],
                "ground_truth": "No"  
            }
        ],
        "hF1": [
            {
                "prompt": prompts["hF1"],
                "ground_truth": {
                    "damage": "Rellenar",
                    "part": "Rellenar",
                    "severity": "Rellenar"
                }
            }
        ],
        "PDS": [
            {
                "prompt": prompts["PDS"],
                "ground_truth": {
                    "damage": "Rellenar",
                    "part": "Rellenar",
                    "severity": "Rellenar"
                },
                "nivel": 5  
            }
        ]
    }
    return {tarea: estructura_completa[tarea] for tarea in TAREAS_A_INCLUIR if tarea in estructura_completa}


def procesar_carpeta_dataset():
    """Recorre la carpeta de datos, actualizando o creando los JSONs con las tareas seleccionadas."""
    print("--- Iniciando script de actualización de dataset ---")
    print(f"Tareas a procesar: {', '.join(TAREAS_A_INCLUIR)}")
    
    prompts = generar_prompts_desde_taxonomia(RUTA_TAXONOMIA_JSON)
    if not prompts:
        return 
    
    if not os.path.isdir(CARPETA_DATOS):
        print(f"Error: La carpeta de datos '{CARPETA_DATOS}' no se encuentra.")
        return

    print(f"Procesando archivos en la carpeta '{CARPETA_DATOS}'...")
    
    archivos_procesados = 0
    archivos_json_asociados = {os.path.splitext(f)[0] for f in os.listdir(CARPETA_DATOS) if f.lower().endswith(EXTENSIONES_IMAGEN)}

    if not archivos_json_asociados:
        print(f"Advertencia: No se encontraron imágenes en la carpeta '{CARPETA_DATOS}'.")
        return

    for base_nombre in sorted(list(archivos_json_asociados)):
        archivos_procesados += 1
        ruta_json = os.path.join(CARPETA_DATOS, f"{base_nombre}.json")
        
        estructura_base = obtener_estructura_base(prompts)

        if os.path.exists(ruta_json):
            try:
                with open(ruta_json, 'r+', encoding='utf-8') as f:
                    datos = json.load(f)

                    for task_name in TAREAS_A_INCLUIR:
                        if task_name not in datos:
                            if task_name in estructura_base:
                                datos[task_name] = estructura_base[task_name]
                                print(f"  -> Añadiendo tarea '{task_name}' a: {base_nombre}.json")
                        else:
                            estructura_por_defecto = estructura_base.get(task_name, [])
                            for i, item in enumerate(datos[task_name]):
                                if i < len(estructura_por_defecto):
                                     item['prompt'] = estructura_por_defecto[i]['prompt']

                    f.seek(0)
                    json.dump(datos, f, ensure_ascii=False, indent=2)
                    f.truncate()
                    print(f"  -> Actualizado: {base_nombre}.json")

            except (json.JSONDecodeError, KeyError) as e:
                print(f"  -> Error al procesar {base_nombre}.json: {e}. Se omitirá.")
        else:
            # Crea un nuevo archivo JSON solo con la estructura de las tareas seleccionadas
            with open(ruta_json, 'w', encoding='utf-8') as f:
                json.dump(estructura_base, f, ensure_ascii=False, indent=2)
                print(f"  -> CREADO: {base_nombre}.json (con estructura para tareas seleccionadas)")

    print(f"\n--- Proceso completado. Se han procesado {archivos_procesados} archivos JSON. ---")

if __name__ == "__main__":
    procesar_carpeta_dataset()