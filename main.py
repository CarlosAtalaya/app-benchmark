# main.py
import os
import json
import time
#import cv2

from utils import (
    cargar_configuracion,
    construir_arbol_taxonomia,
    aplicar_y_codificar_distorsiones,
    list_images_by_prefix,
    download_image_to_tmp,
    load_ground_truth,
    put_json_s3,
    image_file_to_b64
)

from models import (
    consultar_modelo_vlm,
    consultar_modelo_text_only,
    normalizar_respuesta
)
from metrics import despachar_calculo_metricas
from reporting import imprimir_resumen_consola, generar_reporte_final


CONFIG_FILE = "config.yaml"

def main():
    print("--- INICIANDO PROCESO DE EVALUACIÓN MULTI-TAREA ---")
    config = cargar_configuracion(CONFIG_FILE)
    if not config:
        print("Falta el archivo de configuración. Finalizando.")
        return
    
    s3_config = config.get("s3_config")
    if not s3_config or not all(k in s3_config for k in ["input_bucket", "input_prefix", "output_bucket"]):
        print("Error: Falta la sección 's3_config' o alguna de sus claves en el config. Finalizando.")
        return
    
    PROMPTS_FILE = config.get("prompts_file")
    if not PROMPTS_FILE or not os.path.exists(PROMPTS_FILE):
        print(f"Error: El archivo de prompts '{PROMPTS_FILE}' definido en el config no existe.")
        return
    
    try:
        with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
            prompts_globales = json.load(f)
        print(f"Archivo de prompts globales '{PROMPTS_FILE}' cargado correctamente.")
    except json.JSONDecodeError:
        print(f"Error: El archivo de prompts '{PROMPTS_FILE}' no es un JSON válido.")
        return

    gestor = config.get("gestor")
    api_key= None
    if gestor == "ollama":
        api_endpoint = config.get("ollama_config", {}).get("endpoint")
    elif gestor == "lm_studio":
        api_endpoint = config.get("lm_studio_config", {}).get("endpoint")
    elif gestor == "vllm":
        api_endpoint = config.get("vllm_config", {}).get("endpoint")
    elif gestor =='qwen3':
        api_endpoint = config.get("qwen_config", {}).get("endpoint")
    elif gestor == "openai":
        api_endpoint = None
        api_key = config.get("openai_config", {}).get("api_key")
    elif gestor == "gemini":
        api_endpoint = None
        api_key = config.get("gemini_config", {}).get("api_key")
    elif gestor =="nova":
        api_key = config.get("nova_config", {}).get("api_key")
        api_endpoint = None
    else:
        print(f"Error: El gestor '{gestor}' no es válido. Opciones: 'ollama', 'lm_studio'.")
        return

    if not api_endpoint and gestor not in ["openai", "gemini"]:
        print(f"Error: No se encontró el 'endpoint' para el gestor '{gestor}' en el config.")
        return
        
    print(f"Usando el gestor: '{gestor}' con el endpoint: '{api_endpoint}'")
    
    OUTPUT_FILE = config.get("output_file")

    tareas_a_ejecutar = config.get("tareas_a_ejecutar", None)
    if tareas_a_ejecutar is not None:
        print(f"Se ejecutarán únicamente las siguientes tareas especificadas en el config: {tareas_a_ejecutar}")
    else:
        print("Advertencia: No se encontró la clave 'tareas_a_ejecutar'. Se ejecutarán todas las tareas encontradas en los archivos JSON.")

    arbol_taxonomia = construir_arbol_taxonomia()
    print(f"Ábol de taxonomía construido con {len(arbol_taxonomia.nodes())} nodos.")

    
    modelo_a_evaluar = config.get("modelo_a_evaluar")
    print(f"Evaluando el modelo '{modelo_a_evaluar}'...")
    
    resultados_agregados_por_tarea = {}
    total_prompt_tokens = 0
    total_eval_tokens = 0
    print(f"\n=======================================================")
    print(f"==== ESCANEANDO IMÁGENES Y PROMPTS EN S3 BUCKET '{s3_config['input_bucket']}/{s3_config['input_prefix']}' ====")
    print(f"=======================================================")
    
    archivos_en_directorio = sorted(list(list_images_by_prefix(s3_config['input_bucket'], s3_config['input_prefix'])))
    for image_key in archivos_en_directorio:
        filename = os.path.basename(image_key)
        ground_truths_para_imagen = load_ground_truth(s3_config['input_bucket'], image_key)
        if not ground_truths_para_imagen:
            print(f"\nAdvertencia: Se encontró la imagen '{filename}' pero no su JSON. Saltando.")
            continue
            
        print(f"\n--- Procesando imagen de S3: {filename} ---")
        image_path = download_image_to_tmp(s3_config['input_bucket'], image_key)

        imagen_b64_original = image_file_to_b64(image_path)
        
        if not imagen_b64_original:
            print(f"No se pudo procesar la imagen '{filename}'. Saltando.")
            os.remove(image_path) 
            continue

        for task_name, gt_info in ground_truths_para_imagen.items():
            if tareas_a_ejecutar is not None and task_name not in tareas_a_ejecutar:
                continue

            prompts_para_tarea = prompts_globales[task_name]
            if isinstance(gt_info, list) and gt_info:
                ground_truth_para_tarea = gt_info[0].get('ground_truth')
            else:
                ground_truth_para_tarea = None

            if ground_truth_para_tarea is None:
                print(f"Advertencia: No se encontró la clave 'ground_truth' para la tarea '{task_name}' en '{filename}'. Saltando tarea.")
                continue

            resultados_agregados_por_tarea.setdefault(task_name, [])
            
            if task_name == "PDS":
                imagen_cv_original = cv2.imread(image_path)
                if imagen_cv_original is None:
                    print(f"Error al leer la imagen '{filename}' con OpenCV para la tarea PDS. Saltando.")
                    continue
                
                for i, item in enumerate(prompts_para_tarea):
                    max_level = item.get("nivel", 0)
                    if max_level == 0:
                        print("Advertencia: item de PDS no tiene 'nivel'. Saltando.")
                        continue
                        
                    for nivel_actual in range(1, max_level + 1):
                        print(f"  -> Tarea '{task_name}', Prompt {i+1}/{len(prompts_para_tarea)}, Nivel de distorsión {nivel_actual}/{max_level}")
                        
                        imagen_b64_distorsionada = aplicar_y_codificar_distorsiones(imagen_cv_original, nivel_actual)
                        
                        respuesta_data = consultar_modelo_vlm(
                            prompt=item['prompt'], image_b64=imagen_b64_distorsionada, model_name=modelo_a_evaluar,
                            temperature=config.get("hiperparametros", {}).get("temperature"),
                            top_k=config.get("hiperparametros", {}).get("top_k"),
                            endpoint=api_endpoint,
                            gestor=gestor,
                            api_key=api_key
                        )
                        
                        if respuesta_data:
                            respuesta_modelo_raw = respuesta_data["response"]
                            total_prompt_tokens += respuesta_data["prompt_tokens"]
                            total_eval_tokens += respuesta_data["eval_tokens"]
                        else:
                            respuesta_modelo_raw = None
                        respuesta_modelo_norm = normalizar_respuesta(respuesta_modelo_raw, ground_truth_para_tarea)
                        
                        print(f"     GT: {ground_truth_para_tarea}")
                        print(f"     Respuesta (Nivel {nivel_actual}): {respuesta_modelo_norm}")

                        resultado_individual = {
                            "imagen": filename,
                            "prompt": item['prompt'],
                            "ground_truth": ground_truth_para_tarea,
                            "respuesta_modelo": respuesta_modelo_raw,
                            "respuesta_normalizada": respuesta_modelo_norm,
                            "nivel": nivel_actual,
                            "prompt_tokens": respuesta_data["prompt_tokens"] if respuesta_data else 0,
                            "eval_tokens": respuesta_data["eval_tokens"] if respuesta_data else 0 
                        }
                        resultados_agregados_por_tarea[task_name].append(resultado_individual)
                        time.sleep(0.1)
            
            else:
                for i, item in enumerate(prompts_para_tarea):
                    print(f"  -> Tarea '{task_name}', Prompt {i+1}/{len(prompts_para_tarea)}") 
                    
                    start_time = time.perf_counter()
                    respuesta_data = consultar_modelo_vlm(
                        prompt=item['prompt'], image_b64=imagen_b64_original, model_name=modelo_a_evaluar,
                        temperature=config.get("hiperparametros", {}).get("temperature"),
                        top_k=config.get("hiperparametros", {}).get("top_k"),
                        endpoint=api_endpoint,
                        gestor=gestor, api_key=api_key
                    )
                    end_time = time.perf_counter()
                    tiempo_inferencia = end_time - start_time
                    print(f"     [DEBUG] Tiempo inferencia: {tiempo_inferencia:.4f}s")
                    if respuesta_data:
                        respuesta_modelo_raw = respuesta_data["response"]
                        total_prompt_tokens += respuesta_data["prompt_tokens"]
                        total_eval_tokens += respuesta_data["eval_tokens"]
                    else:
                        respuesta_modelo_raw = None
                    respuesta_modelo_norm = normalizar_respuesta(respuesta_modelo_raw, ground_truth_para_tarea)
                    

                    print(f"     GT: {ground_truth_para_tarea}")
                    print(f"     Respuesta Multimodal: {respuesta_modelo_norm}")
                    if respuesta_data:
                        print(f"     Prompt Tokens: {respuesta_data['prompt_tokens']}")
                        print(f"     Answer Tokens: {respuesta_data['eval_tokens']}")


                    resultado_individual = {
                        "imagen": filename,
                        "prompt": item['prompt'],
                        "ground_truth": ground_truth_para_tarea,
                        "respuesta_modelo": respuesta_modelo_raw,
                        "respuesta_normalizada": respuesta_modelo_norm,
                        "tiempo_inferencia": tiempo_inferencia,
                        "prompt_tokens": respuesta_data["prompt_tokens"] if respuesta_data else 0,
                        "eval_tokens": respuesta_data["eval_tokens"] if respuesta_data else 0,
                    }
                    resultados_agregados_por_tarea[task_name].append(resultado_individual)
                    time.sleep(0.1) 
        try:
            os.remove(image_path)
        except OSError as e:
            print(f"Aviso: No se pudo eliminar el archivo temporal {image_path}: {e}")

    print(f"\n\n=======================================================")
    print(f"====== CÁLCULO FINAL DE MÉTRICAS PARA TODAS LAS TAREAS ======")
    print(f"=======================================================")
    
    evaluacion_completa = {}
    
    task_order = ["hF1", "Hallucination", "PDS"] 
    processed_tasks = set(resultados_agregados_por_tarea.keys())
    
    all_tasks_ordered = [t for t in task_order if t in processed_tasks] + \
                        [t for t in processed_tasks if t not in task_order]

    for task_name in all_tasks_ordered:
        resultados_detallados = resultados_agregados_por_tarea[task_name]
        #Metricas globales
        metricas = despachar_calculo_metricas(task_name, resultados_detallados, arbol_taxonomia, metricas_base=evaluacion_completa)
        times = [r['tiempo_inferencia'] for r in resultados_detallados if 'tiempo_inferencia' in r]
        if times:
            metricas['tiempo_medio_inferencia_s'] = round(sum(times) / len(times), 2)
        imprimir_resumen_consola(task_name, modelo_a_evaluar, metricas)
        metricas_por_prompt = {}
        resultados_por_prompt = {}
        for resultado in resultados_detallados:
            prompt = resultado['prompt']
            if prompt not in resultados_por_prompt:
                resultados_por_prompt[prompt] = []
            resultados_por_prompt[prompt].append(resultado)
        
        for idx, (prompt, resultados_prompt) in enumerate(resultados_por_prompt.items(), 1):
            metricas_prompt = despachar_calculo_metricas(task_name, resultados_prompt, arbol_taxonomia, metricas_base=evaluacion_completa)
            prompt_key = f"Prompt_{idx}"
            metricas_por_prompt[prompt_key] = {
                "prompt_text": prompt,
                "metricas": metricas_prompt
            }
        evaluacion_completa[task_name] = {
            "metricas": metricas,
            "metricas_por_prompt": metricas_por_prompt,
            "resultados_detallados": resultados_detallados
        }

    if not evaluacion_completa:
        print("\nNo se procesó ninguna imagen con un JSON válido o ninguna tarea coincide con la configuración. No se generará ningún reporte.")
    else:
        modelo_para_filename = modelo_a_evaluar.replace(":", "-")

        nombre_base, extension = os.path.splitext(OUTPUT_FILE)
        nombre_archivo_dinamico = f"{nombre_base}_{modelo_para_filename}{extension}"

        # --- CAMBIO REALIZADO ---
        # Usamos el prefijo definido en el config.yaml en lugar de "reportes/"
        prefix = s3_config.get('output_prefix', 'reportes/') # Usa 'reportes/' solo si falla el config
        ruta_salida_final = f"{prefix}{nombre_archivo_dinamico}" 
        # ------------------------

        reporte_final = generar_reporte_final(config, evaluacion_completa, total_prompt_tokens, total_eval_tokens)
        
        print(f"\nSubiendo resultados a S3 en 's3://{s3_config['output_bucket']}/{ruta_salida_final}'...")

        put_json_s3(s3_config['output_bucket'], ruta_salida_final, reporte_final)
        
        print(f"Resultados de las tareas ejecutadas guardados en S3.")
        print(f"\n--- RESUMEN DE TOKENS ---")
        print(f"Total tokens de entrada (prompt): {total_prompt_tokens}")
        print(f"Total tokens de salida (generados): {total_eval_tokens}")
        print(f"Total tokens: {total_prompt_tokens + total_eval_tokens}")
    print("--- PROCESO DE EVALUACIÓN FINALIZADO ---")


if __name__ == "__main__":
    main()