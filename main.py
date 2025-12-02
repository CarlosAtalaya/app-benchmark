# main.py
import os
import json
import time
import sys
import cv2 

# --- Importaciones del proyecto original ---
from utils import (
    cargar_configuracion,
    list_images_by_prefix,
    download_image_to_tmp,
    load_ground_truth,
    put_json_s3,
    image_file_to_b64,
    aplicar_y_codificar_distorsiones,
    construir_arbol_taxonomia
)

from models import (
    consultar_modelo_vlm,
    normalizar_respuesta
)
from metrics import despachar_calculo_metricas
from reporting import imprimir_resumen_consola, generar_reporte_final

# --- Importación Robusta del RAG ---
try:
    from rag.inference_pipeline import MultimodalRAGPipeline
    RAG_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Módulo RAG no disponible o dependencias faltantes: {e}")
    print("⚠️  Se ejecutará SIN contexto RAG.")
    RAG_AVAILABLE = False


CONFIG_FILE = "config.yaml"

def main():
    print("--- INICIANDO PROCESO DE EVALUACIÓN CON RAG MULTIMODAL ---")
    
    # 1. Cargar Configuración
    config = cargar_configuracion(CONFIG_FILE)
    if not config:
        print("Error: No se pudo cargar config.yaml.")
        return
    
    s3_config = config.get("s3_config")
    if not s3_config:
        print("Error: Falta configuración de S3 en el config.yaml.")
        return

    # 2. Configurar APIs y Modelos
    gestor = config.get("gestor")
    api_key = None
    api_endpoint = None
    
    # Mapeo de configuración según gestor
    if gestor == "ollama":
        api_endpoint = config.get("ollama_config", {}).get("endpoint")
    elif gestor == "lm_studio":
        api_endpoint = config.get("lm_studio_config", {}).get("endpoint")
    elif gestor == "vllm":
        api_endpoint = config.get("vllm_config", {}).get("endpoint")
    elif gestor == 'qwen3':
        api_endpoint = config.get("qwen_config", {}).get("endpoint")
    elif gestor == "openai":
        api_key = config.get("openai_config", {}).get("api_key")
    elif gestor == "gemini":
        api_key = config.get("gemini_config", {}).get("api_key")
    elif gestor == "nova":
        api_key = config.get("nova_config", {}).get("api_key")

    print(f"Usando gestor: '{gestor}' | Endpoint: {api_endpoint or 'N/A'}")
    
    # 3. Cargar Prompts Globales
    prompts_file = config.get("prompts_file", "prompts.json")
    try:
        with open(prompts_file, 'r', encoding='utf-8') as f:
            prompts_globales = json.load(f)
        print(f"Prompts cargados desde {prompts_file}")
    except Exception as e:
        print(f"Error cargando prompts: {e}")
        return

    # ==========================================
    # 4. INICIALIZACIÓN DEL RAG
    # ==========================================
    rag_pipeline = None
    if RAG_AVAILABLE and config.get("rag_config", {}).get("enabled"):
        try:
            print("\n🔧 Inicializando Pipeline RAG (Assets Locales)...")
            rag_pipeline = MultimodalRAGPipeline(config)
            print("✅ RAG Inicializado correctamente.\n")
        except Exception as e:
            print(f"❌ Error crítico inicializando RAG: {e}")
            print("➡️  Continuando ejecución SIN RAG.\n")
            rag_pipeline = None

    # Preparar estructuras
    modelo_a_evaluar = config.get("modelo_a_evaluar")
    resultados_agregados_por_tarea = {}
    total_prompt_tokens = 0
    total_eval_tokens = 0
    tareas_a_ejecutar = config.get("tareas_a_ejecutar", [])

    print("Construyendo árbol de taxonomía...")
    arbol_taxonomia = construir_arbol_taxonomia()

    # 5. Bucle Principal de Imágenes
    bucket = s3_config['input_bucket']
    prefix = s3_config['input_prefix']
    
    print(f"Escaneando imágenes en s3://{bucket}/{prefix} ...")
    imagenes_s3 = list(list_images_by_prefix(bucket, prefix))
    print(f"Se encontraron {len(imagenes_s3)} imágenes.")
    
    for image_key in imagenes_s3:
        filename = os.path.basename(image_key)
        print(f"\n--- Procesando: {filename} ---")
        
        # A. Descargar Imagen y GT
        try:
            image_path = download_image_to_tmp(bucket, image_key)
            ground_truths = load_ground_truth(bucket, image_key)
        except Exception as e:
            print(f"Error descargando datos de S3 para {filename}: {e}")
            continue
            
        if not ground_truths:
            print(f"Advertencia: No se encontró JSON de Ground Truth para {filename}. Saltando.")
            if os.path.exists(image_path): os.remove(image_path)
            continue

        # B. Generar Contexto RAG
        rag_context = ""
        rag_used = False
        if rag_pipeline:
            try:
                print("   🔍 Ejecutando análisis RAG...")
                rag_context = rag_pipeline.run(image_path)
                rag_used = True
                print("   ✅ Contexto RAG generado.")
            except Exception as e:
                print(f"   ⚠️ Fallo en inferencia RAG: {e}")
                rag_context = ""

        # C. Procesar Tareas
        imagen_b64 = image_file_to_b64(image_path)
        if not imagen_b64:
            print(f"Error al convertir imagen a Base64. Saltando.")
            if os.path.exists(image_path): os.remove(image_path)
            continue
        
        imagen_cv = None
        if "PDS" in tareas_a_ejecutar:
            imagen_cv = cv2.imread(image_path)

        for task_name, gt_list in ground_truths.items():
            if tareas_a_ejecutar and task_name not in tareas_a_ejecutar:
                continue
                
            prompts_tarea = prompts_globales.get(task_name, [])
            gt_data = gt_list[0].get('ground_truth') if gt_list else None
            
            resultados_agregados_por_tarea.setdefault(task_name, [])

            # --- CASO 1: Tarea PDS ---
            if task_name == "PDS" and imagen_cv is not None:
                for i, p_item in enumerate(prompts_tarea):
                    max_nivel = p_item.get("nivel", 3)
                    prompt_base = p_item['prompt']
                    
                    # Inyección RAG
                    prompt_final = prompt_base
                    if rag_used and rag_context:
                        prompt_final += f"\n\n### ADDITIONAL CONTEXT FROM SIMILAR CASES:\n{rag_context}"

                    for nivel in range(1, max_nivel + 1):
                        print(f"   -> PDS | Prompt {i+1} | Nivel {nivel}")
                        
                        img_dist_b64 = aplicar_y_codificar_distorsiones(imagen_cv, nivel)
                        
                        start_t = time.perf_counter() # ✅ RESTAURADO: Medición de tiempo
                        resp = consultar_modelo_vlm(
                            prompt=prompt_final,
                            image_b64=img_dist_b64,
                            model_name=modelo_a_evaluar,
                            temperature=config.get("hiperparametros", {}).get("temperature", 0),
                            top_k=config.get("hiperparametros", {}).get("top_k", 50),
                            endpoint=api_endpoint,
                            gestor=gestor,
                            api_key=api_key
                        )
                        end_t = time.perf_counter()
                        
                        texto_resp = resp["response"] if resp else None
                        norm_resp = normalizar_respuesta(texto_resp, gt_data)
                        
                        if resp:
                            total_prompt_tokens += resp.get("prompt_tokens", 0)
                            total_eval_tokens += resp.get("eval_tokens", 0)

                        resultados_agregados_por_tarea[task_name].append({
                            "imagen": filename,
                            "prompt": prompt_base,
                            "ground_truth": gt_data,
                            "respuesta_modelo": texto_resp,
                            "respuesta_normalizada": norm_resp,
                            "nivel": nivel,
                            "tiempo_inferencia": end_t - start_t, # ✅ RESTAURADO: Guardado de tiempo
                            "rag_used": rag_used,
                            "rag_context_preview": rag_context[:100] + "..." if rag_used else None
                        })
                        time.sleep(0.1)

            # --- CASO 2: Tareas Estándar ---
            else:
                for i, p_item in enumerate(prompts_tarea):
                    print(f"   -> {task_name} | Prompt {i+1}")
                    prompt_base = p_item['prompt']
                    
                    # Inyección RAG
                    prompt_final = prompt_base
                    if rag_used and rag_context:
                        prompt_final += f"\n\n### ADDITIONAL CONTEXT FROM SIMILAR CASES:\n{rag_context}"

                    start_t = time.perf_counter()
                    resp = consultar_modelo_vlm(
                        prompt=prompt_final,
                        image_b64=imagen_b64,
                        model_name=modelo_a_evaluar,
                        temperature=config.get("hiperparametros", {}).get("temperature", 0),
                        top_k=config.get("hiperparametros", {}).get("top_k", 50),
                        endpoint=api_endpoint,
                        gestor=gestor,
                        api_key=api_key
                    )
                    end_t = time.perf_counter()

                    texto_resp = resp["response"] if resp else None
                    norm_resp = normalizar_respuesta(texto_resp, gt_data)
                    
                    if resp:
                        total_prompt_tokens += resp.get("prompt_tokens", 0)
                        total_eval_tokens += resp.get("eval_tokens", 0)

                    resultados_agregados_por_tarea[task_name].append({
                        "imagen": filename,
                        "prompt": prompt_base,
                        "ground_truth": gt_data,
                        "respuesta_modelo": texto_resp,
                        "respuesta_normalizada": norm_resp,
                        "tiempo_inferencia": end_t - start_t,
                        "rag_used": rag_used,
                        "rag_context_preview": rag_context[:100] + "..." if rag_used else None
                    })
                    print(f"      Respuesta: {norm_resp}")
                    time.sleep(0.1)

        if os.path.exists(image_path):
            os.remove(image_path)

    # 6. Generación de Reportes y Subida a S3
    print("\n\n=======================================================")
    print(f"====== CÁLCULO FINAL DE MÉTRICAS PARA TODAS LAS TAREAS ======")
    print(f"=======================================================")
    
    evaluacion_completa = {}
    
    # ✅ RESTAURADO: Lógica de ordenación de tareas original
    task_order = ["hF1", "Hallucination", "PDS"] 
    processed_tasks = set(resultados_agregados_por_tarea.keys())
    
    all_tasks_ordered = [t for t in task_order if t in processed_tasks] + \
                        [t for t in processed_tasks if t not in task_order]

    for task_name in all_tasks_ordered:
        resultados_detallados = resultados_agregados_por_tarea[task_name]
        
        # Metricas globales
        metricas = despachar_calculo_metricas(task_name, resultados_detallados, arbol_taxonomia, metricas_base=evaluacion_completa)
        
        # ✅ RESTAURADO: Cálculo de tiempo medio de inferencia
        times = [r['tiempo_inferencia'] for r in resultados_detallados if 'tiempo_inferencia' in r]
        if times:
            metricas['tiempo_medio_inferencia_s'] = round(sum(times) / len(times), 2)
            
        imprimir_resumen_consola(task_name, modelo_a_evaluar, metricas)
        
        # ✅ RESTAURADO: Desglose de métricas por prompt individual
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
        print("\nNo se procesó ninguna imagen válida o ninguna tarea.")
    else:
        # ✅ Mantenida la nomenclatura de salida con S3
        output_filename = config.get("output_file", "resultados.json")
        if config.get("rag_config", {}).get("enabled"):
            base, ext = os.path.splitext(output_filename)
            output_filename = f"{base}_RAG{ext}"
            
        prefix = s3_config.get('output_prefix', 'reportes/') 
        ruta_salida_final = f"{prefix}{output_filename}"
        
        reporte_final = generar_reporte_final(config, evaluacion_completa, total_prompt_tokens, total_eval_tokens)
        
        print(f"\nSubiendo resultados a S3 en 's3://{s3_config['output_bucket']}/{ruta_salida_final}'...")
        put_json_s3(s3_config['output_bucket'], ruta_salida_final, reporte_final)
        
        print(f"Resultados guardados en S3.")
        print(f"\n--- RESUMEN DE TOKENS ---")
        print(f"Total tokens de entrada (prompt): {total_prompt_tokens}")
        print(f"Total tokens de salida (generados): {total_eval_tokens}")
        print(f"Total tokens: {total_prompt_tokens + total_eval_tokens}")
        
    print("--- PROCESO DE EVALUACIÓN FINALIZADO ---")


if __name__ == "__main__":
    main()