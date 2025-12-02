"""utils_cloud.py
 
Utilidades compartidas para la ejecución del benchmark en AWS Lambda:
 - Cache global (prompts)
 - Acceso S3 (listar, descargar, subir JSON)
 - Procesamiento de batches (sin concurrencia interna aún)
 - Retries para llamadas a modelos
"""
 
from __future__ import annotations
 
import os
import io
import json
import time
import base64
from math import ceil
from typing import Any, Dict, List, Iterable
 
import boto3
import yaml
 
from models import consultar_modelo_vlm, normalizar_respuesta
 
# -------------------- Config / Globals --------------------
CONFIG_FILE = "config.yaml"
 
GLOBAL_CACHE: Dict[str, Any] = {
    "prompts": None,
}
 
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
SOFT_TIMEOUT_MS = int(os.getenv("TIMEOUT_SOFT_MS", "170000"))
MAX_WORKERS_DEFAULT = int(os.getenv("MAX_WORKERS_DEFAULT", "5"))
 
s3_client = boto3.client("s3")
 
def log_debug(msg: str):
    if LOG_LEVEL == "DEBUG":
        print(f"DEBUG: {msg}")
 
# -------------------- Cargar configuración local --------------------
def cargar_configuracion(filepath: str) -> Dict[str, Any] | None:
    if not os.path.exists(filepath):
        print(f"Error: El archivo de configuración '{filepath}' no se encontró.")
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
 
def load_local_json(filename: str) -> Any:
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Archivo local requerido no existe: {filename}")
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)
 
# Taxonomía eliminada en versión reducida (no necesaria para Hallucination)
 
# -------------------- Imagen / Distorsiones --------------------
def image_to_base64(filepath: str) -> str | None:
    if not os.path.exists(filepath):
        print(f"Error: No se encuentra el archivo de imagen en '{filepath}'")
        return None
    with open(filepath, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
 
# Función de distorsiones eliminada (PDS no soportado)
 
def image_file_to_b64(local_path: str) -> str | None:
    try:
        with open(local_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        print(f"Error codificando imagen {local_path}: {e}")
        return None
 
# -------------------- S3 utilities --------------------
def list_images_by_prefix(bucket: str, prefix: str) -> Iterable[str]:
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.lower().endswith(('.png', '.jpg', '.jpeg')):
                yield key
 
def download_image_to_tmp(bucket: str, key: str) -> str:
    local_path = os.path.join('/tmp', os.path.basename(key))
    s3_client.download_file(bucket, key, local_path)
    return local_path
 
def load_ground_truth(bucket: str, image_key: str) -> Dict[str, Any] | None:
    json_key = os.path.splitext(image_key)[0] + '.json'
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=json_key)
        data = obj['Body'].read().decode('utf-8')
        return json.loads(data)
    except s3_client.exceptions.NoSuchKey:
        return None
    except Exception as e:
        print(f"Error leyendo ground truth {json_key}: {e}")
        return None
 
def put_json_s3(bucket: str, key: str, data: Dict[str, Any]):
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data, ensure_ascii=False, indent=2).encode('utf-8'),
        ContentType='application/json'
    )
 
def chunked(iterable: Iterable[str], size: int) -> Iterable[List[str]]:
    batch: List[str] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
 
# -------------------- Modelo helpers --------------------
def build_model_params(event: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "temperature": event.get("temperature", float(os.getenv("TEMPERATURE_DEFAULT", 0.0))),
        "top_k": event.get("top_k", int(os.getenv("TOP_K_DEFAULT", 50)))
    }
 
def call_vlm_with_retry(prompt: str, image_b64: str, model_name: str, gestor: str, api_key: str | None, params: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any] | None:
    backoff = 0.5
    for _ in range(max_retries):
        resp = consultar_modelo_vlm(prompt, image_b64, model_name, params.get("temperature"), params.get("top_k"), endpoint=None, gestor=gestor, api_key=api_key)
        if resp:
            return resp
        time.sleep(backoff)
        backoff *= 2
    return None
 
# Llamadas text-only eliminadas (hF1 no soportado)
 
# -------------------- Batch processing --------------------
def process_batch(batch_index: int, image_keys: List[str], bucket: str, event: Dict[str, Any], run_id: str, acumulador: Dict[str, List[Dict[str, Any]]], token_counters: Dict[str, int], api_key: str | None, start_time: float, remaining_time_ms_fn, output_bucket: str, output_prefix: str) -> List[str]:
    errores: List[str] = []
    tareas = event.get("tareas", [])
    gestor = event.get("gestor")
    modelo = event.get("modelo")
    # Eliminados parámetros include_text_only y pds_levels
    params_modelo = build_model_params(event)
 
    for img_key in image_keys:
        if remaining_time_ms_fn() < SOFT_TIMEOUT_MS:
            errores.append("Soft timeout alcanzado antes de completar batch")
            break
        gt_map = load_ground_truth(bucket, img_key)
        if not gt_map:
            errores.append(f"Ground truth faltante para {img_key}")
            continue
        local_path = download_image_to_tmp(bucket, img_key)
        image_b64_original = image_file_to_b64(local_path)
        if not image_b64_original:
            errores.append(f"No se pudo codificar imagen {img_key}")
            continue
        try:
            os.remove(local_path)
        except OSError:
            pass
        for task_name, gt_info in gt_map.items():
            if tareas and task_name not in tareas:
                continue
            if task_name != 'Hallucination':
                # Ignoramos otras tareas
                continue
            prompts_para_tarea = GLOBAL_CACHE["prompts"].get(task_name)
            if not prompts_para_tarea:
                errores.append(f"Prompts no encontrados para tarea {task_name}")
                continue
            if isinstance(gt_info, list) and gt_info:
                ground_truth_para_tarea = gt_info[0].get('ground_truth')
            else:
                ground_truth_para_tarea = None
            if ground_truth_para_tarea is None:
                errores.append(f"ground_truth ausente para tarea {task_name} en {img_key}")
                continue
            acumulador.setdefault(task_name, [])
 
            for item in prompts_para_tarea:
                texto_prompt = item['prompt']
                resp_data = call_vlm_with_retry(texto_prompt, image_b64_original, modelo, gestor, api_key, params_modelo)
                respuesta_raw = resp_data.get('response') if resp_data else ''
                respuesta_norm = normalizar_respuesta(respuesta_raw, ground_truth_para_tarea)
                acumulador[task_name].append({
                    'imagen': img_key,
                    'prompt': texto_prompt,
                    'respuesta_cruda': respuesta_raw,
                    'respuesta_normalizada': respuesta_norm,
                    'ground_truth': ground_truth_para_tarea
                })
                if resp_data:
                    token_counters['prompt_tokens'] += resp_data.get('prompt_tokens', 0)
                    token_counters['eval_tokens'] += resp_data.get('eval_tokens', 0)
 
    partial_key = f"{output_prefix}{run_id}/batch_{batch_index}.json"
    partial_payload = {
        'run_id': run_id,
        'batch_index': batch_index,
        'imagenes_en_batch': len(image_keys),
        'tareas_presentes': list(acumulador.keys()),
        'tokens_parciales': token_counters,
        'errores': errores,
        'timestamp': time.time()
    }
    put_json_s3(output_bucket, partial_payload and partial_key, partial_payload)  # ensure key then payload
    return errores
 
 
 