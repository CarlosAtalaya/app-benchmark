import os
import cv2
import json
import numpy as np
import boto3
import tempfile
from PIL import Image
from tqdm import tqdm

# --- IMPORTANTE: CREDENCIALES POC HARDCODEADAS ---
# Reemplaza los valores con las credenciales reales de prueba.
POC_ACCESS_KEY = "AKIA..."
POC_SECRET_KEY = "pP/N6KAK...XZ"
REGION_NAME = "eu-west-1"

# --- CONFIGURACIÓN DE RUTAS S3 (Basadas en tu solicitud) ---
# Usamos el bucket de tu config.yaml, pero podrías hardcodearlo aquí si fuera necesario.
BUCKET_NAME = "vda-benchmark-datasets-colombia-peru-espana" 
INPUT_PREFIX_OVERRIDE = "peru/"    # Carpeta origen
OUTPUT_PREFIX_SUFFIX = "masked/"               # Sufijo para la carpeta destino
CONFIG_FILE = "config.yaml"

# --- Importamos tu wrapper y utilidades existentes ---
from rag.sam3_wrapper import SAM3Segmenter
from utils import cargar_configuracion

# --- Funciones Auxiliares de S3 (Adaptadas para usar el cliente global) ---

def s3_download(client, bucket, key, local_path):
    client.download_file(bucket, key, local_path)

def s3_upload(client, bucket, key, local_path):
    client.upload_file(local_path, bucket, key)

def s3_put_json(client, bucket, key, data):
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data, ensure_ascii=False, indent=2).encode('utf-8'),
        ContentType='application/json'
    )

def extract_polygon_from_masked_image(pil_image):
    """
    Detecta el contorno del objeto no negro y devuelve los puntos del polígono.
    (Misma lógica que la versión anterior).
    """
    img_np = np.array(pil_image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return []
    
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.001 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    polygon = approx.squeeze().tolist()
    return polygon

def main():
    print("--- INICIANDO PROCESO POC DE GENERACIÓN DE MÁSCARAS S3 ---")
    
    # 1. Inicializar S3 con credenciales POC
    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=POC_ACCESS_KEY,
            aws_secret_access_key=POC_SECRET_KEY,
            region_name=REGION_NAME
        )
        s3.list_objects_v2(Bucket=BUCKET_NAME, MaxKeys=1) # Test de conexión
        print("✅ Conexión S3 con credenciales POC establecida correctamente.")
    except Exception as e:
        print(f"❌ Error al conectar con S3 usando credenciales POC: {e}")
        return

    # 2. Cargar Configuración mínima para SAM3 (ruta del BPE)
    config = cargar_configuracion(CONFIG_FILE)
    if not config:
        print("Error: No se pudo cargar config.yaml.")
        return

    # 3. Definir Prefijos
    input_prefix = INPUT_PREFIX_OVERRIDE
    output_prefix = input_prefix.rstrip('/') + "_" + OUTPUT_PREFIX_SUFFIX
    
    print(f"Bucket: {BUCKET_NAME}")
    print(f"Input Prefix:  {input_prefix}")
    print(f"Output Prefix: {output_prefix}")

    # 4. Inicializar SAM3
    print("\nInicializando SAM3...")
    try:
        sam3_segmenter = SAM3Segmenter(config)
        sam3_segmenter.load_model()
    except Exception as e:
        print(f"Error inicializando SAM3. Asegúrate de que SAM3 y sus dependencias estén instaladas. Error: {e}")
        return

    # 5. Listar imágenes (usando el cliente S3 con credenciales POC)
    print(f"\nEscaneando imágenes en s3://{BUCKET_NAME}/{input_prefix} ...")
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=input_prefix)
    
    image_keys = []
    for page in pages:
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                image_keys.append(key)
    
    print(f"Se encontraron {len(image_keys)} imágenes.")
    
    all_polygons_summary = {}

    # 6. Procesamiento
    with tempfile.TemporaryDirectory() as temp_dir:
        for img_key in tqdm(image_keys, desc="Procesando imágenes"):
            filename = os.path.basename(img_key)
            base_name_no_ext = os.path.splitext(filename)[0]
            local_img_path = os.path.join(temp_dir, filename)
            
            # A. Descargar Imagen
            try:
                s3_download(s3, BUCKET_NAME, img_key, local_img_path)
            except Exception as e:
                print(f"Error descargando {img_key}: {e}")
                continue

            # B. Segmentar
            masked_pil, bbox = sam3_segmenter.process_image(local_img_path)
            
            if masked_pil is None:
                continue

            # C. Extraer Polígono
            polygon = extract_polygon_from_masked_image(masked_pil)
            all_polygons_summary[filename] = {
                "bbox": bbox,
                "polygon": polygon
            }

            # D. Subir Imagen Enmascarada
            local_masked_path = os.path.join(temp_dir, f"masked_{filename}")
            masked_pil.save(local_masked_path)
            output_img_key = os.path.join(output_prefix, filename) 
            s3_upload(s3, BUCKET_NAME, output_img_key, local_masked_path)

            # E. Copiar el JSON Ground Truth (si existe)
            original_json_key = os.path.join(os.path.dirname(img_key), f"{base_name_no_ext}.json")
            target_json_key = os.path.join(output_prefix, f"{base_name_no_ext}.json")
            
            try:
                s3.head_object(Bucket=BUCKET_NAME, Key=original_json_key)
                copy_source = {'Bucket': BUCKET_NAME, 'Key': original_json_key}
                s3.copy_object(CopySource=copy_source, Bucket=BUCKET_NAME, Key=target_json_key)
            except s3.exceptions.ClientError:
                pass

    # 7. Generar JSON Final de Polígonos
    print("\nGenerando fichero resumen de polígonos...")
    summary_key = os.path.join(output_prefix, "dataset_polygons_summary.json")
    s3_put_json(s3, BUCKET_NAME, summary_key, all_polygons_summary)
    
    # Limpieza final
    sam3_segmenter.unload_model()
    print(f"\n--- PROCESO POC FINALIZADO ---")

if __name__ == "__main__":
    main()