import os
import base64
import json
import io
from math import ceil


#import cv2
import networkx as nx
#import numpy as np
import yaml
#from PIL import Image, ImageEnhance

import boto3

CONFIG_FILE = "config.yaml"

def cargar_configuracion(filepath):
    if not os.path.exists(filepath):
        print(f"Error: El archivo de configuración '{filepath}' no se encontró.")
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

s3_client = boto3.client("s3")

def list_images_by_prefix(bucket: str, prefix: str):
    """Lista imágenes en S3 por prefijo."""
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.lower().endswith(('.png', '.jpg', '.jpeg')):
                yield key

def download_image_to_tmp(bucket: str, key: str) -> str:
    """Descarga imagen de S3 a /tmp."""
    local_path = os.path.join('/tmp', os.path.basename(key))
    s3_client.download_file(bucket, key, local_path)
    return local_path

def load_ground_truth(bucket: str, image_key: str):
    """Carga JSON de ground truth desde S3."""
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

def image_file_to_b64(local_path: str):
    """Convierte imagen local a base64."""
    try:
        with open(local_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        print(f"Error codificando imagen {local_path}: {e}")
        return None

def put_json_s3(bucket: str, key: str, data: dict):
    """Sube JSON a S3."""
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data, ensure_ascii=False, indent=2).encode('utf-8'),
        ContentType='application/json'
    )



def aplicar_y_codificar_distorsiones(imagen_cv, nivel):
    """
    Aplica distorsiones a una imagen en formato OpenCV y la devuelve como base64.
    """
    img_procesada = imagen_cv.copy()
    
    # Desenfoque Gaussiano
    kernel_size = 2 * nivel * 2 + 1
    img_procesada = cv2.GaussianBlur(img_procesada, (kernel_size, kernel_size), 0)

    # Ruido Gaussiano
    std_dev = nivel * 12
    ruido = np.zeros(img_procesada.shape, np.uint8)
    cv2.randn(ruido, (0, 0, 0), (std_dev, std_dev, std_dev))
    img_procesada = cv2.add(img_procesada, ruido)
    
    # Cambio de Brillo
    img_pil = Image.fromarray(cv2.cvtColor(img_procesada, cv2.COLOR_BGR2RGB))
    porcentaje_cambio = ceil(nivel / 2.0) * 15
    factor_brillo = 1.0 + (porcentaje_cambio / 100.0) if nivel % 2 != 0 else 1.0 - (porcentaje_cambio / 100.0)
    enhancer = ImageEnhance.Brightness(img_pil)
    img_pil_brillo = enhancer.enhance(factor_brillo)

    # Compresión JPEG simulada y codificación a Base64
    calidad_jpeg = 100 - (15 * nivel)
    buffer = io.BytesIO()
    img_pil_brillo.save(buffer, format="JPEG", quality=calidad_jpeg)
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return img_base64

def cargar_taxonomia_desde_archivo(ruta_archivo='taxonomia.json'): #HARDCODE
    """Carga la taxonomía desde un archivo YAML o JSON."""
    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        return json.load(f)

def construir_arbol_taxonomia():
    taxonomia = cargar_taxonomia_desde_archivo()
    PIEZAS = taxonomia['Parts']
    DAÑOS_COMUNES = taxonomia['Common damages']
    
    G = nx.DiGraph()
    G.add_node("root")
    
    for daño in DAÑOS_COMUNES:
        G.add_edge("root", daño)
        for pieza in PIEZAS:
            nodo_pieza = f"{daño}_{pieza}"
            G.add_edge(daño, nodo_pieza)
                
    return G

def image_to_base64(filepath):
    if not os.path.exists(filepath):
        print(f"Error: No se encuentra el archivo de imagen en '{filepath}'")
        return None
    with open(filepath, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')