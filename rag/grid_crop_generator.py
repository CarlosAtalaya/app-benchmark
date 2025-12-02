# grid_crop_generator.py
import numpy as np
import cv2
from typing import List, Dict

class GridCropGenerator:
    """
    Generador de crops en rejilla (Grid) optimizado para inferencia.
    Diferencia vs Fase 1: Trabaja en memoria (Numpy) sin recargar de disco.
    """
    def __init__(
        self, 
        crop_size: int = 336,    # Nativo MetaCLIP H/14
        overlap: float = 0.20,   # 20% solapamiento para no perder bordes
        min_content_ratio: float = 0.30  # Mínimo % de píxeles no negros
    ):
        self.crop_size = crop_size
        self.overlap = overlap
        self.min_content_ratio = min_content_ratio

    def generate(self, image_pil, vehicle_bbox: List[int]) -> List[Dict]:
        """
        Genera la lista de crops.
        Args:
            image_pil: Imagen PIL (resultado de SAM3, con fondo negro).
            vehicle_bbox: [x, y, w, h] que define dónde está el coche.
        """
        # Convertir a numpy para slicing rápido
        img_np = np.array(image_pil)
        img_h, img_w = img_np.shape[:2]
        
        bx, by, bw, bh = vehicle_bbox
        
        # Definir el paso (stride) basado en el overlap
        stride = int(self.crop_size * (1 - self.overlap))
        
        crops = []
        
        # Definir límites de escaneo (Solo escaneamos dentro del BBox del coche + margen)
        # Esto optimiza mucho el tiempo vs escanear todo el fondo negro
        start_y = max(0, by)
        end_y = min(img_h, by + bh)
        start_x = max(0, bx)
        end_x = min(img_w, bx + bw)

        # Bucle de Grid
        # Usamos un while para asegurar control total sobre los bordes
        y = start_y
        while y < end_y:
            x = start_x
            while x < end_x:
                # Coordenadas preliminares
                x_end = x + self.crop_size
                y_end = y + self.crop_size
                
                # Gestión de bordes: Si el crop se sale, lo empujamos hacia adentro
                # (Shift back strategy) para mantener siempre crops de 336x336
                valid_x = x
                valid_y = y
                
                if x_end > img_w:
                    valid_x = max(0, img_w - self.crop_size)
                    x_end = img_w
                
                if y_end > img_h:
                    valid_y = max(0, img_h - self.crop_size)
                    y_end = img_h
                
                # Extraer crop
                crop = img_np[valid_y:y_end, valid_x:x_end]
                
                # Validación de dimensiones (Paranoia check)
                if crop.shape[0] != self.crop_size or crop.shape[1] != self.crop_size:
                    # Si la imagen original es mas pequeña que el crop, hacemos resize
                    # (Caso raro de imagen de entrada diminuta)
                    crop = cv2.resize(crop, (self.crop_size, self.crop_size))

                # --- FILTRADO DE CALIDAD ---
                # Verificar que el crop no sea puro fondo negro (resultado de SAM3)
                # Calculamos ratio de píxeles no negros
                gray_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                non_zero_pixels = cv2.countNonZero(gray_crop)
                total_pixels = self.crop_size * self.crop_size
                ratio = non_zero_pixels / total_pixels
                
                if ratio >= self.min_content_ratio:
                    crops.append({
                        'crop_array': crop,  # Numpy array listo para PIL
                        'bbox': [valid_x, valid_y, self.crop_size, self.crop_size],
                        'ratio': ratio
                    })

                # Avanzar X
                if x_end == img_w: break # Ya tocamos el borde
                x += stride
            
            # Avanzar Y
            if y + self.crop_size >= img_h: break # Ya tocamos el borde
            y += stride
            
        return crops