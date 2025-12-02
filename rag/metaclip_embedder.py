# metaclip_embedder.py
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel
from typing import Union, Dict, Optional
from collections import Counter
import os

class MetaCLIPEmbedder:
    """
    Motor de Embeddings Unificado usando MetaCLIP 2 (ViT-L/14).
    Gestiona la fusión multimodal y la generación de prompts contextuales.
    """
    
    # Modelo oficial alineado con tu Fase 3
    MODEL_NAME = "facebook/metaclip-h14-fullcc2.5b"
    # Fallback robusto por si HuggingFace falla o no tienes acceso al repo privado
    FALLBACK_MODEL = "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"

    def __init__(self, device: Optional[str] = None, verbose: bool = False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.verbose = verbose
        
        if self.verbose:
            print(f"🔧 [MetaCLIP] Inicializando motor en {self.device} (Lazy Loading)...")

        self.processor = None
        self.model = None
        # Dimensión nativa de H-14
        self.embedding_dim = 1024 

    def load_model(self):
        """Carga el modelo en memoria si no está cargado."""
        if self.model is not None:
            return

        if self.verbose:
            print(f"   ⏳ [MetaCLIP] Cargando modelo en {self.device}...")

        try:
            self.processor = AutoProcessor.from_pretrained(self.MODEL_NAME)
            self.model = AutoModel.from_pretrained(self.MODEL_NAME).to(self.device)
            if self.verbose:
                print(f"   ✅ Modelo cargado: {self.MODEL_NAME}")
        except Exception as e:
            if self.verbose:
                print(f"   ⚠️ No se pudo cargar {self.MODEL_NAME}: {e}")
                print(f"   🔄 Usando fallback: {self.FALLBACK_MODEL}")
            self.processor = AutoProcessor.from_pretrained(self.FALLBACK_MODEL)
            self.model = AutoModel.from_pretrained(self.FALLBACK_MODEL).to(self.device)

        self.model.eval()

    def unload_model(self):
        """Descarga el modelo de la memoria y fuerza GC."""
        if self.model is not None:
            if self.verbose:
                print("   🗑️ [MetaCLIP] Descargando modelo de RAM...")
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if self.verbose:
                print("   ✅ [MetaCLIP] Memoria liberada.") 

    def build_text_description(self, metadata: Dict) -> str:
        """
        Reconstruye el prompt textual usado en Fase 2/3.
        Vital para consistencia si pasamos metadata (ej: testing).
        """
        has_damage = metadata.get('has_damage', False)
        # Limpieza robusta de tipos
        defect_types = metadata.get('defect_types', []) or metadata.get('damage_types', [])
        if isinstance(defect_types, str): defect_types = [defect_types]
        
        # Filtrar "tipos" que en realidad son etiquetas de "limpio"
        clean_keywords = {'none', 'clean', 'no damage', 'unknown', ''}
        valid_defects = [d for d in defect_types if d.lower() not in clean_keywords]
        
        zone_desc = metadata.get('zone_description', 'vehicle surface')
        zone_area = metadata.get('zone_area', 'inspection area')

        # Lógica de Negocio (Idéntica a Fase 2)
        if not has_damage or not valid_defects:
            return f"Clean surface of {zone_desc} ({zone_area}). No defects visible. Inspection grid section."
        
        # Resumen de daños
        counts = Counter(valid_defects)
        summary = " and ".join([f"{c} {t.replace('_', ' ')}" for t, c in counts.items()])
        details = ", ".join(counts.keys())
        
        return f"{summary} on {zone_desc} ({zone_area}). Specific defects visible: {details}."

    def generate_embedding(
        self, 
        image: Union[str, Image.Image, np.ndarray], 
        metadata: Optional[Dict] = None,
        force_text: Optional[str] = None
    ) -> np.ndarray:
        """
        Genera el vector de embedding unificado.
        
        Args:
            image: Path, PIL Image o Numpy array.
            metadata: Diccionario con info del daño (para indexado).
            force_text: Texto manual (para queries libres).
        
        Returns:
            Numpy array (1024,) float32 normalizado.
        """
        # 1. Normalización de Imagen
        if isinstance(image, str):
            if not os.path.exists(image):
                # Retornar vector de ceros si falla la carga (fail-safe)
                return np.zeros(self.embedding_dim, dtype=np.float32)
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")

        # 2. Determinación del Texto (Prompt)
        if force_text:
            text_prompt = force_text
        elif metadata:
            text_prompt = self.build_text_description(metadata)
        else:
            # PROMPT NEUTRO PARA INFERENCIA CIEGA (Grid Crops)
            # Esto es crucial: cuando cortamos un trozo del coche y no sabemos qué es,
            # le damos un contexto genérico para que MetaCLIP se centre en lo visual.
            text_prompt = "A detail view of a vehicle surface part during inspection."

        # 3. Inferencia
        if self.model is None:
            self.load_model()
            
        try:
            inputs = self.processor(
                text=[text_prompt],
                images=pil_image,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Normalización Individual (L2)
                img_emb = outputs.image_embeds
                img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
                
                txt_emb = outputs.text_embeds
                txt_emb = txt_emb / txt_emb.norm(p=2, dim=-1, keepdim=True)
                
                # Fusión Average (Estrategia Fase 3)
                unified = (img_emb + txt_emb) / 2.0
                
                # Renormalización Final (Vital para FAISS InnerProduct/Cosine)
                unified = unified / unified.norm(p=2, dim=-1, keepdim=True)

            return unified.cpu().numpy().flatten().astype(np.float32)

        except Exception as e:
            print(f"❌ [MetaCLIP] Error generando embedding: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)