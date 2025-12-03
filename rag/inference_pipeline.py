# rag/inference_pipeline.py
import os
import time
from PIL import Image
from typing import Dict, List, Optional
from pathlib import Path
from collections import Counter  # <--- SOLUCIÓN AL ERROR

# Imports relativos
from .grid_crop_generator import GridCropGenerator
from .metaclip_embedder import MetaCLIPEmbedder
from .retriever import DamageRAGRetriever

class MultimodalRAGPipeline:
    """
    Pipeline OPTIMIZADO para RAG sobre imágenes pre-procesadas (Offline SAM).
    Flujo: Imagen Enmascarada + JSON Metadatos -> Grid Crops -> MetaCLIP -> FAISS -> Contexto.
    """
    
    def __init__(self, config: Dict, polygons_summary: Dict):
        print("\n🚀 [Pipeline] Inicializando Multimodal RAG Pipeline (Modo Offline SAM)...")
        self.config = config
        self.polygons_data = polygons_summary # Diccionario cargado del dataset_polygons_summary.json
        
        rag_conf = config.get("rag_config", {})
        
        # 1. Generador de Grid
        self.grid_generator = GridCropGenerator(
            crop_size=336,
            overlap=0.25,
            min_content_ratio=0.30
        )
        
        # 2. Motor de Embeddings
        self.embedder = MetaCLIPEmbedder(verbose=True)
        
        # 3. Base de Datos Vectorial
        base_path = rag_conf.get("index_path", "")
        index_path = os.path.join(base_path, rag_conf.get("index_filename", ""))
        meta_path = os.path.join(base_path, rag_conf.get("metadata_filename", ""))
        conf_path = os.path.join(base_path, rag_conf.get("config_filename", ""))
        
        self.retriever = DamageRAGRetriever(
            Path(index_path), 
            Path(meta_path), 
            config_path=Path(conf_path) if os.path.exists(conf_path) else None
        )
        
        self.top_k_crop = rag_conf.get("top_k_per_crop", 3)
        self.similarity_threshold = rag_conf.get("similarity_threshold", 0.60)
        
        print("✅ [Pipeline] Sistema listo para inferencia sobre dataset enmascarado.\n")

    def run(self, image_path: str) -> str:
        """
        Ejecuta el análisis usando la imagen ya enmascarada y sus metadatos pre-calculados.
        """
        start_t = time.time()
        filename = os.path.basename(image_path)
        
        print(f"📸 [Pipeline] Procesando RAG para: {filename}")

        # PASO A: Recuperar Geometría (BBox) del JSON Pre-calculado
        img_metadata = self.polygons_data.get(filename)
        
        bbox = None
        if img_metadata:
            bbox = img_metadata.get("bbox") # Esperamos [x, y, w, h]
        
        if not bbox:
            print(f"   ⚠️ ADVERTENCIA: No hay bbox para {filename}. Usando imagen completa.")
            with Image.open(image_path) as img:
                w, h = img.size
                bbox = [0, 0, w, h]

        # Cargar la imagen (que ya es la masked desde la carpeta _masked)
        try:
            masked_image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"   ❌ Error abriendo imagen {filename}: {e}")
            return ""

        # PASO B: Tiling (Grid Crops)
        crops = self.grid_generator.generate(masked_image, bbox)
        print(f"   🧩 Crops generados: {len(crops)}")

        if not crops:
            return "Note: Unable to extract valid visual segments from the image."

        # PASO C: Retrieval Loop (Embedding + Búsqueda)
        all_findings = []
        self.embedder.load_model()
        
        try:
            for i, crop_data in enumerate(crops):
                crop_pil = Image.fromarray(crop_data['crop_array'])
                emb_vector = self.embedder.generate_embedding(crop_pil)
                results = self.retriever.search(emb_vector, k=self.top_k_crop)
                
                relevant_results = []
                for r in results:
                    if r.distance > self.similarity_threshold:
                        relevant_results.append(r)
                
                all_findings.extend(relevant_results)
        finally:
            self.embedder.unload_model()

        # PASO D: Agregación
        final_context = self._aggregate_findings(all_findings, len(crops))
        
        print(f"   ✅ Contexto generado en {time.time() - start_t:.2f}s")
        return final_context

    def _aggregate_findings(self, findings: List[Dict], total_crops: int) -> str:
        """
        Convierte la lista de hallazgos crudos en un texto narrativo técnico.
        """
        if not findings:
            return (
                "## Historical Database Analysis:\n"
                "No specific defects or similar damage patterns were found in the reference database "
                "with high confidence. The surface features appear unique or generic."
            )

        # 1. Estadísticas Globales
        damage_hits = [f for f in findings if f.has_damage]
        
        damage_ratio = len(damage_hits) / len(findings)
        
        # 2. Identificar Zonas y Tipos (Usando Counter importado)
        detected_zones = Counter([f.zone_description for f in damage_hits])
        detected_types = Counter([f.damage_type for f in damage_hits])
        
        # 3. Construcción del Prompt
        lines = ["## RAG Analysis (Visual Similarity Retrieval):"]
        
        if damage_ratio > 0.3: # Si más del 30% de lo recuperado es daño
            lines.append(f"⚠️ **Potential Defects Detected**: The system retrieved {len(damage_hits)} visual matches of damaged parts similar to this vehicle.")
            
            if detected_types:
                lines.append("\n**Most likely defect types identified:**")
                for dtype, count in detected_types.most_common(3):
                    lines.append(f"- {dtype} (matched {count} times with high similarity)")
            
            if detected_zones:
                lines.append(f"\n**Affected Areas:** {', '.join(detected_zones.keys())}")
                
        else:
            lines.append("✅ **Clean Surface Indicators**: The majority of retrieved visual segments match 'Clean/No Damage' examples from the database.")
            lines.append("The visual texture is consistent with undamaged vehicle panels.")

        # 4. Ejemplos de Referencia (Evidence)
        best_matches = sorted(findings, key=lambda x: x.distance, reverse=True)[:3]
        
        if best_matches:
            lines.append("\n**Top Reference Cases from Database:**")
            for i, m in enumerate(best_matches, 1):
                status = "Damaged" if m.has_damage else "Clean"
                detail = m.damage_type
                zone = m.zone_description
                conf = m.distance * 100
                lines.append(f"{i}. [{status}] {detail} on {zone} (Confidence: {conf:.1f}%)")

        return "\n".join(lines)