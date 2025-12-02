# inference_pipeline.py
import os
import numpy as np
from PIL import Image
from collections import Counter
from typing import Dict, List, Optional
import time

# Imports relativos (asumiendo estructura de paquete)
from .sam3_wrapper import SAM3Segmenter
from .grid_crop_generator import GridCropGenerator
from .metaclip_embedder import MetaCLIPEmbedder
from .retriever import DamageRAGRetriever

class MultimodalRAGPipeline:
    """
    Pipeline End-to-End para RAG de Alta Resolución.
    Flujo: Imagen -> SAM3 -> Grid Crops -> MetaCLIP -> FAISS -> Contexto Agregado.
    """
    
    def __init__(self, config: Dict):
        print("\n🚀 [Pipeline] Inicializando Multimodal RAG Pipeline...")
        self.config = config
        rag_conf = config.get("rag_config", {})
        
        # 1. Segmentador (Pesado, carga en GPU)
        # Pasamos la config específica si existe
        self.sam3 = SAM3Segmenter(config.get("sam3_config", {}))
        
        # 2. Generador de Grid
        self.grid_generator = GridCropGenerator(
            crop_size=336,  # Sincronizado con MetaCLIP
            overlap=0.25,
            min_content_ratio=0.30
        )
        
        # 3. Motor de Embeddings
        self.embedder = MetaCLIPEmbedder(verbose=True)
        
        # 4. Base de Datos Vectorial (MODIFICADO)
        rag_conf = config.get("rag_config", {})
        base_path = rag_conf.get("index_path", "")
        
        # Construimos las rutas completas uniendo la carpeta + el nombre del fichero
        index_path = os.path.join(base_path, rag_conf.get("index_filename", ""))
        meta_path = os.path.join(base_path, rag_conf.get("metadata_filename", ""))
        conf_path = os.path.join(base_path, rag_conf.get("config_filename", "")) # <--- NUEVO
        
        # Le pasamos el config_path al Retriever
        from pathlib import Path
        self.retriever = DamageRAGRetriever(
            Path(index_path), 
            Path(meta_path), 
            config_path=Path(conf_path) if os.path.exists(conf_path) else None
        )
        
        # Hiperparámetros de Inferencia
        self.top_k_crop = rag_conf.get("top_k_per_crop", 3)
        self.similarity_threshold = rag_conf.get("similarity_threshold", 0.65) # 0.65 es un buen punto de corte para CLIP/MetaCLIP
        
        print("✅ [Pipeline] Sistema listo para inferencia.\n")

    def run(self, image_path: str) -> str:
        """
        Ejecuta el análisis completo sobre una imagen.
        Retorna: Un string de contexto listo para inyectar en el prompt del VLM.
        """
        start_t = time.time()
        filename = os.path.basename(image_path)
        print(f"📸 [Pipeline] Procesando: {filename}")

        # PASO A: Segmentación (Aislar el coche)
        # sam3.process_image devuelve (PIL.Image, bbox_list)
        masked_image, bbox = self.sam3.process_image(image_path)
        
        if masked_image is None:
            print("   ⚠️ Fallo en segmentación SAM3. Usando imagen completa como fallback.")
            masked_image = Image.open(image_path).convert("RGB")

            bbox = [0, 0, masked_image.width, masked_image.height]

        # ✨ OPTIMIZACIÓN DE MEMORIA: Descargar SAM3 inmediatamente
        self.sam3.unload_model()

        # PASO B: Tiling (Grid Crops)
        # grid_generator espera PIL y bbox
        crops = self.grid_generator.generate(masked_image, bbox)
        print(f"   🧩 Crops generados: {len(crops)}")

        if not crops:
            return "Note: Unable to extract valid visual segments from the image."

        # PASO C: Retrieval Loop (Embedding + Búsqueda por Crop)
        all_findings = []
        
        # ✨ Cargar MetaCLIP solo ahora
        self.embedder.load_model()
        
        try:
            for i, crop_data in enumerate(crops):
                # crop_data es dict: {'crop_array': np, 'bbox': ...}
                crop_pil = Image.fromarray(crop_data['crop_array'])
                
                # Generar embedding (Prompt neutro automático)
                emb_vector = self.embedder.generate_embedding(crop_pil)
                
                # Buscar vecinos
                results = self.retriever.search(emb_vector, k=self.top_k_crop)
                
                # Filtrar ruido (Solo nos quedamos con matches de alta confianza)
                # result['distance'] en FAISS HNSW InnerProduct suele ser similitud coseno (mayor es mejor)
                # Si tu índice es L2, menor es mejor. 
                # MetaCLIPEmbedder normaliza -> Producto Punto = Coseno. Rango [-1, 1].
                # Asumiremos similaridad coseno donde > 0.6 es relevante.
                
                relevant_results = []
                for r in results:
                    score = r.distance
                    # Ajusta esta lógica según si tu FAISS devuelve distancia L2 o Inner Product
                    # Si usaste IndexHNSWFlat con MetricInnerProduct, score es Coseno.
                    if score > self.similarity_threshold:
                        relevant_results.append(r)
                
                all_findings.extend(relevant_results)
        finally:
            # ✨ Descargar MetaCLIP inmediatamente después de usarlo
            self.embedder.unload_model()

        print(f"   🔍 Hallazgos relevantes totales: {len(all_findings)} (filtrados por umbral {self.similarity_threshold})")

        # PASO D: Agregación de Contexto (Contextualizer)
        final_context = self._aggregate_findings(all_findings, len(crops))
        
        total_time = time.time() - start_t
        print(f"   ✅ Pipeline finalizado en {total_time:.2f}s")
        
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
        clean_hits = [f for f in findings if not f.has_damage]
        
        damage_ratio = len(damage_hits) / len(findings)
        
        # 2. Identificar Zonas y Tipos
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
        # Tomar los 3 mejores matches absolutos para dar "grounding" al VLM
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