# rag/retriever.py

from pathlib import Path
import json
import numpy as np
import faiss
import pickle
from typing import List, Dict, Optional, TYPE_CHECKING
from dataclasses import dataclass, field

from .taxonomy_normalizer import TaxonomyNormalizer

if TYPE_CHECKING:
    from .embedder import MultimodalEmbedder


@dataclass
class SearchResult:
    """Resultado de búsqueda con metadata enriquecida (con/sin daño)"""
    index: int
    distance: float
    
    # Info básica
    image_path: str
    image_name: str = ""
    
    # ✨ NUEVO: Flag de daño
    has_damage: bool = True  # Por defecto True para compatibilidad con código legacy
    
    # Para compatibilidad con código antiguo (crops)
    crop_path: str = ""
    bbox: List[float] = field(default_factory=list)
    spatial_zone: str = ""
    
    # Defectos (AHORA OPCIONALES - pueden estar vacíos si has_damage=False)
    damage_type: str = ""                    # Tipo dominante normalizado
    damage_type_original: str = ""           # Original del dataset
    damage_type_confidence: float = 1.0      # Confianza normalización
    damage_types: List[str] = field(default_factory=list)  # ✨ Todos los tipos
    total_defects: int = 0
    defect_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Zona del vehículo (NUEVO)
    vehicle_zone: str = ""
    zone_description: str = ""
    zone_area: str = ""
    
    # Descripción textual (NUEVO)
    description: str = ""
    
    # Distribución espacial (NUEVO)
    spatial_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Metadata completa
    metadata: Dict = field(default_factory=dict)
    
    # Compatibilidad
    @property
    def all_damage_types(self) -> List[str]:
        """Alias para compatibilidad"""
        return self.damage_types


class DamageRAGRetriever:
    """
    Retriever unificado para:
    - Full Images CON daño (embeddings visuales o híbridos)
    - Full Images SIN daño (embeddings visuales o híbridos)
    - Crops (legacy, si es necesario)
    
    Detecta automáticamente el tipo por dimensión del índice.
    """
    
    def __init__(
        self,
        index_path: Path,
        metadata_path: Path,
        config_path: Path = None,
        enable_taxonomy_normalization: bool = True
    ):
        print(f"🔧 Inicializando DamageRAGRetriever...")
        
        # Cargar índice y metadata
        self.index = faiss.read_index(str(index_path))
        
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        self.embedding_dim = self.index.d
        
        # Detectar tipo de embeddings por dimensión
        self.is_hybrid = self.embedding_dim > 1024
        self.data_type = 'hybrid_embeddings' if self.is_hybrid else 'full_images'
        
        embed_type = "hybrid (visual+text)" if self.is_hybrid else "visual only"
        
        print(f"   ✅ Índice: {self.index.ntotal} vectores ({embed_type})")
        print(f"   ✅ Dimensión: {self.embedding_dim}")
        print(f"   ✅ Metadata: {len(self.metadata)} entries")
        
        # ✨ Detectar presencia de imágenes sin daño
        self.has_no_damage_samples = any(
            not m.get('has_damage', True) for m in self.metadata
        )
        
        if self.has_no_damage_samples:
            n_damage = sum(1 for m in self.metadata if m.get('has_damage', True))
            n_no_damage = sum(1 for m in self.metadata if not m.get('has_damage', True))
            print(f"   ✅ Dataset composition:")
            print(f"      - CON daño: {n_damage}")
            print(f"      - SIN daño: {n_no_damage}")
        
        # Config opcional
        self.config = {}
        if config_path and config_path.exists():
            with open(config_path) as f:
                self.config = json.load(f)
        
        # Taxonomy normalizer
        self.enable_normalization = enable_taxonomy_normalization
        if self.enable_normalization:
            self.taxonomy_normalizer = TaxonomyNormalizer()
            print(f"   ✅ Taxonomy normalizer activado")
        else:
            self.taxonomy_normalizer = None
            print(f"   ℹ️  Taxonomy normalizer desactivado")
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filters: Optional[Dict] = None,
        return_normalized: bool = True
    ) -> List[SearchResult]:
        """
        Búsqueda con metadata enriquecida y normalización
        
        Args:
            query_embedding: Vector de consulta (1024 o 1408 dims)
            k: Número de resultados
            filters: Filtros opcionales:
                # ✨ NUEVOS FILTROS
                - has_damage: bool - Filtrar por presencia de daño
                - exclude_no_damage: bool - Excluir imágenes sin daño
                - include_no_damage_only: bool - Solo imágenes sin daño
                
                # Filtros existentes
                - damage_type: str o List[str]
                - vehicle_zone: str o List[str]
                - zone_area: str o List[str]
                - spatial_zone: str o List[str] (compatibilidad crops)
                - min_defects: int
                - max_defects: int
            return_normalized: Normalizar tipos de daño
        """
        # Preparar query
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype('float32')
        
        # Verificar dimensión
        if query_embedding.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Query embedding dim ({query_embedding.shape[1]}) "
                f"!= index dim ({self.embedding_dim})"
            )
        
        # Búsqueda FAISS
        k_search = k * 5 if filters else k  # ✨ Aumentado para filtrado
        k_search = min(k_search, self.index.ntotal)
        
        distances, indices = self.index.search(query_embedding, k_search)
        
        # Construir resultados
        results = []
        
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            
            meta = self.metadata[idx]
            
            # ✨ Detectar si tiene daño
            has_damage = meta.get('has_damage', True)
            
            # ✨ Extraer tipos de daño (solo si tiene daño)
            if has_damage:
                damage_types_raw = self._extract_damage_types(meta)
                
                # Normalizar tipos de daño
                if return_normalized and self.taxonomy_normalizer:
                    damage_types = [
                        self.taxonomy_normalizer.normalize(dt)['benchmark_label']
                        for dt in damage_types_raw
                    ]
                    dominant_type = damage_types[0] if damage_types else 'Unknown'
                else:
                    damage_types = damage_types_raw
                    dominant_type = damage_types[0] if damage_types else 'unknown'
            else:
                # ✨ Sin daño
                damage_types_raw = []
                damage_types = []
                dominant_type = 'No damage'
            
            # ✨ Aplicar filtros (pasando has_damage)
            if filters and not self._apply_filters(meta, filters, damage_types, has_damage):
                continue
            
            # Construir SearchResult (unificado)
            result = SearchResult(
                index=int(idx),
                distance=float(dist),
                
                # Paths
                image_path=meta.get('source_image', meta.get('image_path', '')),
                image_name=Path(meta.get('image_path', '')).name,
                crop_path=meta.get('crop_path', ''),  # Legacy
                
                # ✨ Flag de daño
                has_damage=has_damage,
                
                # Daños (pueden estar vacíos si has_damage=False)
                damage_type=dominant_type,
                damage_type_original=damage_types_raw[0] if damage_types_raw else 'no_damage',
                damage_types=damage_types,
                total_defects=meta.get('total_defects', len(damage_types_raw)),
                defect_distribution=meta.get('defect_distribution', {}),
                
                # Geometría
                bbox=meta.get('cluster_bbox', meta.get('bbox', [])),
                spatial_zone=meta.get('spatial_zone', 'unknown'),
                
                # Zonas vehículo (NUEVO)
                vehicle_zone=meta.get('vehicle_zone', 'unknown'),
                zone_description=meta.get('zone_description', 'unknown'),
                zone_area=meta.get('zone_area', 'unknown'),
                
                # Descripción (NUEVO)
                description=meta.get('description', ''),
                
                # Distribución espacial (NUEVO)
                spatial_distribution=meta.get('spatial_distribution', {}),
                
                # Metadata completa
                metadata=meta
            )
            
            results.append(result)
            
            if len(results) >= k:
                break
        
        return results
    
    def search_hybrid(
        self,
        query_image: Path,
        query_metadata: Dict,
        multimodal_embedder: 'MultimodalEmbedder',
        k: int = 5,
        filters: Optional[Dict] = None,
        return_normalized: bool = True
    ) -> List[SearchResult]:
        """
        Búsqueda usando embedding híbrido de query
        
        Args:
            query_image: Path a imagen query
            query_metadata: Metadata (con defect_types, zone, etc.)
            multimodal_embedder: Instancia de MultimodalEmbedder
            k: Número de resultados
            filters: Filtros opcionales
        """
        # Generar embedding híbrido
        query_embedding, _ = multimodal_embedder.generate_hybrid_embedding(
            image_path=query_image,
            metadata=query_metadata,
            normalize=True
        )
        
        # Búsqueda normal
        return self.search(
            query_embedding=query_embedding,
            k=k,
            filters=filters,
            return_normalized=return_normalized
        )
    
    def _extract_damage_types(self, meta: Dict) -> List[str]:
        """Extrae tipos de daño (compatible con ambos formatos)"""
        # Formato full images
        if 'defect_types' in meta:
            return meta['defect_types']
        
        # Formato crops (legacy)
        if 'damage_types' in meta:
            return meta['damage_types']
        
        # Single type
        if 'dominant_type' in meta:
            return [meta['dominant_type']]
        
        if 'damage_type' in meta:
            return [meta['damage_type']]
        
        return []  # ✨ Cambiado: retorna lista vacía en lugar de ['unknown']
    
    def _apply_filters(
        self, 
        meta: Dict, 
        filters: Dict, 
        normalized_types: List[str] = None,
        has_damage: bool = True  # ✨ NUEVO parámetro
    ) -> bool:
        """Aplica filtros a un resultado"""
        
        # ✨ NUEVO: Filtro explícito por presencia de daño
        if 'has_damage' in filters:
            required_damage_status = filters['has_damage']
            if has_damage != required_damage_status:
                return False
        
        # ✨ NUEVO: Excluir imágenes sin daño
        if filters.get('exclude_no_damage', False):
            if not has_damage:
                return False
        
        # ✨ NUEVO: Solo imágenes sin daño
        if filters.get('include_no_damage_only', False):
            if has_damage:
                return False
        
        # Filtro por tipo de daño (solo aplica si tiene daño)
        if 'damage_type' in filters:
            if not has_damage:
                # Si no tiene daño, skip este filtro
                return True
            
            allowed = filters['damage_type']
            if isinstance(allowed, str):
                allowed = [allowed]
            
            if normalized_types:
                if not any(dt in allowed for dt in normalized_types):
                    return False
            else:
                meta_types = self._extract_damage_types(meta)
                if not any(dt in allowed for dt in meta_types):
                    return False
        
        # Filtro por zona del vehículo (NUEVO)
        if 'vehicle_zone' in filters:
            allowed_zones = filters['vehicle_zone']
            if isinstance(allowed_zones, str):
                allowed_zones = [allowed_zones]
            if meta.get('vehicle_zone', 'unknown') not in allowed_zones:
                return False
        
        # Filtro por área de zona (NUEVO)
        if 'zone_area' in filters:
            allowed_areas = filters['zone_area']
            if isinstance(allowed_areas, str):
                allowed_areas = [allowed_areas]
            if meta.get('zone_area', 'unknown') not in allowed_areas:
                return False
        
        # Filtro por zona espacial (legacy)
        if 'spatial_zone' in filters:
            allowed = filters['spatial_zone']
            if isinstance(allowed, str):
                allowed = [allowed]
            if meta.get('spatial_zone', 'unknown') not in allowed:
                return False
        
        # Filtro por número de defectos (NUEVO - solo aplica si tiene daño)
        if 'min_defects' in filters:
            if has_damage and meta.get('total_defects', 0) < filters['min_defects']:
                return False
        
        if 'max_defects' in filters:
            if has_damage and meta.get('total_defects', 0) > filters['max_defects']:
                return False
        
        return True
    
    def build_rag_context(
        self,
        results: List[SearchResult],
        max_examples: int = 3,
        include_confidence: bool = False,
        include_spatial: bool = True
    ) -> str:
        """
        Construye contexto RAG enriquecido
        
        Args:
            results: Lista de SearchResult
            max_examples: Número máximo de ejemplos
            include_confidence: Mostrar confianza normalización
            include_spatial: Incluir distribución espacial
        """
        if not results:
            return "No similar examples found in the database."
        
        lines = ["## 🔍 Similar Verified Cases from Database:\n"]
        
        for i, r in enumerate(results[:max_examples], 1):
            lines.append(f"\n### Example {i}:")
            
            # Descripción textual (si existe)
            if r.description:
                lines.append(f"**Description**: {r.description}")
            
            # ✨ NUEVO: Manejo diferenciado por has_damage
            if r.has_damage:
                # Caso CON daño
                if r.damage_types and len(r.damage_types) > 1:
                    types_str = ", ".join(set(r.damage_types))
                    lines.append(f"**Damage types**: {types_str}")
                else:
                    lines.append(f"**Damage type**: {r.damage_type}")
                
                # Detalles de defectos
                if r.total_defects:
                    lines.append(f"**Total defects**: {r.total_defects}")
            else:
                # ✨ Caso SIN daño
                lines.append(f"**Damage status**: No visible damage")
                lines.append(f"**Quality**: Clean surface verified")
            
            # Similitud
            similarity_pct = (1 - r.distance) * 100
            lines.append(f"**Similarity**: {similarity_pct:.1f}%")
            
            # Info de zona (común para ambos)
            if r.zone_description != 'unknown':
                lines.append(f"**Vehicle zone**: {r.zone_description} ({r.zone_area})")
            elif r.spatial_zone != 'unknown':
                lines.append(f"**Vehicle area**: {self._format_zone(r.spatial_zone)}")
            
            # Distribución espacial (NUEVO - opcional, solo para con daño)
            if include_spatial and r.has_damage and r.spatial_distribution:
                spatial_info = self._format_spatial_distribution(r.spatial_distribution)
                lines.append(f"**Spatial distribution**: {spatial_info}")
            
            # Confianza (opcional)
            if include_confidence and r.has_damage and r.damage_type_confidence < 0.95:
                lines.append(
                    f"**Note**: Approximate match (original: {r.damage_type_original})"
                )
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_zone(self, zone: str) -> str:
        """Formatea zonas espaciales (legacy crops)"""
        zones = {
            "top_left": "Upper left", "top_center": "Upper center", "top_right": "Upper right",
            "middle_left": "Left side", "middle_center": "Center", "middle_right": "Right side",
            "bottom_left": "Lower left", "bottom_center": "Lower center", "bottom_right": "Lower right"
        }
        return zones.get(zone, zone)
    
    def _format_spatial_distribution(self, spatial_dist: Dict[str, int]) -> str:
        """Formatea distribución espacial (NUEVO)"""
        parts = []
        for zone, count in sorted(spatial_dist.items(), key=lambda x: -x[1]):
            zone_formatted = zone.replace('_', ' ').title()
            plural = "s" if count > 1 else ""
            parts.append(f"{count} defect{plural} in {zone_formatted}")
        
        return ", ".join(parts) if parts else "distributed across image"
    
    def get_stats(self) -> Dict:
        """Estadísticas del índice"""
        stats = {
            'n_vectors': self.index.ntotal,
            'embedding_dim': self.embedding_dim,
            'is_hybrid': self.is_hybrid,
            'data_type': self.data_type,
            'normalization_enabled': self.enable_normalization,
            'has_no_damage_samples': self.has_no_damage_samples  # ✨ NUEVO
        }
        
        # Estadísticas de dataset
        from collections import Counter
        
        all_types = []
        all_zones = []
        total_defects = 0
        
        # ✨ Separar por tipo
        damage_samples = 0
        no_damage_samples = 0
        
        for m in self.metadata:
            has_damage = m.get('has_damage', True)
            
            if has_damage:
                damage_samples += 1
                types = self._extract_damage_types(m)
                all_types.extend(types)
                
                # Total defectos
                if 'total_defects' in m:
                    total_defects += m['total_defects']
            else:
                no_damage_samples += 1
            
            # Zona vehículo (para ambos)
            if 'vehicle_zone' in m:
                all_zones.append(m['vehicle_zone'])
        
        stats['dataset_stats'] = {
            'total_images': len(self.metadata),
            'damage_images': damage_samples,  # ✨ NUEVO
            'no_damage_images': no_damage_samples,  # ✨ NUEVO
            'total_defects': total_defects,
            'avg_defects_per_damage_image': total_defects / damage_samples if damage_samples else 0,  # ✨ Ajustado
            'damage_type_distribution': dict(Counter(all_types)),
            'zone_distribution': dict(Counter(all_zones)) if all_zones else {}
        }
        
        return stats