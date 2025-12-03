# rag/__init__.py

# Exponemos las clases principales al exterior
from .retriever import DamageRAGRetriever, SearchResult
from .metaclip_embedder import MetaCLIPEmbedder
from .inference_pipeline import MultimodalRAGPipeline

# Opcional: Si quieres acceder a componentes internos desde fuera
# from .sam3_wrapper import SAM3Segmenter
from .taxonomy_normalizer import TaxonomyNormalizer