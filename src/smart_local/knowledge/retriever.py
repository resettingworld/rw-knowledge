from typing import List, Optional, Union
from PIL import Image

from smart_local.knowledge.schema import RetrievalResult
from smart_local.knowledge.store import KnowledgeStore
from smart_local.models.embedding import EmbeddingGemma

class MultimodalRetriever:
    """
    Multimodal retriever using EmbeddingGemma 2 with Cross-Lingual & Hybrid Search.
    Handles Catalan/English queries targeting Spanish digital library documents.
    """
    def __init__(self, embedder: Optional[EmbeddingGemma] = None):
        self.embedder = embedder or EmbeddingGemma()

        # Cross-lingual dictionary for survival concepts (Catalan -> Spanish)
        self.cross_lingual_dict = {
            'aigua': 'agua potabilizar hervir',
            'sed': 'sed deshidratación',
            'selva': 'selva jungla tropical lianas',
            'jungla': 'jungla selva',
            'bosc': 'bosque arboles',
            'obtenir': 'conseguir obtener recolectar',
            'trobar': 'encontrar buscar hallar',
            'foc': 'fuego fogata chispa pedernal',
            'refugi': 'refugio cobertizo vivac ramas',
            'menjar': 'comida alimentos cazar plantas comestibles',
            'ferida': 'herida curar vendaje desinfección',
            'ferides': 'heridas curar vendaje',
            'sagnat': 'hemorragia sangrado torniquete',
            'hemorragia': 'hemorragia sangrado presion',
            'cremada': 'quemadura enfriar',
            'fractura': 'fractura entablillar inmovilizar',
            'verí': 'veneno serpiente picadura mordedura suero',
            'picada': 'picadura mordedura insecto araña alacran',
            'motxilla': 'mochila equipo kit emergencia 72 horas',
            'orientar': 'orientación brújula estrellas sol norte',
            'brúixola': 'brújula orientación norte mapa',
            'fred': 'frío hipotermia abrigo manta termica',
            'calor': 'calor golpe insolación sombra',
            'deshidratacio': 'deshidratacion sed agua suero',
            'deshidratació': 'deshidratacion sed agua suero',
            'supervivencia': 'supervivencia preparacionismo emergencia',
            'supervivència': 'supervivencia preparacionismo emergencia',
            'farmaciola': 'botiquín medicinas gasas primeros auxilios'
        }

    def expand_query(self, query: str) -> str:
        """
        Expands query terms cross-lingually to bridge language gap with the library manuals.
        """
        tokens = query.lower().replace("?", " ").replace("!", " ").replace(".", " ").replace(",", " ").split()
        expanded_parts = list(tokens)

        for t in tokens:
            t_clean = t.strip()
            if t_clean in self.cross_lingual_dict:
                expanded_parts.append(self.cross_lingual_dict[t_clean])

        return " ".join(expanded_parts)

    def retrieve(
        self,
        store: KnowledgeStore,
        query: Union[str, Image.Image, bytes],
        top_k: int = 5,
        min_score: Optional[float] = None,
        modality: Optional[str] = None
    ) -> List[RetrievalResult]:
        """
        Computes embedding for query (with cross-lingual expansion for text)
        and executes hybrid semantic + keyword search in the knowledge store.
        """
        raw_text_query = None

        if isinstance(query, Image.Image):
            query_vec = self.embedder.embed_image(query)
        elif isinstance(query, bytes):
            try:
                decoded = query.decode("utf-8")
                raw_text_query = decoded
                expanded = self.expand_query(decoded)
                query_vec = self.embedder.embed_text(expanded)
            except UnicodeDecodeError:
                query_vec = self.embedder.embed_audio(query)
        elif isinstance(query, str):
            if query.startswith("data:image/") or (query.endswith((".png", ".jpg", ".jpeg", ".webp")) and len(query) < 500):
                query_vec = self.embedder.embed_image(query)
            elif query.endswith((".wav", ".mp3", ".ogg")) and len(query) < 500:
                query_vec = self.embedder.embed_audio(query)
            else:
                raw_text_query = query
                expanded = self.expand_query(query)
                query_vec = self.embedder.embed_text(expanded)
        else:
            raw_text_query = str(query)
            expanded = self.expand_query(str(query))
            query_vec = self.embedder.embed_text(expanded)

        expanded_query_for_search = self.expand_query(raw_text_query) if raw_text_query else None

        return store.search(
            query_vector=query_vec,
            query_text=expanded_query_for_search,
            top_k=top_k,
            min_score=min_score,
            modality=modality
        )
