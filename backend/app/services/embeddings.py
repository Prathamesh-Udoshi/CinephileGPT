from sentence_transformers import SentenceTransformer
from app.core.config import settings

class EmbeddingService:
    _model = None

    @classmethod
    def get_model(cls) -> SentenceTransformer:
        """
        Lazy-load the SentenceTransformer model to save startup memory/time.
        """
        if cls._model is None:
            # Downloads the model on first call and caches it locally
            cls._model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        return cls._model

    @classmethod
    def get_embedding(cls, text: str) -> list[float]:
        """
        Generate a 384-dimensional vector embedding for the input text.
        """
        model = cls.get_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
