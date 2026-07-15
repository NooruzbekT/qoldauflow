from functools import lru_cache

# LaBSE выбрана по кросс-языковому бенчмарку ru<->kk: recall@5=1.00 против 0.54 у MiniLM
# (docs/benchmarks/bench_embeddings.py)
MODEL_NAME = "sentence-transformers/LaBSE"
EMBEDDING_DIM = 768


class TicketEmbedder:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self._model_name = model_name
        self._model = None

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self._model_name)
        model_dim = model.get_embedding_dimension()
        if model_dim != EMBEDDING_DIM:
            # размерность колонки в БД зафиксирована миграцией — смена модели требует новую миграцию
            raise RuntimeError(
                f"Embedding model {self._model_name} has dim {model_dim}, "
                f"but DB column expects {EMBEDDING_DIM}"
            )
        self._model = model

    def embed(self, text: str) -> list[float]:
        if self._model is None:
            raise RuntimeError("Embedding model is not loaded")
        return self._model.encode(text, normalize_embeddings=True).tolist()


@lru_cache
def get_embedder() -> TicketEmbedder:
    return TicketEmbedder()
