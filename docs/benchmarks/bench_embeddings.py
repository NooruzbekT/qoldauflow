"""Кросс-языковой бенчмарк embedding-моделей на своих данных.

Для каждой ru-заявки из пары ищем её kk-близнеца среди всех 480 тикетов
по косинусной близости. Метрики: recall@1, recall@5, MRR, средняя similarity пары.

Запуск из корня репозитория: python docs/benchmarks/bench_embeddings.py

Результаты (2026-07):
    MiniLM    recall@1=0.30  recall@5=0.54  MRR=0.406  pair_sim=0.628
    LaBSE     recall@1=0.95  recall@5=1.00  MRR=0.973  pair_sim=0.856  <- выбрана
    mE5-base  recall@1=0.00  recall@5=0.09  MRR=0.063  pair_sim=0.852
"""
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# (ru_id, kk_id) — семантические близнецы из data/tickets.csv
PAIRS = [
    (1, 29), (15, 34), (21, 40), (22, 41), (25, 42), (186, 232), (195, 239),
    (227, 257), (230, 258),
    (47, 75), (48, 77), (52, 80), (65, 81), (73, 91), (286, 318), (265, 311),
    (93, 121), (94, 122), (97, 124), (98, 127), (112, 131), (118, 138),
    (139, 167), (140, 168), (145, 172), (146, 173), (165, 183), (152, 177),
]

MODELS = {
    "MiniLM (текущая)": ("paraphrase-multilingual-MiniLM-L12-v2", ""),
    "LaBSE": ("sentence-transformers/LaBSE", ""),
    "mE5-base": ("intfloat/multilingual-e5-base", "query: "),
}


def evaluate(model_name: str, prefix: str, df: pd.DataFrame) -> dict:
    model = SentenceTransformer(model_name)
    texts = [prefix + t for t in df["text"].tolist()]
    emb = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    id_to_idx = {tid: i for i, tid in enumerate(df["id"].tolist())}

    ranks, pair_sims = [], []
    for ru_id, kk_id in PAIRS + [(k, r) for r, k in PAIRS]:  # обе стороны
        a, b = id_to_idx[ru_id], id_to_idx[kk_id]
        sims = emb @ emb[a]
        sims[a] = -1.0
        rank = int((sims > sims[b]).sum()) + 1
        ranks.append(rank)
        pair_sims.append(float(emb[a] @ emb[b]))

    ranks = np.array(ranks)
    return {
        "recall@1": float((ranks == 1).mean()),
        "recall@5": float((ranks <= 5).mean()),
        "MRR": float((1 / ranks).mean()),
        "mean_pair_sim": float(np.mean(pair_sims)),
    }


def main() -> None:
    df = pd.read_csv("data/tickets.csv")
    print(f"пар: {len(PAIRS)} x 2 направления, корпус: {len(df)} тикетов\n")
    for label, (name, prefix) in MODELS.items():
        m = evaluate(name, prefix, df)
        print(
            f"{label:18} recall@1={m['recall@1']:.2f}  recall@5={m['recall@5']:.2f}  "
            f"MRR={m['MRR']:.3f}  pair_sim={m['mean_pair_sim']:.3f}"
        )


if __name__ == "__main__":
    main()
