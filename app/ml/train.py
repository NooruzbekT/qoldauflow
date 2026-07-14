import json

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from app.core.config import BASE_DIR, get_settings

DATA_PATH = BASE_DIR / "data" / "tickets.csv"
REPORTS_DIR = BASE_DIR / "reports"
MODEL_FILENAME = "model.joblib"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5


def char_tfidf() -> TfidfVectorizer:
    # char_wb n-граммы устойчивы к смешанному ru/kk тексту и опечаткам
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)


def word_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(analyzer="word", ngram_range=(1, 2))


def candidates() -> list[tuple[str, Pipeline, dict]]:
    return [
        (
            "char_logreg",
            Pipeline(
                [
                    ("tfidf", char_tfidf()),
                    ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
                ]
            ),
            {"tfidf__ngram_range": [(2, 4), (3, 5)], "clf__C": [1, 3, 10]},
        ),
        (
            "char_word_logreg",
            Pipeline(
                [
                    ("features", FeatureUnion([("char", char_tfidf()), ("word", word_tfidf())])),
                    ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
                ]
            ),
            {"clf__C": [1, 3, 10]},
        ),
        (
            "char_svc_calibrated",
            Pipeline(
                [
                    ("tfidf", char_tfidf()),
                    ("clf", CalibratedClassifierCV(LinearSVC(random_state=RANDOM_STATE), cv=3)),
                ]
            ),
            {"clf__estimator__C": [0.5, 1, 3]},
        ),
        (
            "char_complement_nb",
            Pipeline([("tfidf", char_tfidf()), ("clf", ComplementNB())]),
            {"clf__alpha": [0.3, 1.0]},
        ),
    ]


def train() -> None:
    df = pd.read_csv(DATA_PATH)

    x_train, x_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=TEST_SIZE,
        stratify=df["label"],
        random_state=RANDOM_STATE,
    )

    comparison = {}
    best_name, best_search = None, None
    for name, pipeline, param_grid in candidates():
        search = GridSearchCV(
            pipeline, param_grid, cv=CV_FOLDS, scoring="f1_macro", n_jobs=-1
        )
        search.fit(x_train, y_train)
        comparison[name] = {
            "cv_macro_f1": round(search.best_score_, 4),
            "best_params": {k: str(v) for k, v in search.best_params_.items()},
        }
        print(f"{name}: cv macro F1 = {search.best_score_:.4f} {search.best_params_}")
        if best_search is None or search.best_score_ > best_search.best_score_:
            best_name, best_search = name, search

    model = best_search.best_estimator_
    y_pred = model.predict(x_test)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(f"\nbest model: {best_name}")
    print(classification_report(y_test, y_pred))
    print(f"test macro F1: {macro_f1:.4f}")

    REPORTS_DIR.mkdir(exist_ok=True)
    metrics = {
        "model": best_name,
        "best_params": comparison[best_name]["best_params"],
        "cv_macro_f1": comparison[best_name]["cv_macro_f1"],
        "macro_f1": round(macro_f1, 4),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "model_comparison": comparison,
        "train_size": len(x_train),
        "test_size": len(x_test),
        "test_split": TEST_SIZE,
        "random_state": RANDOM_STATE,
    }
    metrics_path = REPORTS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"metrics saved to {metrics_path}")

    settings = get_settings()
    settings.artifacts_dir.mkdir(exist_ok=True)
    model_path = settings.artifacts_dir / MODEL_FILENAME
    joblib.dump(model, model_path)
    print(f"model saved to {model_path}")


if __name__ == "__main__":
    train()
