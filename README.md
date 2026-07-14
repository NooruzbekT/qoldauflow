# QoldauFlow

API маршрутизации обращений в службу поддержки. Принимает обращения на русском и казахском, классифицирует их по четырём категориям (`payment`, `delivery`, `account`, `technical`), возвращает уверенность модели и отправляет сомнительные случаи на ручную проверку оператору. Ответы клиентам сервис не генерирует.

## Стек

Python 3.13 · FastAPI · Pydantic · PostgreSQL (SQLAlchemy Async + Alembic) · scikit-learn · sentence-transformers + pgvector (бонус) · Docker Compose · pytest · GitHub Actions

## Быстрый старт с чистого клона

Требуются Docker и Python 3.13+.

```bash
git clone https://github.com/NooruzbekT/qoldauflow.git
cd qoldauflow
cp .env.example .env

# зависимости и обучение модели (артефакт попадёт в artifacts/)
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m app.ml.train

# запуск API и PostgreSQL
docker compose up --build
```

После запуска:

- Swagger: http://localhost:8000/docs
- Здоровье сервиса: http://localhost:8000/health

Миграции Alembic применяются автоматически при старте контейнера API.

Остановка: `docker compose down` (данные БД сохраняются в volume; полная очистка — `docker compose down -v`).

> Если порт 5432 на машине занят локальным PostgreSQL — поменяй `POSTGRES_PORT` в `.env` (например, на 5433), compose пробросит базу на этот порт наружу; внутри docker-сети API всегда ходит на `db:5432`.

## Примеры запросов

Создать обращение:

```bash
curl -X POST http://localhost:8000/tickets \
  -H "Content-Type: application/json" \
  -d '{"text": "Төлем екі рет шешілді, ақшам қайтпады.", "language": "kk"}'
```

Ответ:

```json
{
  "id": 1,
  "text": "Төлем екі рет шешілді, ақшам қайтпады.",
  "language": "kk",
  "predicted_label": "payment",
  "confidence": 0.9785,
  "top_predictions": [
    {"label": "payment", "confidence": 0.9785},
    {"label": "delivery", "confidence": 0.0071}
  ],
  "needs_review": false,
  "status": "open",
  "created_at": "2026-07-15T10:00:00Z"
}
```

Остальные эндпоинты:

```bash
curl http://localhost:8000/tickets/1                          # обращение + фидбеки оператора
curl "http://localhost:8000/tickets?predicted_label=payment&needs_review=true&limit=20&offset=0"
curl -X POST http://localhost:8000/tickets/1/feedback \
  -H "Content-Type: application/json" \
  -d '{"correct_label": "payment", "comment": "Клиент сообщил о повторном списании"}'
curl http://localhost:8000/tickets/1/similar                  # бонус: 5 похожих обращений
curl http://localhost:8000/health
```

Ошибки: 422 — пустой текст, неизвестный язык или категория; 404 — несуществующее обращение; 503 — модель или база недоступны.

## ML-пайплайн

### Данные

`data/tickets.csv` — 480 синтетических обезличенных обращений: по 120 на категорию, в каждой 74 русских и 46 казахских. Тексты разнообразные: формальные и разговорные, с опечатками, суммами и номерами заказов.

### Обучение

```bash
python -m app.ml.train
```

Стратифицированный split 80/20 (`random_state=42`). GridSearchCV (5-fold, macro F1) сравнивает четыре кандидата:

| Кандидат | CV macro F1 |
| --- | ---: |
| **char TF-IDF + LogisticRegression** | **0.9187** |
| char TF-IDF + LinearSVC (калиброванный) | 0.9135 |
| char+word TF-IDF + LogisticRegression | 0.8901 |
| char TF-IDF + ComplementNB | 0.8766 |

Победитель — `TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), sublinear_tf=True)` + `LogisticRegression(C=10)`. Символьные n-граммы устойчивы к смешанному русско-казахскому тексту и опечаткам.

**Test macro F1 = 0.9061** (цель ТЗ ≥ 0.65). Полный `classification_report` и сравнение моделей — в `reports/metrics.json`, он воспроизводится командой обучения. Артефакт пайплайна сохраняется в `artifacts/model.joblib` (в git не попадает).

### Порог needs_review

Предсказанная категория выбирается как argmax вероятностей; порог решает только, показывать ли обращение оператору. Выбран порог **0.5** по распределению confidence на test-наборе:

- средняя уверенность верных предсказаний — 0.73, ошибочных — 0.49;
- 6 из 9 ошибок на test имеют confidence < 0.5 и уходят на ручную проверку;
- нагрузка на оператора — около 20% потока (19 из 96).

Порог настраивается через `CONFIDENCE_THRESHOLD` в `.env`.

## База данных

Две сущности:

- `tickets` — текст, язык, прогноз модели (категория, confidence, top-2 в JSONB), `needs_review`, `status` (`open`/`reviewed`), embedding (бонус), дата создания. Прогноз записывается один раз и никогда не изменяется.
- `ticket_feedback` — правки оператора отдельной таблицей (FK на тикет): каждая проверка — новая строка, история не переписывается.

Соединение асинхронное (asyncpg), миграции — Alembic (`alembic upgrade head`).

## Бонус: похожие обращения

`GET /tickets/{id}/similar` возвращает 5 ближайших обращений.

- **Модель эмбеддингов**: `sentence-transformers/LaBSE` (768 измерений), локальная, работает на CPU. Вектор считается при создании обращения.
- **Метрика похожести**: косинусная, `similarity = 1 − cosine_distance`, поиск оператором `<=>` pgvector.
- **Индекс**: HNSW по `vector_cosine_ops` (создаётся миграцией). На текущем объёме данных поиск фактически точный; индекс — задел на рост.
- **Выбор модели**: собственный кросс-языковой бенчмарк на 28 парах ru↔kk из нашего датасета (`docs/benchmarks/bench_embeddings.py`): LaBSE — recall@5 = 1.00, MRR = 0.973; MiniLM — 0.54/0.406; multilingual-e5-base — 0.09/0.063. LaBSE обучена на выравнивании переводов в 109 языках включая казахский, поэтому русское обращение и его казахский аналог попадают рядом.
- **Ограничения**: датасет маленький и синтетический — оценка похожести на реальных обращениях может быть слабее; LaBSE добавляет ~1.8 ГБ к Docker-образу и заметно тяжелее MiniLM на CPU — осознанный размен на качество казахского.

## Тесты

```bash
pytest
```

14 интеграционных тестов против реального PostgreSQL: валидации (пустой текст, язык, категория), структура прогноза, правило `needs_review`, получение по ID, 404, сохранение feedback и неизменность исходного прогноза, фильтры и пагинация, similar, `/health`. Тесты сами создают отдельную базу `qoldauflow_test`.

В GitHub Actions тесты запускаются на каждый push и pull request: поднимается postgres-service, обучается модель, затем pytest.

## Структура

```
qoldauflow/
├── app/
│   ├── api/            # роутеры: tickets, health
│   ├── core/           # конфигурация (pydantic-settings)
│   ├── db/             # async engine и сессии
│   ├── ml/             # train.py, predictor.py, embedder.py
│   ├── models/         # SQLAlchemy: Ticket, TicketFeedback
│   ├── schemas/        # Pydantic-схемы
│   └── main.py
├── alembic/            # миграции
├── data/tickets.csv    # датасет
├── docs/
│   ├── ai-log.md
│   └── benchmarks/     # бенчмарк embedding-моделей
├── reports/metrics.json
├── tests/
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

Отступление от структуры в ТЗ: вместо `pyproject.toml` используется единый `requirements.txt` — осознанное упрощение поставки зависимостей.
