import logging
import socket
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.exc import DBAPIError
from sqlalchemy.exc import TimeoutError as PoolTimeoutError

from app.api.health import router as health_router
from app.api.tickets import router as tickets_router
from app.ml.embedder import get_embedder
from app.ml.predictor import get_predictor

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # сервис стартует и деградирует без моделей: /health и чтение работают,
    # POST /tickets и similar вернут 503
    try:
        get_predictor().load()
        logger.info("ML model loaded")
    except FileNotFoundError as exc:
        logger.warning("ML model is not available: %s", exc)
    except Exception:
        logger.exception("Failed to load ML model")
    try:
        get_embedder().load()
        logger.info("Embedding model loaded")
    except Exception:
        logger.exception("Failed to load embedding model")
    yield


app = FastAPI(
    title="QoldauFlow",
    description="API маршрутизации обращений в поддержку",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(tickets_router)
app.include_router(health_router)


# DBAPIError покрывает Operational/Interface/DataError; ConnectionError и gaierror —
# голые ошибки asyncpg при недоступном хосте; PoolTimeoutError — исчерпание пула
@app.exception_handler(DBAPIError)
@app.exception_handler(ConnectionError)
@app.exception_handler(socket.gaierror)
@app.exception_handler(PoolTimeoutError)
async def database_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Database is unavailable: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "Database is unavailable"},
    )
