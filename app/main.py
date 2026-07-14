import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.exc import DBAPIError, InterfaceError, OperationalError

from app.api.health import router as health_router
from app.api.tickets import router as tickets_router
from app.ml.predictor import get_predictor

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_predictor().load()
        logger.info("ML model loaded")
    except FileNotFoundError as exc:
        # сервис стартует и без модели: /health и чтение работают, POST /tickets вернёт 503
        logger.warning("ML model is not available: %s", exc)
    yield


app = FastAPI(
    title="QoldauFlow",
    description="API маршрутизации обращений в поддержку",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(tickets_router)
app.include_router(health_router)


@app.exception_handler(OperationalError)
@app.exception_handler(InterfaceError)
@app.exception_handler(DBAPIError)
@app.exception_handler(ConnectionError)  # asyncpg бросает голый ConnectionRefusedError при недоступной БД
async def database_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Database is unavailable: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "Database is unavailable"},
    )
