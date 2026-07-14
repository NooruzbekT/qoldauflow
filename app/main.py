from contextlib import asynccontextmanager

from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    # здесь при старте загружается ML-модель (этап ML)
    yield


app = FastAPI(
    title="QoldauFlow",
    description="API маршрутизации обращений в поддержку",
    version="0.1.0",
    lifespan=lifespan,
)
