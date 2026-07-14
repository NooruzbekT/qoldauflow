import asyncpg
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.db.session import get_session
from app.main import app
from app.ml.predictor import get_predictor
from app.models.base import Base

TEST_DB = "qoldauflow_test"


async def ensure_test_database() -> None:
    settings = get_settings()
    conn = await asyncpg.connect(
        user=settings.postgres_user,
        password=settings.postgres_password,
        host=settings.postgres_host,
        port=settings.postgres_port,
        database=settings.postgres_db,
    )
    exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", TEST_DB)
    if not exists:
        await conn.execute(f'CREATE DATABASE "{TEST_DB}"')
    await conn.close()


@pytest.fixture
async def engine():
    await ensure_test_database()
    settings = get_settings()
    url = (
        f"postgresql+asyncpg://{settings.postgres_user}:{settings.postgres_password}"
        f"@{settings.postgres_host}:{settings.postgres_port}/{TEST_DB}"
    )
    engine = create_async_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def client(engine):
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async def override_get_session():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_session] = override_get_session

    predictor = get_predictor()
    if not predictor.is_ready:
        predictor.load()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as test_client:
        yield test_client

    app.dependency_overrides.clear()
