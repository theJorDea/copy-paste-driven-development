from importlib.metadata import version

from fastapi import FastAPI
from pydantic import BaseModel

from _collections_abc import AsyncIterator
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    print("Resource initialized")

    yield
    print("Resource terminated")

app = FastAPI(lifespan=lifespan)
APP_VERSION = version("mlops-health-service")


class HealthStatusResponse(BaseModel):
    status: str

class VersionResponse(BaseModel):
    version: str

@app.get("/healthz")
async def health_check() -> HealthStatusResponse:
    return HealthStatusResponse(status='healthy')

@app.get("/api/v1/version")
async def get_version() -> VersionResponse:
    return VersionResponse(version=APP_VERSION)

@app.get('/api/v1/health')
async def health_check_v1() -> HealthStatusResponse:
    return HealthStatusResponse(status='healthy')