from importlib.metadata import version

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
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
