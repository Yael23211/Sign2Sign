from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from src.resources.temporary_session_store import TemporarySessionStore
from .dependencies import load_components
from .routes import router
from .settings import Settings
from .translation_controller import FlowError, TranslationController


def create_app(component_factory=load_components, settings=None):
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app):
        components = await run_in_threadpool(component_factory, settings)
        store = TemporarySessionStore(settings.max_sessions)
        app.state.controller = TranslationController(store, components, settings)
        try:
            yield
        finally:
            store.clear()
            await run_in_threadpool(components.close)

    app = FastAPI(title="Sign2Sign SC-02", lifespan=lifespan)

    @app.exception_handler(FlowError)
    async def flow_error(request: Request, exc: FlowError):
        return JSONResponse(status_code=exc.status, content={"error": {
            "code": exc.code, "message": exc.message, "stage": exc.stage}})

    app.include_router(router)
    return app


app = create_app()
