from fastapi import APIRouter, Request, Response
from fastapi.responses import FileResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from .schemas import CaptureResponse, CaptureImage, Command, CreateSession, Language, SessionResponse, TranslationResponse
from .translation_controller import FlowError

router = APIRouter(prefix="/api/v1")


@router.get("/health")
def health():
    return {"status": "ok", "storage": "memory", "workers": 1}


@router.post("/sessions", response_model=SessionResponse, status_code=201)
def create_session(body: CreateSession, request: Request):
    return request.app.state.controller.create(body.source_language)


@router.get("/sessions/{session_id}", response_model=SessionResponse)
def get_session(session_id: str, request: Request):
    return request.app.state.controller.inspect(session_id)


@router.post("/sessions/{session_id}/captures", response_model=CaptureResponse, openapi_extra={
    "requestBody": {"required": True, "content": {
        mime: {"schema": {"type": "string", "format": "binary"}}
        for mime in ("image/jpeg", "image/png")
    }}
})
async def capture(session_id: str, request: Request):
    controller = request.app.state.controller
    content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type not in ("image/jpeg", "image/png"):
        raise FlowError(415, "unsupported_image_type", "Send image/jpeg or image/png as the request body")
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > controller.settings.max_capture_bytes:
            raise FlowError(413, "capture_too_large", "Capture exceeds size limit")
    try:
        image = CaptureImage(content_type=content_type, data=bytes(data))
    except ValidationError:
        raise FlowError(422, "invalid_image", "An image is required") from None
    return await run_in_threadpool(controller.capture, session_id, image)


@router.post("/sessions/{session_id}/commands", response_model=SessionResponse)
def command(session_id: str, body: Command, request: Request):
    return request.app.state.controller.command(session_id, body.command)


@router.post("/sessions/{session_id}/translations", response_model=TranslationResponse)
def translate(session_id: str, request: Request):
    return request.app.state.controller.translate(session_id)


@router.delete("/sessions/{session_id}", status_code=204)
def delete(session_id: str, request: Request):
    request.app.state.controller.delete(session_id)
    return Response(status_code=204)


@router.get("/resources/{language}/{concept_id}")
def resource(language: Language, concept_id: str, request: Request):
    path = request.app.state.controller.resource_path(language, concept_id)
    return FileResponse(path, media_type="video/mp4")
