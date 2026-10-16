"""Document management routes."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from backend.app.api.dependencies import rate_limit, require_api_key
from backend.app.schemas import IngestTextRequest
from backend.app.services.document_service import DocumentService

router = APIRouter(tags=["documents"])

_document_service: DocumentService | None = None


def get_document_service() -> DocumentService:
    global _document_service
    if _document_service is None:
        _document_service = DocumentService()
    return _document_service


@router.get("/documents", dependencies=[Depends(require_api_key), Depends(rate_limit)])
def list_documents() -> dict:
    service = get_document_service()
    return {"ok": True, "documents": service.list_documents(), "stats": service.stats()}


@router.delete("/documents/{doc_id}", dependencies=[Depends(require_api_key), Depends(rate_limit)])
def delete_document(doc_id: str) -> dict:
    if not doc_id or doc_id.strip() in {".", "..", "/"}:
        raise HTTPException(status_code=400, detail="Invalid doc_id")
    service = get_document_service()
    res = service.delete_document(doc_id)
    if res["deleted_chroma"] == 0 and res["deleted_bm25"] == 0:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found in index")
    return {"ok": True, **res, "stats": service.stats()}


@router.post("/ingest/text", dependencies=[Depends(require_api_key), Depends(rate_limit)])
def ingest_text(req: IngestTextRequest) -> dict:
    service = get_document_service()
    res = service.ingest_text(req.text, req.source)
    return {"ok": True, **res, "stats": service.stats()}


@router.post("/ingest/upload", dependencies=[Depends(require_api_key), Depends(rate_limit)])
async def ingest_upload(files: list[UploadFile] = File(...)) -> dict:
    save_dir = Path("./data/uploads")
    save_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for f in files:
        if not f.filename:
            continue
        dest = save_dir / f.filename
        content = await f.read()
        dest.write_bytes(content)
        paths.append(str(dest))
    if not paths:
        raise HTTPException(status_code=400, detail="No files uploaded")
    service = get_document_service()
    res = service.ingest_paths(paths)
    return {"ok": True, "files": paths, **res, "stats": service.stats()}


@router.post("/clear", dependencies=[Depends(require_api_key), Depends(rate_limit)])
def clear() -> dict:
    service = get_document_service()
    service.clear()
    return {"ok": True, "stats": service.stats()}
