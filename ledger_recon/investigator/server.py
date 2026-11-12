"""Local review console. Start with python -m ledger_recon.investigator.server."""
from __future__ import annotations
import argparse
import io
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from pypdf import PdfReader
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .engine import Engine

MAX_UPLOAD = 5 * 1024 * 1024


def create_app(root: Path, mode: str = 'rules', model: str = 'qwen2.5:7b') -> FastAPI:
    engine = Engine(root, mode=mode, model=model)
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='investigation')
    lock = threading.RLock()
    operations = {}
    case_cache = engine.list_cases()

    @asynccontextmanager
    async def lifespan(app):
        yield
        pool.shutdown(wait=True, cancel_futures=True)
        engine.shutdown()

    app = FastAPI(title='Ledger Recon · Payment Investigator', lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost', 'testserver'])

    @app.middleware('http')
    async def local_writes(request: Request, call_next):
        if request.method in ('POST', 'PUT', 'PATCH', 'DELETE'):
            origin = request.headers.get('origin')
            if origin and origin != f'{request.url.scheme}://{request.url.netloc}':
                return JSONResponse({'detail': 'This local console only accepts same-origin changes.'}, status_code=403)
            if request.headers.get('sec-fetch-site') == 'cross-site':
                return JSONResponse({'detail': 'Cross-site changes are not allowed.'}, status_code=403)
            try:
                length = int(request.headers.get('content-length', 0))
            except ValueError:
                return JSONResponse({'detail': 'Invalid content length.'}, status_code=400)
            if length > MAX_UPLOAD:
                return JSONResponse({'detail': 'Upload must be 5 MB or smaller.'}, status_code=413)
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'none'"
        return response

    @app.exception_handler(ValueError)
    def invalid(request, exc):
        return JSONResponse({'detail': str(exc)}, status_code=422)

    @app.exception_handler(KeyError)
    def missing(request, exc):
        return JSONResponse({'detail': 'Case or evidence was not found.'}, status_code=404)

    @app.get('/', response_class=HTMLResponse)
    def index():
        return (Path(__file__).parent / 'static/index.html').read_text()

    @app.get('/api/status')
    def status():
        with lock:
            return {'mode': mode, 'model': model if mode == 'ollama' else None,
                    'sandbox': True, 'operations': dict(operations)}

    @app.get('/api/cases')
    def cases():
        nonlocal case_cache
        with lock:
            if not any(v['state'] == 'running' for v in operations.values()):
                case_cache = engine.list_cases()
            return {'cases': case_cache}

    @app.get('/api/cases/{case_id}')
    def case(case_id: str):
        with lock:
            if operations.get(case_id, {}).get('state') == 'running':
                return {'case_id': case_id, 'status': 'investigating', 'operation': operations[case_id]}
        return engine.get_case(case_id)

    def assert_idle():
        if any(v['state'] == 'running' for v in operations.values()):
            raise HTTPException(409, 'An investigation is running. Wait for its result before changing the evidence.')

    @app.post('/api/import')
    async def ingest(request: Request):
        nonlocal case_cache
        chunks, size = [], 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_UPLOAD:
                raise HTTPException(413, 'Upload must be 5 MB or smaller.')
            chunks.append(chunk)
        raw = b''.join(chunks)
        import json
        try:
            bundle = json.loads(raw)
        except (ValueError, UnicodeError):
            raise HTTPException(422, 'Choose a valid JSON bundle.')
        with lock:
            assert_idle()
            counts = engine.import_bundle(bundle)
            case_cache = engine.list_cases()
            return counts

    @app.post('/api/documents')
    async def document(document_id: str = Form(...), file: UploadFile = File(...)):
        content = await file.read(MAX_UPLOAD + 1)
        if len(content) > MAX_UPLOAD:
            raise HTTPException(413, 'Document must be 5 MB or smaller.')
        filename = Path(file.filename or 'document.txt').name
        if filename.lower().endswith('.pdf'):
            try:
                reader = PdfReader(io.BytesIO(content))
                if reader.is_encrypted or len(reader.pages) > 50:
                    raise ValueError('Use an unencrypted PDF with at most 50 pages.')
                text = '\n\n'.join(page.extract_text() or '' for page in reader.pages)
                if not text.strip():
                    raise ValueError('This PDF has no readable text. Upload a text PDF or TXT file; OCR is not supported.')
            except Exception as exc:
                raise HTTPException(422, str(exc) if isinstance(exc, ValueError) else 'Could not read this PDF.')
        elif filename.lower().endswith('.txt'):
            try:
                text = content.decode('utf-8')
            except UnicodeError:
                raise HTTPException(422, 'TXT documents must use UTF-8.')
        else:
            raise HTTPException(422, 'Choose a .txt or text-based .pdf document.')
        with lock:
            assert_idle()
            return engine.import_bundle({'invoices': [], 'transactions': [],
                'documents': [{'id': document_id, 'filename': filename, 'text': text}]})

    @app.get('/api/search')
    def search(q: str, limit: int = 5):
        if not q.strip() or len(q) > 1000:
            raise HTTPException(422, 'Search query must be 1 to 1000 characters.')
        with lock:
            assert_idle()
            return engine.search(q, limit=max(1, min(limit, 10)))

    @app.get('/api/demo')
    def demo():
        from .demo import make_bundle
        return make_bundle()[0]

    @app.post('/api/cases/{case_id}/investigate', status_code=202)
    def investigate(case_id: str):
        nonlocal case_cache
        with lock:
            assert_idle()
            engine.get_case(case_id)
            operations[case_id] = {'state': 'running', 'message': 'Retrieving documents and checking the proposed allocation.'}
        def run():
            nonlocal case_cache
            try:
                result = engine.investigate(case_id)
                with lock:
                    operations[case_id] = {'state': 'complete', 'result_status': result.get('status')}
                    case_cache = engine.list_cases()
            except Exception as exc:
                with lock:
                    operations[case_id] = {'state': 'error', 'message': str(exc)[:500]}
        pool.submit(run)
        return {'case_id': case_id, 'status': 'investigating'}

    @app.post('/api/cases/{case_id}/approve')
    async def approve(case_id: str, request: Request):
        nonlocal case_cache
        body = await request.json()
        if not isinstance(body, dict) or not isinstance(body.get('proposal_id'), str):
            raise HTTPException(422, 'Name the exact proposal being approved.')
        with lock:
            assert_idle()
            result = engine.approve(case_id, body['proposal_id'])
            case_cache = engine.list_cases()
            return result

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path.home() / '.local/share/ledger-investigator')
    parser.add_argument('--mode', choices=['rules', 'ollama'], default='ollama')
    parser.add_argument('--model', default='qwen2.5:7b')
    parser.add_argument('--port', type=int, default=8789)
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(create_app(args.data, args.mode, args.model), host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
