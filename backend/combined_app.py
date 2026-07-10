"""Combined ASGI app serving REST API + MCP SSE on one port"""
from backend.api.rag_api import app as fastapi_app
from backend.mcp.mcp_server import app as mcp_app

_db_initialized = False


async def app(scope, receive, send):
    global _db_initialized
    if not _db_initialized:
        _db_initialized = True
        from backend.rag.database import is_available, init_db
        if is_available():
            try:
                init_db()
            except Exception:
                pass
    path = scope.get("path", "")
    if path.startswith("/mcp"):
        await mcp_app(scope, receive, send)
    else:
        await fastapi_app(scope, receive, send)
