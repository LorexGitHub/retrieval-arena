"""Combined ASGI app serving REST API + MCP SSE on one port"""
from src.api.rag_api import app as fastapi_app
from src.mcp.mcp_server import app as mcp_app


async def app(scope, receive, send):
    path = scope.get("path", "")
    if path.startswith("/mcp"):
        await mcp_app(scope, receive, send)
    else:
        await fastapi_app(scope, receive, send)
