"""
mcp_server.py — a CUSTOM MCP server for a Databricks App (this standalone example).

WHAT THIS DOES
--------------
It exposes the app's two in-house LangGraph agents as MCP *tools* over
streamable-HTTP, mounted on a FastAPI app. An external LangChain/LangGraph
agent (see ../client/langchain_mcp_client.py) connects to the resulting
`https://<app-url>/mcp` endpoint, authenticates with Databricks OAuth, and
calls these tools like any other tool.

The two tools (RAG leads — it is the simplest worked path):
    - ask_credit_question(query, client_id)        -> cited answer      [simple]
    - classify_document(file_ref)                  -> category, conf.   [simple]

HOW THIS FOLDS INTO YOUR EXISTING app/serve.py
------------------------------------------------
Say your Databricks App already ships a FastAPI backend at app/serve.py (served by uvicorn
on $DATABRICKS_APP_PORT). To add MCP, you do NOT create a new app resource or
add MCP-specific databricks.yml keys — the mount is purely in-code. You:

    1. Build the FastMCP server and its streamable-HTTP sub-app (below).
    2. Run the FastMCP session manager inside the FastAPI lifespan  <-- CRITICAL.
    3. `app.mount("/mcp", mcp_app)` alongside your existing routes.

In your real serve.py it looks like this (illustrative):

    from contextlib import asynccontextmanager
    from fastapi import FastAPI
    from server.mcp_server import mcp, mcp_app, ForwardedAccessTokenMiddleware
    # ... your existing imports / routers ...

    @asynccontextmanager
    async def lifespan(app):                          # keep your startup logic too
        async with mcp.session_manager.run():         # <-- CRITICAL (see note below)
            yield

    app = FastAPI(lifespan=lifespan)
    app.mount("/mcp", ForwardedAccessTokenMiddleware(mcp_app))  # -> https://<app-url>/mcp
    # app.include_router(your_existing_router)         # your API routes, etc.

This module is written so you can either run it standalone for the demo
(`uvicorn server.mcp_server:app ...`) OR import `mcp` / `mcp_app` /
`ForwardedAccessTokenMiddleware` into your serve.py.

AUTH NOTES
----------
- Server -> workspace (inside a tool): `WorkspaceClient()` with NO args uses the
  app's injected service-principal creds (DATABRICKS_CLIENT_ID / _SECRET).
- To act AS THE SIGNED-IN USER for governed data (SQL / UC / Vector Search),
  read the OBO user token from the `x-forwarded-access-token` request header
  (present when user-authorization is enabled on the app). FastMCP tools do not
  receive the raw HTTP request, so a small ASGI middleware
  (ForwardedAccessTokenMiddleware, below) captures that header into a contextvar
  that `_obo_user_token()` reads. We use it in ask_credit_question.
"""

from __future__ import annotations

import contextvars
import os
from typing import Any

from mcp.server.fastmcp import FastMCP  # requires mcp>=1.9,<2 (v2 renamed FastMCP->MCPServer)
from starlette.types import ASGIApp, Receive, Scope, Send

from server import agent_shims

# ---------------------------------------------------------------------------
# 1. Build the FastMCP server.
#    - stateless_http=True: each request is self-contained (no server-side
#      session affinity) — the right default for a horizontally-scaled App.
#    - streamable_http_path="/": the sub-app serves at its own root; because we
#      mount it under "/mcp" below, the final endpoint is /mcp (NOT /mcp/mcp).
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "example-credit-app-mcp",
    stateless_http=True,
    streamable_http_path="/",
)


# ---------------------------------------------------------------------------
# OBO (on-behalf-of-user) token plumbing.
#
# When user-authorization is enabled on the Databricks App, the signed-in
# user's token arrives in the `x-forwarded-access-token` HTTP header. A tool
# can use it to construct a WorkspaceClient that acts AS THAT USER for governed
# data access; if it is absent (e.g. pure M2M), the tool falls back to the
# app's service principal via a no-arg WorkspaceClient().
#
# WHY A MIDDLEWARE: FastMCP does NOT hand the tool the live Starlette request.
# (`mcp.get_context().request_context.request` is the parsed MCP protocol
# request, not the HTTP request, so it has no HTTP headers.) The reliable,
# transport-stable way to read an incoming HTTP header is to capture it in an
# ASGI middleware — where we DO have the raw request scope — and stash it in a
# contextvar that the tool reads. Because FastMCP runs sync tools inline in the
# same request task, the contextvar set here is visible inside the tool.
# ---------------------------------------------------------------------------
OBO_HEADER = b"x-forwarded-access-token"

# Per-request storage for the forwarded user token.
_forwarded_user_token: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "forwarded_user_token", default=None
)


class ForwardedAccessTokenMiddleware:
    """ASGI middleware that stashes the OBO user token into a contextvar.

    Wrap the mounted MCP sub-app with this so tools can read the forwarded
    `x-forwarded-access-token` header via `_obo_user_token()`.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        token: str | None = None
        for key, value in scope.get("headers", []):
            if key.lower() == OBO_HEADER:
                token = value.decode("latin-1")
                break
        reset_token = _forwarded_user_token.set(token)
        try:
            await self.app(scope, receive, send)
        finally:
            _forwarded_user_token.reset(reset_token)


def _obo_user_token() -> str | None:
    """Return the forwarded user access token for the current request, or None.

    Populated by ForwardedAccessTokenMiddleware. None means the request carried
    no `x-forwarded-access-token` header (e.g. a pure M2M call). We deliberately
    do NOT swallow errors here — a genuine failure should surface, not silently
    disable OBO.
    """
    return _forwarded_user_token.get()


# ===========================================================================
# TOOL 1 (LEAD): ask_credit_question — the simplest, fully-worked path.
# ===========================================================================
@mcp.tool()
def ask_credit_question(query: str, client_id: str) -> dict[str, Any]:
    """Answer a credit question about a specific client, WITH citations.

    Args:
        query: The natural-language question (e.g. "What is the DTI ratio?").
        client_id: The client whose credit file to ground the answer in.

    Returns:
        {"answer": str, "citations": [...], "model": str}
    """
    # --- Act as the signed-in user for governed retrieval -------------------
    # In production you'd build a per-request WorkspaceClient from the OBO token:
    #
    #   user_token = _obo_user_token()
    #   if user_token:
    #       from databricks.sdk import WorkspaceClient
    #       ws = WorkspaceClient(host=os.environ["DATABRICKS_HOST"], token=user_token)
    #       # ... use `ws` for Vector Search / SQL under the user's own grants ...
    #   else:
    #       from databricks.sdk import WorkspaceClient
    #       ws = WorkspaceClient()  # app service principal
    #
    # The shim ignores auth so the demo runs with no workspace, but we still
    # read the OBO token so you can see the plumbing work end-to-end. We expose
    # only WHICH identity the call ran under — never the token itself.
    user_token = _obo_user_token()
    result = agent_shims.run_rag_agent(query=query, client_id=client_id)
    result["acting_as"] = "user (OBO)" if user_token else "app service principal"
    return result


# ===========================================================================
# TOOL 2: classify_document.
# ===========================================================================
@mcp.tool()
def classify_document(file_ref: str) -> dict[str, Any]:
    """Classify a credit document into a category with a confidence score.

    Args:
        file_ref: A path/identifier the pipeline can resolve (UC Volume path,
            signed URL, etc.).

    Returns:
        {"category": str, "confidence": float, "file_ref": str}
    """
    # SECURITY (production): `file_ref` is attacker-influenced input. Before the real
    # pipeline resolves/fetches it, VALIDATE it: enforce length/format limits; prefer
    # opaque UC Volume / object ids over arbitrary paths or URLs; if URLs are allowed,
    # allowlist schemes/hosts and block local/private-network targets (SSRF); and
    # never echo signed URLs back to the caller or into logs.
    return agent_shims.run_classification(file_ref=file_ref)


# ---------------------------------------------------------------------------
# 2. Build the streamable-HTTP ASGI sub-app and wire it onto FastAPI.
#
# CRITICAL: the FastMCP StreamableHTTP *session manager* must be running for
# the lifetime of the app, or requests to /mcp hang forever (the session
# manager never initializes). You do that by running it inside the FastAPI
# lifespan. `mcp.session_manager` is created lazily, so you must call
# `mcp.streamable_http_app()` BEFORE accessing it.
#
# (Some Databricks docs write this as `lifespan=mcp_app.lifespan`; that relied
# on an older SDK where the sub-app exposed a `.lifespan`. The session-manager
# form below is equivalent and version-safe for mcp>=1.9,<2.)
# ---------------------------------------------------------------------------
from contextlib import asynccontextmanager  # noqa: E402

mcp_app = mcp.streamable_http_app()

# Only import FastAPI when assembling the standalone app, so the module can be
# imported by tools/tests without FastAPI present. In your serve.py you
# already have a FastAPI() instance — you just add this lifespan + the mount.
from fastapi import FastAPI  # noqa: E402  (kept here for the standalone runner)


@asynccontextmanager
async def lifespan(_app: "FastAPI"):
    # Run the MCP session manager for the app's lifetime.
    # If your serve.py already has startup/shutdown logic, nest it here:
    #   async with mcp.session_manager.run():
    #       await your_startup(); yield; await your_shutdown()
    async with mcp.session_manager.run():
        yield


app = FastAPI(
    title="Example Credit App (+ MCP)",
    lifespan=lifespan,  # <-- the one line people forget
)

# ===========================================================================
# ⚠️  SECURITY BOUNDARY — READ BEFORE HOSTING THIS ANYWHERE BUT DATABRICKS APPS
# ---------------------------------------------------------------------------
# This shim does NOT authenticate the caller: it never validates the incoming
# bearer token, and ForwardedAccessTokenMiddleware TRUSTS the inbound
# `x-forwarded-access-token` header verbatim. That is safe ONLY because the
# Databricks Apps ingress terminates auth in front of this process and sets that
# header itself. If you expose this behind ANY other ingress (or bind the port
# directly), you MUST:
#   - validate the bearer's issuer / audience / signature / expiry, and the
#     caller's authorization to use the app, and
#   - NEVER trust a client-supplied `x-forwarded-access-token` (a caller could
#     otherwise forge the OBO user identity) — derive it only from a verified token.
# ===========================================================================

# Mount the MCP server. Final endpoint: https://<app-url>/mcp
# (streamable_http_path="/" above means NO extra /mcp prefix inside the sub-app.)
# ForwardedAccessTokenMiddleware captures the OBO `x-forwarded-access-token`
# header so tools can read it via _obo_user_token().
app.mount("/mcp", ForwardedAccessTokenMiddleware(mcp_app))


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Trivial health check (and a reminder your own routes live alongside /mcp)."""
    return {"status": "ok", "mcp_endpoint": "/mcp"}


# Local run: `uvicorn server.mcp_server:app --host 0.0.0.0 --port 8000`
# On Databricks Apps the platform runs the equivalent via app.yaml using
# $DATABRICKS_APP_PORT (see app.yaml).
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("DATABRICKS_APP_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
