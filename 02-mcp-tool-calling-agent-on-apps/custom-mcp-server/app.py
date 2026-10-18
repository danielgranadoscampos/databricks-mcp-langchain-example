"""
app.py — a CUSTOM MCP server hosted on a Databricks App (example 02).

WHAT THIS DOES
--------------
It exposes a few genericized "operations" tools as MCP tools over
streamable-HTTP, mounted on a FastAPI app. The LangGraph tool-calling agent in
`../agent/` connects to the resulting `https://<app-url>/mcp/` endpoint,
authenticates with Databricks OAuth, and calls these tools alongside the
Databricks *managed* MCP servers (UC functions, Genie, Vector Search).

The tools return clearly-fictional sample data so the whole thing runs with no
workspace, no database, and no secrets — swap the bodies for real queries
(e.g. a SQL warehouse read via `WorkspaceClient`) when you adapt it.

The three tools (a tiny finance/operations domain):
    - list_accounts_payable()      -> fictional payables      [static sample]
    - list_accounts_receivable()   -> fictional receivables    [static sample]
    - get_pending_approvals()      -> per-user approvals; demonstrates OBO

DATABRICKS APPS PATTERN (and the gotchas this file honors)
----------------------------------------------------------
This follows the current custom-MCP-server-on-Apps pattern
(github.com/databrickslabs/mcp → examples/custom-server and the Databricks
"Custom MCP servers" docs). Key points:

  * FastMCP builds a streamable-HTTP ASGI sub-app. We mount it at ``/mcp`` with
    ``streamable_http_path="/"`` so the real endpoint is ``/mcp/`` (NOT
    ``/mcp/mcp``). **Clients must connect to ``/mcp/`` WITH the trailing slash.**
    A request to ``/mcp`` (no slash) gets a 307 redirect whose ``Location`` can
    come back as the app's *internal* host and may not be followed — target
    ``/mcp/`` directly (the agent and ``smoke.py`` both do).
  * The FastMCP StreamableHTTP *session manager* must run for the lifetime of
    the app, or requests to ``/mcp/`` hang. We do that in the FastAPI lifespan.
  * Databricks Apps RESERVES ``/healthz`` — the platform answers it with an
    empty 200 BEFORE the request reaches your process, SHADOWING any app route
    at that path. So this app exposes its health JSON on ``/health`` (a
    non-reserved path), never ``/healthz``.
  * The Apps front door expects an **OAuth** token (not a legacy PAT) at the
    edge. Service-principal (M2M) callers must have ``CAN_USE`` on this app.
  * For OBO (on-behalf-of-user) calls, enable user authorization on the app and
    declare ``user_api_scopes`` in ``app.yaml``; the signed-in user's token then
    arrives in the ``x-forwarded-access-token`` header (see ``_obo_user_token``).

Run locally:   uvicorn app:app --host 0.0.0.0 --port 8000
Offline smoke: python smoke.py   (starts this app in-process, lists tools)
"""

from __future__ import annotations

import contextvars
import os
from typing import Any

from mcp.server.fastmcp import FastMCP  # requires mcp>=1.9,<2 (v2 renamed FastMCP->MCPServer)
from starlette.types import ASGIApp, Receive, Scope, Send

# ---------------------------------------------------------------------------
# 1. Build the FastMCP server.
#    - stateless_http=True: each request is self-contained (no server-side
#      session affinity) — the right default for a horizontally-scaled App.
#    - streamable_http_path="/": the sub-app serves at its own root; because we
#      mount it under "/mcp" below, the final endpoint is /mcp/ (NOT /mcp/mcp).
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "example-operations-mcp",
    stateless_http=True,
    streamable_http_path="/",
)


# ---------------------------------------------------------------------------
# OBO (on-behalf-of-user) token plumbing.
#
# When user authorization is enabled on the Databricks App, the signed-in
# user's token arrives in the `x-forwarded-access-token` HTTP header. A tool can
# use it to construct a WorkspaceClient that acts AS THAT USER for governed data
# access; if it is absent (e.g. pure M2M), the tool falls back to the app's
# service principal via a no-arg WorkspaceClient().
#
# WHY A MIDDLEWARE: FastMCP does NOT hand the tool the live Starlette request,
# so the reliable, transport-stable way to read an incoming HTTP header is to
# capture it in an ASGI middleware (where we DO have the raw request scope) and
# stash it in a contextvar the tool reads. FastMCP runs sync tools inline in the
# same request task, so the contextvar set here is visible inside the tool.
# ---------------------------------------------------------------------------
OBO_HEADER = b"x-forwarded-access-token"

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
    no `x-forwarded-access-token` header (e.g. a pure M2M call).
    """
    return _forwarded_user_token.get()


# ===========================================================================
# TOOL 1: list_accounts_payable
# ===========================================================================
@mcp.tool()
def list_accounts_payable() -> list[dict[str, Any]]:
    """List the company's outstanding accounts payable (money owed to suppliers).

    Returns a list of items, each with: supplier, location, amount, due_date,
    description. (Fictional sample data — replace with a real query.)
    """
    # In production you would read this from a governed table, e.g.:
    #   from databricks.sdk import WorkspaceClient
    #   ws = _workspace_client_for_request()   # SP, or OBO user (see TOOL 3)
    #   rows = ws.statement_execution.execute_statement(... "SELECT ... FROM ap")
    return [
        {
            "supplier": "Northwind Cloud Services",
            "location": "Austin, TX, USA",
            "amount": "$9,500.00",
            "due_date": "2026-11-13",
            "description": "Cloud compute and storage",
        },
        {
            "supplier": "Riverbend Office Supplies",
            "location": "Sacramento, CA, USA",
            "amount": "$7,800.00",
            "due_date": "2026-12-01",
            "description": "Office equipment and supplies",
        },
        {
            "supplier": "Summit Facilities Group",
            "location": "Denver, CO, USA",
            "amount": "$10,000.00",
            "due_date": "2026-12-21",
            "description": "Building maintenance and utilities",
        },
    ]


# ===========================================================================
# TOOL 2: list_accounts_receivable
# ===========================================================================
@mcp.tool()
def list_accounts_receivable() -> list[dict[str, Any]]:
    """List the company's outstanding accounts receivable (money owed by customers).

    Returns a list of items, each with: customer, location, amount,
    expected_payment_date, description. (Fictional sample data.)
    """
    return [
        {
            "customer": "Lakeside Analytics",
            "location": "Chicago, IL, USA",
            "amount": "$5,000.00",
            "expected_payment_date": "2026-11-11",
            "description": "Two small clusters for ETL",
        },
        {
            "customer": "Meridian Commerce Ltd.",
            "location": "Manchester, UK",
            "amount": "$53,800.00",
            "expected_payment_date": "2026-11-13",
            "description": "GPU cluster for model deployment",
        },
        {
            "customer": "Prairie Home Goods",
            "location": "Topeka, KS, USA",
            "amount": "$12,000.00",
            "expected_payment_date": "2026-12-23",
            "description": "Operations dashboard build-out",
        },
    ]


# ===========================================================================
# TOOL 3: get_pending_approvals — demonstrates the OBO path.
# ===========================================================================
@mcp.tool()
def get_pending_approvals() -> dict[str, Any]:
    """List invoice approvals awaiting the signed-in user's sign-off.

    In a real deployment this is naturally per-user data, so the tool acts AS
    THE SIGNED-IN USER (OBO) when a forwarded user token is present, falling
    back to the app's service principal otherwise. We expose only WHICH identity
    the call ran under — never the token itself.

    Returns: {"acting_as": str, "pending_approvals": [...]}
    """
    # Act as the signed-in user for governed data when the OBO token is present:
    #
    #   user_token = _obo_user_token()
    #   from databricks.sdk import WorkspaceClient
    #   if user_token:
    #       ws = WorkspaceClient(host=os.environ["DATABRICKS_HOST"],
    #                            token=user_token, auth_type="pat")
    #   else:
    #       ws = WorkspaceClient()   # app service principal
    #   # ... use `ws` under the caller's own Unity Catalog grants ...
    #
    # The sample below ignores auth so the demo runs with no workspace, but we
    # still read the OBO token so you can see the plumbing work end to end.
    user_token = _obo_user_token()
    return {
        "acting_as": "user (OBO)" if user_token else "app service principal",
        "pending_approvals": [
            {
                "invoice_id": "INV-2042",
                "vendor": "Northwind Cloud Services",
                "amount": "$9,500.00",
                "submitted": "2026-10-30",
            },
            {
                "invoice_id": "INV-2048",
                "vendor": "Summit Facilities Group",
                "amount": "$10,000.00",
                "submitted": "2026-11-02",
            },
        ],
    }


# ---------------------------------------------------------------------------
# 2. Build the streamable-HTTP ASGI sub-app and wire it onto FastAPI.
#
# CRITICAL: the FastMCP StreamableHTTP session manager must be running for the
# lifetime of the app, or requests to /mcp/ hang forever (the session manager
# never initializes). You do that by running it inside the FastAPI lifespan.
# `mcp.session_manager` is created lazily, so call `mcp.streamable_http_app()`
# BEFORE accessing it.
# ---------------------------------------------------------------------------
from contextlib import asynccontextmanager  # noqa: E402

from fastapi import FastAPI  # noqa: E402

mcp_app = mcp.streamable_http_app()


@asynccontextmanager
async def lifespan(_app: "FastAPI"):
    # Run the MCP session manager for the app's lifetime. If you fold MCP into an
    # app that already has startup/shutdown logic, nest it here:
    #   async with mcp.session_manager.run():
    #       await your_startup(); yield; await your_shutdown()
    async with mcp.session_manager.run():
        yield


app = FastAPI(title="Example Operations MCP server", lifespan=lifespan)

# ===========================================================================
# SECURITY BOUNDARY — READ BEFORE HOSTING THIS ANYWHERE BUT DATABRICKS APPS
# ---------------------------------------------------------------------------
# This demo does NOT authenticate the caller: it never validates the incoming
# bearer token, and ForwardedAccessTokenMiddleware TRUSTS the inbound
# `x-forwarded-access-token` header verbatim. That is safe ONLY because the
# Databricks Apps ingress terminates auth in front of this process and sets that
# header itself. Behind ANY other ingress you MUST validate the bearer's
# issuer / audience / signature / expiry and the caller's authorization, and
# NEVER trust a client-supplied `x-forwarded-access-token` (a caller could
# otherwise forge the OBO user identity).
# ===========================================================================

# Mount the MCP server. Final endpoint: https://<app-url>/mcp/  (trailing slash).
app.mount("/mcp", ForwardedAccessTokenMiddleware(mcp_app))


@app.get("/health")
def health() -> dict[str, str]:
    """Health check on a NON-reserved path.

    Do NOT use `/healthz`: Databricks Apps reserves it and answers it with an
    empty 200 before the request reaches your app, shadowing any route there.
    """
    return {"status": "ok", "mcp_endpoint": "/mcp/"}


# Local run: `uvicorn app:app --host 0.0.0.0 --port 8000`.
# On Databricks Apps the platform runs the equivalent via app.yaml using
# $DATABRICKS_APP_PORT.
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("DATABRICKS_APP_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
