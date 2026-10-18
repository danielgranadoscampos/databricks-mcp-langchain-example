"""
smoke.py — prove the custom MCP server works with NO Databricks creds.

Two checks, both run this example's MCP server in-process (shim mode) and talk
to it over streamable-HTTP. No workspace, no model endpoints, no OAuth.

  1. tools/list returns exactly the 2 MCP tools.
  2. OBO header path: a request carrying `x-forwarded-access-token` is actually
     observed by the server tool (and a request without it is not). This proves
     ForwardedAccessTokenMiddleware -> contextvar -> _obo_user_token() wiring.

Run from the repo root:
    python scripts/smoke.py

Depends only on the SERVER requirements (mcp, fastapi, uvicorn). It does NOT
import langchain / databricks-langchain, so you can run it before setting up
the client environment.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import socket
import sys
import threading
import time
import uuid

# Make the repo root importable so `server` resolves regardless of CWD.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn

# Import the server module (not just `app`) so the OBO check can observe the
# token the tool sees via the module-level _obo_user_token().
import server.mcp_server as mcp_server
from server.mcp_server import app

EXPECTED_TOOLS = {
    "ask_credit_question",
    "classify_document",
}


def _free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _ServerThread(threading.Thread):
    """Run uvicorn in a background thread so we can query it from the main one."""

    def __init__(self, port: int) -> None:
        super().__init__(daemon=True)
        self._config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        self.server = uvicorn.Server(self._config)

    def run(self) -> None:
        self.server.run()


async def _list_tools(url: str) -> list[str]:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            return [t.name for t in result.tools]


async def _call_tool(url: str, name: str, arguments: dict, headers: dict | None = None):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(url, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await session.call_tool(name, arguments)


def _check_tools_list(url: str) -> bool:
    tools = asyncio.run(_list_tools(url))
    print("tools/list returned:", tools)
    got = set(tools)
    missing = EXPECTED_TOOLS - got
    unexpected = got - EXPECTED_TOOLS
    if missing:
        print("FAIL — missing expected tools:", sorted(missing))
        return False
    if unexpected:
        print("FAIL — unexpected extra tools:", sorted(unexpected))
        return False
    print(f"OK — exactly the {len(EXPECTED_TOOLS)} expected tools discovered over /mcp/.")
    return True


def _check_obo_header(url: str) -> bool:
    """Issue a tool call WITH and WITHOUT the OBO header; assert the server sees it.

    We temporarily wrap the server's _obo_user_token() to record, in-process,
    exactly what the tool observed. The raw token never crosses back over the
    wire (best practice: don't return/log user tokens) — the assertion happens
    here in Python, in the same process as the server.
    """
    observed: list[str | None] = []
    real_obo = mcp_server._obo_user_token

    def recording_obo() -> str | None:
        token = real_obo()
        observed.append(token)
        return token

    mcp_server._obo_user_token = recording_obo  # patched module global; tool resolves it at call time
    try:
        sentinel = f"obo-sentinel-{uuid.uuid4().hex}"
        result_with = asyncio.run(
            _call_tool(
                url,
                "ask_credit_question",
                {"query": "ping", "client_id": "CLT-1"},
                headers={"x-forwarded-access-token": sentinel},
            )
        )
        observed_with = observed[-1]

        asyncio.run(
            _call_tool(url, "ask_credit_question", {"query": "ping", "client_id": "CLT-1"})
        )
        observed_without = observed[-1]
    finally:
        mcp_server._obo_user_token = real_obo

    # The tool also reports which identity it ran under (never the token itself).
    acting_as = None
    structured = getattr(result_with, "structuredContent", None)
    if isinstance(structured, dict):
        acting_as = structured.get("acting_as")

    print(
        "OBO check: with-header observed =",
        "<matches sent sentinel>" if observed_with == sentinel else repr(observed_with),
        "| without-header observed =",
        repr(observed_without),
        "| tool acting_as =",
        repr(acting_as),
    )

    if observed_with != sentinel:
        print("FAIL — server did NOT observe the forwarded x-forwarded-access-token value.")
        return False
    if observed_without is not None:
        print("FAIL — server saw a token when none was forwarded.")
        return False
    if acting_as != "user (OBO)":
        print(f"FAIL — tool did not report acting as the OBO user (got {acting_as!r}).")
        return False
    print("OK — forwarded OBO header is observed by the server tool; absent header -> None.")
    return True


def main() -> int:
    port = _free_port()
    # Trailing slash: the server mounts the FastMCP sub-app at /mcp with
    # streamable_http_path="/", so the real endpoint is /mcp/. (Hitting /mcp here
    # would still work, but only because the same-origin 307 redirect to /mcp/ is
    # auto-followed. Deployment-observed: on the app we deployed, that redirect's
    # Location came back as the internal host and was NOT followed — the exact
    # Location may differ behind other ingresses, so target /mcp/ directly.)
    url = f"http://127.0.0.1:{port}/mcp/"

    server = _ServerThread(port)
    server.start()

    # Wait for the port to accept connections (lifespan/session-manager init).
    deadline = time.time() + 20
    while time.time() < deadline and not server.server.started:
        time.sleep(0.05)

    try:
        ok_tools = _check_tools_list(url)
        ok_obo = _check_obo_header(url)
    finally:
        server.server.should_exit = True
        server.join(timeout=10)

    return 0 if (ok_tools and ok_obo) else 1


if __name__ == "__main__":
    sys.exit(main())
