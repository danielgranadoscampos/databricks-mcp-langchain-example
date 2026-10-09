"""
langchain_mcp_client.py — an EXTERNAL LangChain/LangGraph agent that calls a
Databricks App's in-app agents over MCP.

THE FLOW (all of this runs OUTSIDE Databricks)
-----------------------------------------------
    1. Mint a Databricks OAuth M2M token (client-credentials) for the EXTERNAL
       CALLER's OWN service principal — NOT the app's injected SP. M2M acts as that
       caller SP; it does NOT acquire the signed-in user's identity (M2M != OBO).
    2. Build a MultiServerMCPClient pointed at https://<app-url>/mcp/ over the
       streamable-HTTP transport, sending `Authorization: Bearer <token>`.
    3. `get_tools()` loads the two MCP tools as LangChain tools.
    4. Use `ChatDatabricks(model="databricks-llama-4-maverick")` as the agent's
       brain — i.e. a LangChain app OUTSIDE Databricks still uses Databricks
       foundation models.
    5. `create_react_agent(llm, tools)` builds a ReAct agent; we run a sample
       `ask_credit_question` query and print the cited answer + which tool fired.

TWO-TIER AGENT NUANCE
---------------------
The LLM in THIS process decides to call an MCP tool. That tool is itself a
Databricks-hosted LangGraph agent running its OWN LLM calls. So one user turn
can trigger two layers of model calls (outer ReAct + inner RAG agent) — you pay
for both and their round-trips stack. See the README 'Two-tier agents'.

AUTH / TOKEN LIFETIME
---------------------
The M2M token expires in ~1 hour (expires_in: 3600). For a long-lived client,
re-mint on expiry. We keep a tiny time-based cache below and document it; the
sample run is short enough that one token suffices.
"""

from __future__ import annotations

import os
import sys
import time

# ---------------------------------------------------------------------------
# STEP 0 — Validate configuration BEFORE importing any heavy / networked libs.
#
# This is deliberate: with no .env configured, the script must exit with a
# friendly message pointing at .env.example — NOT a raw traceback and NOT a
# slow import failure. So we check env vars first, using only the stdlib.
# ---------------------------------------------------------------------------

# python-dotenv is optional; load a local .env if present, but don't require it.
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

# Always needed. The remaining required vars depend on the auth mode:
#   - M2M (default):   DATABRICKS_CLIENT_ID + DATABRICKS_CLIENT_SECRET
#   - Direct bearer:   USE_PAT=1 + DATABRICKS_TOKEN  (local PAT or OAuth access_token; no M2M creds)
BASE_VARS = [
    "DATABRICKS_HOST",  # https://<workspace-hostname>
    "APP_URL",          # https://<app-name>-<id>.<region>.databricksapps.com
]
M2M_VARS = [
    "DATABRICKS_CLIENT_ID",     # the EXTERNAL caller's service principal app id (OAuth)
    "DATABRICKS_CLIENT_SECRET",  # that service principal's OAuth secret
]
PAT_VARS = ["DATABRICKS_TOKEN"]  # a direct bearer: local PAT or OAuth access_token


def _using_pat() -> bool:
    return os.environ.get("USE_PAT") == "1"


def _check_config_or_exit() -> dict[str, str]:
    """Return the required config, or print a friendly message and exit(1).

    Branches on USE_PAT so the documented direct-bearer path (USE_PAT=1 +
    DATABRICKS_TOKEN, a local PAT or OAuth access_token; no M2M creds) is reachable.
    """
    required = BASE_VARS + (PAT_VARS if _using_pat() else M2M_VARS)
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        here = os.path.dirname(os.path.abspath(__file__))
        example = os.path.join(here, ".env.example")
        mode = "direct bearer (USE_PAT=1)" if _using_pat() else "OAuth M2M"
        print(
            "\n".join(
                [
                    "",
                    f"  Example Credit App MCP client is not configured (auth mode: {mode}).",
                    "",
                    "  Missing environment variable(s): " + ", ".join(missing),
                    "",
                    f"  1. Copy the template:   cp {example} {os.path.join(here, '.env')}",
                    "  2. Fill in your Databricks host, APP_URL, and either the OAuth",
                    "     client id/secret (M2M) or USE_PAT=1 + DATABRICKS_TOKEN (direct bearer).",
                    "  3. Re-run:               python client/langchain_mcp_client.py",
                    "",
                ]
            )
        )
        sys.exit(1)
    return {v: os.environ[v] for v in required}


# ---------------------------------------------------------------------------
# STEP 1 — Mint (and cache) a Databricks OAuth M2M token.
# ---------------------------------------------------------------------------
_TOKEN_CACHE: dict[str, float | str] = {}


def get_access_token(
    host: str, client_id: str | None = None, client_secret: str | None = None
) -> str:
    """Return a bearer token for the app's /mcp/ endpoint.

    Direct bearer: if USE_PAT=1 and DATABRICKS_TOKEN is set, use that token
    directly (a local PAT or an OAuth access_token) — no M2M client credentials
    required. This branch is checked FIRST so it works without client_id/secret.

    Otherwise mint an OAuth M2M client-credentials token for the EXTERNAL
    caller's service principal:
      POST https://<host>/oidc/v1/token  (HTTP basic auth client_id:client_secret)
           grant_type=client_credentials&scope=all-apis
      -> {"access_token": "...", "token_type": "Bearer", "expires_in": 3600}
    The token lasts ~1h; we cache it and re-mint ~60s before expiry.
    """
    # Direct-bearer shortcut first, so it does not depend on M2M creds (see README).
    if _using_pat() and os.environ.get("DATABRICKS_TOKEN"):
        return os.environ["DATABRICKS_TOKEN"]

    now = time.time()
    if _TOKEN_CACHE.get("token") and float(_TOKEN_CACHE.get("expires_at", 0)) > now:
        return str(_TOKEN_CACHE["token"])

    if not (client_id and client_secret):
        raise RuntimeError(
            "OAuth M2M requires DATABRICKS_CLIENT_ID and DATABRICKS_CLIENT_SECRET "
            "(or set USE_PAT=1 and DATABRICKS_TOKEN for the direct-bearer path)."
        )

    import requests  # imported lazily so STEP 0 stays dependency-free

    resp = requests.post(
        f"{host.rstrip('/')}/oidc/v1/token",
        auth=(client_id, client_secret),  # HTTP basic auth
        # scope=all-apis is a DEMO convenience (broad token), NOT least-privilege.
        # Production: bind the SP's OAuth secret to only the scopes it needs (verify
        # the minimum in the workspace OAuth UI) and grant the SP CAN_USE on the app.
        data={"grant_type": "client_credentials", "scope": "all-apis"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    token = payload["access_token"]
    _TOKEN_CACHE["token"] = token
    _TOKEN_CACHE["expires_at"] = now + float(payload.get("expires_in", 3600)) - 60
    return token


async def main() -> None:
    cfg = _check_config_or_exit()

    # Heavy imports happen only AFTER config validation succeeds.
    from databricks_langchain import ChatDatabricks
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langgraph.prebuilt import create_react_agent

    host = cfg["DATABRICKS_HOST"]
    app_url = cfg["APP_URL"].rstrip("/")

    # STEP 1: token for the Authorization header. In direct-bearer mode cfg has no
    #         client creds, so use .get() — get_access_token() returns it directly.
    token = get_access_token(host, cfg.get("DATABRICKS_CLIENT_ID"), cfg.get("DATABRICKS_CLIENT_SECRET"))

    # STEP 2: MCP client over streamable-HTTP with the bearer token.
    #   NOTE: headers are static for the client instance. The token lasts ~1h;
    #   for a long-lived process, rebuild the client (or use an httpx client
    #   factory) when get_access_token() re-mints. See README 'Token lifetime'.
    client = MultiServerMCPClient(
        {
            "example_credit_app": {
                "transport": "streamable_http",
                # Trailing slash REQUIRED: the server mounts the FastMCP sub-app at
                # /mcp with streamable_http_path="/", so the real endpoint is /mcp/.
                # Deployment-observed: on the app we deployed, hitting /mcp (no slash)
                # returned a 307 whose Location came back as the internal host
                # https://localhost:8000/mcp/, which the streamable-HTTP client won't
                # follow; the exact Location may differ behind other ingresses.
                # Targeting /mcp/ directly avoids the redirect regardless. See README
                # "Deploy pointers".
                "url": f"{app_url}/mcp/",
                "headers": {"Authorization": f"Bearer {token}"},
            }
        }
    )

    # STEP 3: load the MCP tools as LangChain tools (get_tools is async).
    tools = await client.get_tools()
    print("Discovered MCP tools:", [t.name for t in tools])

    # STEP 4: the agent's brain — a Databricks foundation model, used from
    #         OUTSIDE Databricks. Off-workspace auth uses DATABRICKS_HOST +
    #         DATABRICKS_TOKEN; here we reuse the token we just obtained.
    #         Use explicit assignment (NOT setdefault): if DATABRICKS_TOKEN is
    #         already set to some stale/unrelated value, setdefault would leave
    #         it in place and ChatDatabricks would authenticate with the wrong
    #         token while MCP used ours. Assign so both use the same fresh token.
    os.environ["DATABRICKS_HOST"] = host
    os.environ["DATABRICKS_TOKEN"] = token
    llm = ChatDatabricks(model="databricks-llama-4-maverick")

    # STEP 5: build a ReAct agent and run a sample ask_credit_question query.
    #   (langgraph.prebuilt.create_react_agent is, in LangChain 1.x, a
    #    deprecated alias for langchain.agents.create_agent — both work; we use
    #    this import to match the Databricks docs.)
    agent = create_react_agent(llm, tools)

    # DEMO-CLIENT-001 is a synthetic demo value (no real client).
    question = (
        "Use the ask_credit_question tool to answer: what is the debt-to-income "
        "ratio for client DEMO-CLIENT-001, and cite your sources."
    )
    result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})

    # Print the final answer and show which tools were called.
    # NOTE (production): this prints raw tool args + model output for teaching
    # visibility. REDACT these before logging/displaying in production — they can
    # carry PII, credit details, or signed URLs.
    messages = result["messages"]
    print("\n--- Final answer ---")
    print(messages[-1].content)

    print("\n--- Tool calls made ---")
    for m in messages:
        for call in getattr(m, "tool_calls", None) or []:
            print(f"  * {call['name']}({call.get('args')})")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
