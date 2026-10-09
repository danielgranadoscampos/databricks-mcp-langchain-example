# Example Credit App × MCP × LangChain — a guided example

**Goal:** run a LangChain/LangGraph application **outside Databricks** that calls the
agents living **inside** an *Example Credit App* (a Databricks App)
over **MCP** (Model Context Protocol).

> **Disclaimer:** This is an unofficial, educational sample provided **as-is**, without
> warranty of any kind. It is **not** an official Databricks product and is **not**
> officially supported or endorsed by Databricks. Review, test, and harden it for your
> own environment before any production use. Use at your own risk.

This repo is a small, heavily-commented teaching example — not a framework. It has
two halves:

1. **`server/`** — a *custom* MCP server mounted on the app's existing FastAPI
   backend, exposing the two in-app LangGraph agents as MCP tools.
2. **`client/`** — an external LangChain agent that authenticates with Databricks
   OAuth, connects to the server's `/mcp/` endpoint, uses `ChatDatabricks` as its
   brain, and calls the tools.

---

## Architecture

```
   OUTSIDE Databricks                    │          INSIDE Databricks (Azure)
                                         │
  ┌───────────────────────────┐         │   ┌──────────────────────────────────────┐
  │  client/                  │         │   │  Example Credit App (Databricks App)   │
  │  LangChain ReAct agent    │         │   │  app name: example-credit-app          │
  │                           │         │   │  FastAPI (app/serve.py) + React        │
  │  brain = ChatDatabricks   │         │   │                                        │
  │   (llama-4-maverick)  ────┼─────────┼──▶│  foundation model endpoint             │
  │                           │  OAuth  │   │                                        │
  │  tools  = MCP tools  ─────┼─────────┼──▶│  /mcp  (custom MCP server) ──┐         │
  └───────────────────────────┘ Bearer  │   │                              │         │
         │                        token  │   │   ┌──────────────────────────▼──────┐  │
         │  streamable-HTTP              │   │   │ (a) RAG Q&A agent               │  │
         │  Authorization: Bearer <tok>  │   │   │ (b) Document Classification     │  │
         └───────────────────────────────┼──▶│   │                                 │  │
                                         │   │   └─────────────────────────────────┘  │
                                         │   └──────────────────────────────────────┘
```

The outer agent treats a Databricks-hosted agent as *just another tool*. See
**Two-tier agents** below for why that is powerful and what it costs.

---

## The two tools (RAG leads)

| MCP tool                        | Backing agent               | Inputs                          | Output |
|---------------------------------|-----------------------------|---------------------------------|--------|
| `ask_credit_question` ⭐         | RAG Q&A agent               | `query`, `client_id`            | cited answer |
| `classify_document`             | Document Classification     | `file_ref` (path/bytes)         | `category`, `confidence` |

⭐ `ask_credit_question` is the simplest fully-worked path — start here. **Two MCP
tools for two agents:** a RAG Q&A agent that returns a cited answer, and a simple
request→response document classifier.

Model endpoints used inside the app: `databricks-llama-4-maverick` (primary),
`databricks-claude-sonnet-5` (escalation), `databricks-qwen3-embedding` (embeddings).

---

## Custom MCP vs Managed MCP — the decision rule

| | **Custom MCP server** (this example) | **Managed MCP** (Databricks-hosted) |
|---|---|---|
| What it exposes | Bespoke, multi-step **app agents** you wrote (LangGraph graphs) | UC-native **primitives** |
| Where it runs | Inside your Databricks App (you write the server) | Databricks hosts it — **no server code** |
| Use when | Logic is a custom agent/workflow (the app's RAG or classifier) | You just want to expose a Vector Search index, a UC function, a Genie space, or SQL |

**Rule of thumb:** *custom app agents → custom MCP server on Apps (this repo).
UC-native primitives → managed MCP.* They compose — one client can connect to both.

**Be selective — don't wrap everything as MCP.** Expose as an MCP tool what
**multiple parts/teams reuse** (a shared agent, a common lookup); leave
**single-use, in-process logic as DIRECT function calls** — wrapping that in MCP only
adds a network hop, a schema, and an auth surface for no reuse benefit. Exposing both
agents here (the classifier included, whose internal steps you would normally keep as
direct calls) is a deliberate **teaching choice** to show the pattern end-to-end — not
a recommendation to MCP-ify every function.

### Where managed MCP complements the custom server

Managed MCP servers are commonly **schema-scoped**: the server URL ends at the
schema, and each object in that schema (index, function) is exposed as a *tool* of
that server. (Point a client at the schema URL and `list_tools()` returns them all.)

> **Version-specific — verify against current docs.** Databricks now distinguishes
> the newer **Unity Catalog Gateway** managed-MCP endpoints from the **legacy
> per-workspace** ones, and the scope (schema- vs object-level) and URL shape differ
> between them. Copy the scope/URL for your workspace from the current per-server
> reference table rather than assuming the shape below is universal.

- **Retrieval index via AI Search managed MCP** — if you expose the app's credit
  knowledge base as a Databricks AI Search / Vector Search index, that index is
  reachable through managed MCP (no server code), complementing the custom
  `ask_credit_question` agent. Server (schema-scoped):
  `https://<workspace-hostname>/api/2.0/mcp/ai-search/{catalog}/{schema}`
  (the current Azure docs name this service `ai-search`; `vector-search` is an older alias).
- **A credit calc or external lookup wrapped as a UC function** — register it as a
  Unity Catalog function and expose it via the UC Functions managed MCP. Server
  (schema-scoped): `https://<workspace-hostname>/api/2.0/mcp/functions/{catalog}/{schema}`
  — every function in that schema becomes a tool.
- Genie Agent: `https://<workspace-hostname>/api/2.0/mcp/genie/{genie_space_id}`
- Databricks SQL: `https://<workspace-hostname>/api/2.0/mcp/sql`

> The reference table in the Azure docs also shows a `.../{function_name}` and
> `.../{index_name}` form to address a single object; the schema-scoped server URL
> above is what the docs' working client examples use.

---

## The client's brain: `ChatDatabricks`

Orchestration runs outside Databricks, but the agent's LLM is a Databricks-hosted
model via `ChatDatabricks` — the model call goes to the workspace:

```python
from databricks_langchain import ChatDatabricks   # replaces deprecated langchain-databricks
llm = ChatDatabricks(model="databricks-llama-4-maverick")  # `model=` is the current form
```

`ChatDatabricks` accepts both `model=` and `endpoint=` (an accepted alias); prefer
`model=` for new code. Off-workspace auth uses env `DATABRICKS_HOST` +
`DATABRICKS_TOKEN` (the client reuses the OAuth token it mints).

---

## Two-tier agents (cost / latency)

A single user turn can trigger **two layers of LLM calls**:

```
user → [outer ReAct LLM: llama-4-maverick] decides to call a tool
     → MCP tool  ask_credit_question
        → [inner RAG agent LLM, inside the app] retrieves + synthesizes
     ← cited answer bubbles back up to the outer agent
```

Implications:
- **Cost**: you pay for *both* the outer agent's tokens and the inner agent's tokens.
- **Latency**: round-trips stack (outer → MCP → inner → back), so a tool call is not
  free — prefer coarse, high-value tools over chatty fine-grained ones.

---

## Run it locally

### 0. Prerequisites
- Python 3.10+
- Two separate virtualenvs are cleanest (server vs client), but one works too.

### 1. Smoke test — prove the MCP tools are discoverable (NO Databricks creds)

```bash
cd mcp-langchain-example
python -m venv .venv && source .venv/bin/activate
pip install -r server/requirements.txt
python scripts/smoke.py
```

Expected output (ends with):

```
tools/list returned: ['ask_credit_question', 'classify_document']
OK — exactly the 2 expected tools discovered over /mcp/.
OBO check: with-header observed = <matches sent sentinel> | without-header observed = None | tool acting_as = 'user (OBO)'
OK — forwarded OBO header is observed by the server tool; absent header -> None.
```

This starts the server in-process in **shim mode** (the agents are stand-ins — see
`server/agent_shims.py`), lists its tools over `/mcp/`, and then proves the OBO path:
it sends a request carrying `x-forwarded-access-token` and asserts the server tool
actually observes that value (and sees nothing when the header is absent). No
workspace needed.

### 2. Run the MCP server standalone

```bash
pip install -r server/requirements.txt
uvicorn server.mcp_server:app --host 0.0.0.0 --port 8000
# MCP endpoint: http://localhost:8000/mcp/   (trailing slash — see "Deploy pointers")
# Health (LOCAL only): http://localhost:8000/healthz  -> {"status":"ok",...}
#   NB: on a DEPLOYED Databricks App /healthz appeared reserved by the Apps ingress
#       (deployment-observed — see Deploy notes; verify on your own workspace).
```

### 3. Run the external LangChain client

```bash
python -m venv .venv-client && source .venv-client/bin/activate
pip install -r client/requirements.txt
cp client/.env.example client/.env     # then edit client/.env with real values
python client/langchain_mcp_client.py
```

With **no** `.env`, the client exits with a friendly message pointing at
`.env.example` (it does **not** dump a traceback) — try it first.

The client points at the deployed `APP_URL`. To run end-to-end fully locally you
would also point `APP_URL` at your local server and supply a token your local
server accepts; the shim server does not verify the bearer token, so a dummy works
for a local smoke of the full client path.

---

## Auth setup

### Create a service principal + OAuth secret (M2M)
The external client authenticates as **its own service principal** — a separate SP
you create for the client (or a user). This is **not** the app's injected service
principal (that one is the app's *server-side* identity; see "Server-side auth"
below). The external SP must be **granted access to the Databricks App**.

1. In the workspace **Settings → Identity and access → Service principals**, create
   (or reuse) a service principal for the external client.
2. Grant that SP permission on the app (**Databricks Apps permissions**, e.g.
   `CAN_USE`), so it is allowed to call the app's `/mcp/` endpoint.
3. Generate an **OAuth secret** for it → you get a **client ID** and **secret**.
4. Put them in `client/.env` as `DATABRICKS_CLIENT_ID` / `DATABRICKS_CLIENT_SECRET`.

### How the client gets a token (client-credentials)
```
POST https://<workspace-host>/oidc/v1/token
  Authorization: Basic base64(client_id:client_secret)
  body: grant_type=client_credentials&scope=all-apis
→ { "access_token": "...", "token_type": "Bearer", "expires_in": 3600 }
```
Send `Authorization: Bearer <access_token>` to the app's `/mcp/`.

> **`scope=all-apis` is a demo convenience, NOT least-privilege.** It mints a broad
> token so the walkthrough "just works". For production, bind the external SP's OAuth
> secret to only the scopes it actually needs (check the exact minimum in the
> workspace's **scoped-secret / OAuth** UI — don't assume a literal scope string
> here), and grant the SP **`CAN_USE`** on the app (never `CAN_MANAGE`).

### Token lifetime (~1 hour)
`expires_in` is 3600s. The client caches the token and **re-mints ~60s before
expiry** (`get_access_token` in `client/langchain_mcp_client.py`). Because
`MultiServerMCPClient` headers are static per instance, a long-lived process should
rebuild the client (or use an `httpx_client_factory`) when the token rotates.

### PAT quick-test — LOCAL ONLY (a deployed app's front door needs OAuth)
`USE_PAT=1` makes the client send `DATABRICKS_TOKEN` as the bearer **directly**,
skipping the OAuth client-credentials exchange. Whether a **legacy PAT** is accepted
depends on WHERE the MCP server runs:

- **Local / in-process** (the standalone `uvicorn server.mcp_server:app`, or
  `scripts/smoke.py`): there is **no Databricks front door** and the shim server does
  **not** verify the bearer — so a legacy PAT (indeed any string) works. This is the
  "quick-test" case.
- **Deployed Databricks App** (`https://<app>.databricksapps.com/mcp/`): the Apps
  **front door REJECTS a legacy PAT** — you get `401` on `/mcp/` (and other paths
  `302`-redirect to the OIDC `.../oidc/oauth2/v2.0/authorize` endpoint). It requires an
  **OAuth** token. Two supported ways:
  - **U2M**: `databricks auth token -p <profile>` → use its `access_token` as
    `DATABRICKS_TOKEN` (keep `USE_PAT=1`, since it's a pre-obtained bearer the client
    forwards verbatim).
  - **M2M**: leave `USE_PAT` unset and set `DATABRICKS_CLIENT_ID` /
    `DATABRICKS_CLIENT_SECRET`; the client runs the client-credentials flow itself
    (see "How the client gets a token" above).

**In short: PAT → local only; deployed app → OAuth (U2M or M2M).**

### Server-side auth (inside the tools)
- `WorkspaceClient()` with **no args** uses the app's injected service-principal
  creds (`DATABRICKS_CLIENT_ID` / `DATABRICKS_CLIENT_SECRET`).
- To act **as the signed-in user** for governed data (SQL warehouse / UC / Vector
  Search), read the OBO user token from the **`x-forwarded-access-token`** header
  (present when user-authorization is enabled) and pass it to a per-request
  `WorkspaceClient(host=..., token=...)`. See `_obo_user_token()` in
  `server/mcp_server.py`.
- **Enabling OBO:** declare the user API scopes the forwarded token should carry via
  `user_api_scopes` in `app.yaml` (see the commented example there), e.g.
  `iam.current-user:read`, `sql`, `vectorsearch.vector-search-endpoints` — grant only
  what the tools need, and verify the exact scope names in the app's **Authorization**
  UI. **M2M is NOT OBO:** the client-credentials flow above acts as the *caller SP*
  and never carries the signed-in user's identity; only the forwarded-token path does.

### Governance & audit (OBO in a regulated setting)
Because `ask_credit_question` can run **on behalf of the signed-in user** (OBO), the
governed read happens under **that user's own Unity Catalog grants** — not a broad
service identity — and each call is attributable to a real principal, so UC's
per-query lineage/audit gives you the **clear audit trail** a regulated setting needs
(who asked what, against which governed objects, when). M2M (the caller SP) by
contrast is one shared identity — fine for the demo, weaker for attribution.

> **Public Preview caveat:** the Databricks Apps custom-MCP pieces shown here are at
> **Public Preview** maturity at the time of writing. Gate production use in a
> regulated setting on its GA status and your own security review.

---

## Deploy pointers (fold into your app)

The MCP mount is **purely in-code** — no MCP-specific `databricks.yml` keys; it ships
inside the app resource your Databricks App already deploys (in this example, bundle
`example_credit_app`, app `example-credit-app`).

> ⚠️ **SECURITY BOUNDARY — this shim does NOT authenticate the caller.** It never
> validates the incoming bearer token and it trusts the `x-forwarded-access-token`
> header verbatim. That is safe **only** because the Databricks Apps ingress
> terminates auth in front of it and sets that header itself. If you host this behind
> **any other ingress** (or expose the port directly) you **MUST** validate the
> token's issuer / audience / signature / expiry and the caller's authorization, and
> you must **never trust a client-supplied `x-forwarded-access-token`** — otherwise a
> caller could forge the OBO user identity.

1. Copy the two `@mcp.tool()` functions and the `FastMCP` setup from
   `server/mcp_server.py` into your app's `serve.py`, swapping the shim calls for your
   real agent entrypoints (illustrated here as `run_rag_agent` /
   `classification_pipeline`).
2. Add the MCP lifespan to the existing FastAPI app and mount it (wrap the mount
   with `ForwardedAccessTokenMiddleware` from `server/mcp_server.py` so OBO works):
   ```python
   from contextlib import asynccontextmanager
   from fastapi import FastAPI
   from server.mcp_server import mcp, mcp_app, ForwardedAccessTokenMiddleware
   # mcp_app = mcp.streamable_http_app() already ran at import (creates mcp.session_manager, lazy)

   @asynccontextmanager
   async def lifespan(app):
       async with mcp.session_manager.run():   # CRITICAL — else /mcp hangs
           yield                                # (nest your existing startup here)

   app = FastAPI(lifespan=lifespan)
   app.mount("/mcp", ForwardedAccessTokenMiddleware(mcp_app))  # clients connect to https://<app-url>/mcp/ (trailing slash)
   ```
   > Some Databricks docs write this as `app = FastAPI(lifespan=mcp_app.lifespan)`.
   > With the current `mcp` SDK (>=1.9,<2) the sub-app has no `.lifespan` attribute;
   > the `session_manager.run()` form above is the equivalent, version-safe way and
   > achieves the same thing — starting the session manager.
3. Keep `stateless_http=True` and `streamable_http_path="/"`. Mounting the sub-app
   under `/mcp` with `streamable_http_path="/"` makes the real endpoint **`/mcp/`
   (with a trailing slash)** and avoids a `/mcp/mcp` double prefix. **Clients must
   connect to `/mcp/`.** A request to `/mcp` (no slash) gets a Starlette `Mount` `307`
   redirect to `/mcp/`. **Deployment-observed (not portable):** on the deployed app the redirect's `Location` came back as the app's *internal* host
   (`https://localhost:8000/mcp/`), which the streamable-HTTP MCP client refuses to
   follow, so the connect fails; the exact `Location` may differ behind other
   ingresses. Either way, targeting `/mcp/` directly skips the redirect — which is why
   `client/langchain_mcp_client.py` and `scripts/smoke.py` both do.
4. `databricks bundle deploy` and restart the app. The endpoint is
   `https://<app-name>-<id>.<region>.databricksapps.com/mcp/`.
5. **Health check — `/healthz` appeared reserved by the Apps ingress (deployment-observed).**
   On the deployed app, `GET /healthz` was answered by the platform's
   own liveness probe: it returned an empty `200` **before the request reached the app
   and without auth**, so the app's own `/healthz` route (the `{"status": "ok", ...}`
   handler in `server/mcp_server.py`) was **shadowed** and never visible externally.
   This is observed here, not a documented/portable guarantee — verify on your own
   workspace. (Locally, with no front door, `http://localhost:8000/healthz` returns
   the app's JSON normally.) If you need an app-level health JSON reachable on the deployed app,
   expose it on a **non-reserved path** such as `/readyz` or `/api/health`. We leave
   the route as-is here to avoid a server-code change / redeploy — the app's health is
   already provable with a live `/mcp/` `list_tools` call.

---

## Reference docs (Azure Databricks)

- Host your own (custom) MCP server on Databricks Apps:
  https://learn.microsoft.com/en-us/azure/databricks/agents/mcp-tools/custom-mcp
- Managed MCP servers (AI Search / UC functions / Genie / SQL):
  https://learn.microsoft.com/en-us/azure/databricks/agents/mcp-tools/managed-mcp
- Use MCP servers from agent code (LangGraph + `databricks-langchain`, with
  `create_react_agent`): https://learn.microsoft.com/en-us/azure/databricks/agents/mcp-tools/use-mcp-in-agents
- Databricks Apps authorization (app SP vs. user OBO `x-forwarded-access-token`):
  https://learn.microsoft.com/en-us/azure/databricks/dev-tools/databricks-apps/auth
- OAuth M2M (client-credentials for the external service principal):
  https://learn.microsoft.com/en-us/azure/databricks/dev-tools/auth/oauth-m2m

> All five URLs were verified to resolve on 2026-10-07. If one 404s later, search
> "Azure Databricks custom MCP server" / "managed MCP" / "use MCP servers in agents".

---

## File tree

```
mcp-langchain-example/
├── README.md                       # this guide
├── .gitignore
├── server/
│   ├── mcp_server.py               # FastMCP + FastAPI mount; the tools (RAG leads)
│   ├── agent_shims.py              # runnable stand-ins; TODO markers to swap in real agents
│   ├── requirements.txt
│   └── app.yaml                    # Databricks Apps entry point
├── client/
│   ├── langchain_mcp_client.py     # OAuth → MCP tools → ChatDatabricks → ReAct agent
│   ├── requirements.txt
│   └── .env.example                # placeholders only (no secrets)
└── scripts/
    └── smoke.py                    # tools/list over /mcp/, no creds
```

---

## License

Released under the [MIT License](LICENSE) — a permissive license. This remains an
unofficial, educational sample (see the disclaimer at the top), provided as-is.
