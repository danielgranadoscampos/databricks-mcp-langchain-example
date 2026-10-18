# Example 02 — MCP tool-calling LangGraph agent, deployed on Databricks Apps

A LangGraph tool-calling agent, authored with the MLflow **`ResponsesAgent`**
interface, that uses Databricks **managed** MCP servers (UC functions, Genie,
Vector Search) **and** a **custom** MCP server — and is **deployed as a
Databricks App** (not Model Serving).

> **Disclaimer:** This is an unofficial, educational sample provided **as-is**,
> without warranty of any kind. It is **not** an official Databricks product and
> is **not** officially supported or endorsed by Databricks. Review, test, and
> harden it for your own environment before any production use.

This modernizes the "Winning with MCP" agent notebook: it keeps the
`ResponsesAgent` + LangGraph design but **replaces the `agents.deploy` Model
Serving path with a Databricks App**, removes all hardcoded hosts / catalogs /
Genie ids / app URLs / secrets, and genericizes the domain into a clean,
self-contained demo.

---

## Why a Databricks App (not Model Serving)?

Databricks recommends deploying agents on **Databricks Apps** when you want full
control of the code, server configuration, and git-based versioning. The agent
is served by a small FastAPI entrypoint (`agent/app.py`) that exposes the
`ResponsesAgent` over HTTP; `agents.deploy` is gone. An optional MLflow
logging + evaluation script (`agent/log_and_evaluate.py`) remains for tracing
and quality tracking, but the **deployment target is the App**.

---

## Architecture

```
                    ┌──────────────────────────────────────────────┐
   user ──POST──▶   │  Agent (Databricks App)   agent/app.py         │
   /invocations     │  FastAPI → ResponsesAgent (LangGraph)          │
                    │  brain = ChatDatabricks                        │
                    │                                                │
                    │  tools discovered from MCP servers: ───────────┼──┐
                    └──────────────────────────────────────────────┘   │
                                                                        │
        ┌───────────────────────────────────────────────────────────────┘
        ▼
   Databricks MANAGED MCP                        CUSTOM MCP (Databricks App)
   ├─ UC functions   /api/2.0/mcp/functions/{cat}/{schema}   custom-mcp-server/
   ├─ Vector Search  /api/2.0/mcp/vector-search/{cat}/{schema}  → /mcp/
   └─ Genie          /api/2.0/mcp/genie/{space_id}           (payables, etc.)
```

The agent treats every MCP tool — managed or custom — as just another tool.

## The tool sources

| Source | Via | What it provides |
|--------|-----|------------------|
| UC function | Managed MCP (UC functions) | A governed Python function (demo: a CAGR calculator) |
| Vector Search | Managed MCP (Vector Search) | Semantic search over internal policy / contract docs |
| Genie | Managed MCP (Genie) | NL questions over your sales / customer data |
| Custom MCP server | Databricks App (`custom-mcp-server/`) | Accounts payable / receivable, pending approvals |

Each is optional and config-driven — the agent starts with whatever you have
configured and simply skips the rest.

---

## File tree

```
02-mcp-tool-calling-agent-on-apps/
├── README.md                 # this guide
├── .env.example              # placeholders only (no secrets)
├── agent/
│   ├── config.py             # all config from env vars (nothing hardcoded)
│   ├── agent.py              # ResponsesAgent + LangGraph tool-calling agent
│   ├── app.py                # Databricks App entrypoint (serves the agent)
│   ├── app.yaml              # Databricks Apps runtime config
│   ├── requirements.txt      # coherent, resolved pins
│   └── log_and_evaluate.py   # OPTIONAL MLflow log + Agent Evaluation (not run in CI)
├── custom-mcp-server/
│   ├── app.py                # FastMCP streamable-HTTP server (genericized tools)
│   ├── app.yaml
│   ├── requirements.txt
│   ├── smoke.py              # offline tools/list smoke (no workspace)
│   └── README.md
└── setup/
    └── define_tools.py       # genericized UC function + Vector Search index
```

---

## Prerequisites

- Python 3.11+
- A Databricks workspace, the Databricks CLI (`databricks`) configured, and a
  foundation-model endpoint (e.g. `databricks-claude-sonnet-5-5`).

---

## End-to-end setup

### 1. Create the managed-MCP objects (UC function + Vector Search index)

Run `setup/define_tools.py` in a Databricks notebook or job. Set the `catalog`
and `schema` widgets (or `UC_CATALOG` / `UC_SCHEMA` env vars). It creates a UC
function and a Vector Search index over inline sample documents — no internal
tables, no API keys. Note the catalog/schema it prints.

> The **catalog must already exist** — the script creates only the schema (you
> need `USE CATALOG` + `CREATE SCHEMA` on the target catalog). It does not try to
> create the catalog, since `CREATE CATALOG IF NOT EXISTS` requires the
> metastore-level `CREATE CATALOG` privilege even when the catalog is present.

### 2. (Optional) Create a Genie space

In the workspace, create a Genie space over your sales/customer tables and copy
its **space id** from the URL. Skip this to run without the Genie tool.

### 3. Deploy the custom MCP server (a Databricks App)

See [`custom-mcp-server/README.md`](custom-mcp-server/README.md). In short, from
`custom-mcp-server/`:

```bash
databricks apps create mcp-example-operations
DATABRICKS_USERNAME=$(databricks current-user me | jq -r .userName)
databricks sync . "/Users/$DATABRICKS_USERNAME/mcp-example-operations"
databricks apps deploy mcp-example-operations \
  --source-code-path "/Workspace/Users/$DATABRICKS_USERNAME/mcp-example-operations"
```

Copy the deployed URL and append `/mcp/` (trailing slash required).

### 4. Configure and deploy the agent (a Databricks App)

Fill the config in `agent/app.yaml` `env:` (catalog, schema, Genie id, custom
MCP URL — see [`.env.example`](.env.example) for the full list). Then, from
`agent/`:

```bash
databricks apps create mcp-tool-calling-agent
DATABRICKS_USERNAME=$(databricks current-user me | jq -r .userName)
databricks sync . "/Users/$DATABRICKS_USERNAME/mcp-tool-calling-agent"
databricks apps deploy mcp-tool-calling-agent \
  --source-code-path "/Workspace/Users/$DATABRICKS_USERNAME/mcp-tool-calling-agent"
```

**Grant the agent's service principal `CAN_USE` on `mcp-example-operations`** —
the agent reaches the custom MCP app over OAuth as its own service principal.

---

## Call the agent

```bash
# Get an OAuth token for the deployed app (a legacy PAT is rejected at the edge).
TOKEN=$(databricks auth token -p <your-profile> | jq -r .access_token)
APP_URL=https://mcp-tool-calling-agent-<id>.<region>.databricksapps.com

curl -s "$APP_URL/invocations" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"input": [{"role": "user", "content": "How much do we owe suppliers?"}]}'
```

Or from Python:

```python
import os
import requests

app_url = os.environ["APP_URL"]
token = os.environ["DATABRICKS_TOKEN"]  # an OAuth access token for the app
resp = requests.post(
    f"{app_url}/invocations",
    headers={"Authorization": f"Bearer {token}"},
    json={"input": [{"role": "user", "content": "Which customers owe us money?"}]},
    timeout=120,
)
print(resp.json())
```

Add `"stream": true` to the body to receive Server-Sent Events instead.

---

## Offline smoke (no workspace, no creds)

The custom MCP server has a self-contained smoke that lists its tools over
`/mcp/`:

```bash
cd custom-mcp-server
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python smoke.py        # ends with: OK — exactly the 3 expected tools discovered over /mcp/.
```

---

## MLflow tracing + evaluation

`agent/agent.py` calls `mlflow.langchain.autolog()`, so every LLM and tool step
is traced automatically. To log the agent as an MLflow model and run Mosaic AI
Agent Evaluation, see `agent/log_and_evaluate.py` (optional; needs
`databricks-agents`; run against a real workspace, not in CI).

---

## Databricks Apps gotchas (honored here)

- **`/mcp/` needs the trailing slash** — connect to `/mcp/`, not `/mcp`.
- **Never use `/healthz`** — Databricks Apps reserves it; health is on `/health`.
- **OAuth, not PAT, at the front door** — M2M callers need `CAN_USE` on the app.
- **OBO** needs `user_api_scopes` in `app.yaml` + the `x-forwarded-access-token`
  header (demonstrated in `custom-mcp-server/app.py`).

---

## Placeholders you must fill to run it live

| Placeholder | Where | What it is |
|-------------|-------|------------|
| `DATABRICKS_HOST` | `.env` (local) | Workspace URL (injected on a deployed App) |
| `LLM_ENDPOINT_NAME` | `agent/app.yaml`, `.env` | Foundation-model/serving endpoint for the LLM |
| `UC_CATALOG` / `UC_SCHEMA` | `agent/app.yaml`, `.env`, `setup/define_tools.py` | Catalog/schema holding the UC function + VS index |
| `GENIE_SPACE_ID` | `agent/app.yaml`, `.env` | Genie space id (optional) |
| `CUSTOM_MCP_SERVER_URL` | `agent/app.yaml`, `.env` | Deployed custom MCP app URL, ending in `/mcp/` |
| custom MCP app name | step 3 | `databricks apps create mcp-...` name |
| agent app name | step 4 | `databricks apps create ...` name |
| `CAN_USE` grant | step 4 | Agent SP granted `CAN_USE` on the custom MCP app |

---

## Reference docs

- MCP on Databricks: https://docs.databricks.com/aws/en/generative-ai/mcp/
- Use MCP servers in custom agents: https://docs.databricks.com/aws/en/generative-ai/mcp/
- Custom MCP servers (on Apps): https://docs.databricks.com/aws/en/generative-ai/mcp/custom-mcp
- Author an agent (`ResponsesAgent`): https://docs.databricks.com/aws/en/generative-ai/agent-framework/
- Agent Evaluation: https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/
- Templates: `databricks/app-templates` → `agent-langgraph`; `databrickslabs/mcp` → `examples/custom-server`

## License

Released under the [MIT License](../LICENSE). Unofficial, educational sample,
provided as-is (see the disclaimer at the top).
