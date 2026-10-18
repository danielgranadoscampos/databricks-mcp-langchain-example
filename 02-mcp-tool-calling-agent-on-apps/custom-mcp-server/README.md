# Custom MCP server (Databricks App)

A small [FastMCP](https://github.com/modelcontextprotocol/python-sdk)
streamable-HTTP server, hosted on a **Databricks App**, that exposes three
genericized "operations" tools. The LangGraph agent in [`../agent/`](../agent/)
calls these tools alongside the Databricks **managed** MCP servers.

> Part of an **unofficial, educational** example (see the [top-level
> README](../README.md)). Not an official Databricks product.

## Tools

| MCP tool                   | Returns                                            |
|----------------------------|----------------------------------------------------|
| `list_accounts_payable`    | Outstanding payables (fictional sample)            |
| `list_accounts_receivable` | Outstanding receivables (fictional sample)         |
| `get_pending_approvals`    | Per-user invoice approvals — demonstrates **OBO**  |

All data is fictional and generated in-process, so the server runs with **no
workspace, no database, and no secrets**. Swap the tool bodies for real queries
(e.g. a SQL warehouse read via `WorkspaceClient`) when you adapt it.

## Offline smoke (no Databricks creds)

```bash
cd custom-mcp-server
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python smoke.py
```

Expected output ends with:

```
tools/list returned: ['list_accounts_payable', 'list_accounts_receivable', 'get_pending_approvals']
tool count: 3
OK — exactly the 3 expected tools discovered over /mcp/.
OBO check: with-header observed = <matches sent sentinel> | without-header observed = None | tool acting_as = 'user (OBO)'
OK — forwarded OBO header is observed by the server tool; absent header -> None.
```

The smoke starts the server in-process, lists its tools over `/mcp/`, then
proves the OBO path: it sends a request carrying `x-forwarded-access-token` and
asserts the tool observes that value (and sees nothing when the header is
absent).

## Run locally

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
# MCP endpoint: http://localhost:8000/mcp/   (trailing slash!)
# Health:       http://localhost:8000/health -> {"status":"ok","mcp_endpoint":"/mcp/"}
```

## Deploy as a Databricks App

A custom MCP server is just a Databricks App whose name starts with `mcp-`.

```bash
# From this directory (it is the app root — app.yaml lives here).
export DATABRICKS_CONFIG_PROFILE=<your-profile>
databricks apps create mcp-example-operations

DATABRICKS_USERNAME=$(databricks current-user me | jq -r .userName)
databricks sync . "/Users/$DATABRICKS_USERNAME/mcp-example-operations"
databricks apps deploy mcp-example-operations \
  --source-code-path "/Workspace/Users/$DATABRICKS_USERNAME/mcp-example-operations"
```

The deployed endpoint is:

```
https://mcp-example-operations-<id>.<region>.databricksapps.com/mcp/
```

Put that URL in the agent's `CUSTOM_MCP_SERVER_URL` (see [`../.env.example`](../.env.example)).

### Gotchas this server honors

- **`/mcp/` needs the trailing slash.** The FastMCP sub-app is mounted at `/mcp`
  with `streamable_http_path="/"`, so the real endpoint is `/mcp/`. A request to
  `/mcp` gets a 307 redirect whose `Location` may come back as the app's
  internal host and not be followed — connect to `/mcp/` directly.
- **Never use `/healthz`.** Databricks Apps reserves it (empty 200 before the
  request reaches your app), so this server serves health JSON on `/health`.
- **OAuth, not PAT, at the front door.** M2M callers (like the agent's service
  principal) must have `CAN_USE` on this app; a legacy PAT is rejected by the
  Apps front door. Get an OAuth token with `databricks auth token -p <profile>`.
- **OBO** needs user authorization enabled on the app plus `user_api_scopes` in
  [`app.yaml`](app.yaml); the user token then arrives in
  `x-forwarded-access-token`.

## Reference

- Databricks "Custom MCP servers" (host your own MCP server on Apps):
  https://docs.databricks.com/aws/en/generative-ai/mcp/custom-mcp
- `databrickslabs/mcp` → `examples/custom-server` (the canonical template).
