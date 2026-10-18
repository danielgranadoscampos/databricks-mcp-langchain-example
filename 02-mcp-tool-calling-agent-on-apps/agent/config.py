"""
config.py — all agent configuration, driven by environment variables.

NOTHING is hardcoded here: workspace host, catalog/schema, Genie space id,
custom-MCP URL, LLM endpoint and system prompt all come from the environment
(set them in `app.yaml` `env:` for a deployed app, or in a local `.env` — see
`../.env.example`). Placeholder-looking values (empty, or wrapped in <angle
brackets>) are treated as "not set" so the app still starts; the corresponding
MCP server is simply skipped until you fill it in.

On a deployed Databricks App, `DATABRICKS_HOST`, `DATABRICKS_CLIENT_ID` and
`DATABRICKS_CLIENT_SECRET` are injected automatically, so a no-arg
`WorkspaceClient()` authenticates as the app's service principal.
"""

from __future__ import annotations

import os
from typing import List, Optional


def _clean(value: Optional[str]) -> Optional[str]:
    """Return a usable value, or None if it is empty or an obvious placeholder."""
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    # Treat `<...>` and `your-...` style placeholders as unset.
    if value.startswith("<") and value.endswith(">"):
        return None
    return value


# --- Workspace + model -----------------------------------------------------
# Databricks Apps inject DATABRICKS_HOST; strip any trailing slash so building
# `{host}/api/2.0/mcp/...` never produces a double slash.
WORKSPACE_HOST: Optional[str] = (_clean(os.getenv("DATABRICKS_HOST")) or "").rstrip("/") or None

# Any Databricks foundation-model or serving endpoint that supports tool calling.
LLM_ENDPOINT_NAME: str = _clean(os.getenv("LLM_ENDPOINT_NAME")) or "databricks-claude-sonnet-5-5"

# --- Managed MCP resources (optional, config-driven) -----------------------
UC_CATALOG: Optional[str] = _clean(os.getenv("UC_CATALOG"))
UC_SCHEMA: Optional[str] = _clean(os.getenv("UC_SCHEMA"))
GENIE_SPACE_ID: Optional[str] = _clean(os.getenv("GENIE_SPACE_ID"))

# --- Custom MCP server (your deployed Databricks App) ----------------------
# Full URL INCLUDING the trailing-slash /mcp/ path, e.g.
#   https://mcp-example-operations-<id>.<region>.databricksapps.com/mcp/
CUSTOM_MCP_SERVER_URL: Optional[str] = _clean(os.getenv("CUSTOM_MCP_SERVER_URL"))

# --- System prompt ---------------------------------------------------------
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful operations assistant for a fictional company. Answer questions
about the business using the tools available to you, and say which source you
used. The tools you may have, depending on configuration, are:
  - Unity Catalog functions (e.g. financial calculations) via a managed MCP server
  - An internal Genie space for questions about sales and customers
  - A Vector Search index of internal policy / contract documents
  - A custom MCP server exposing accounts payable, accounts receivable, and
    pending invoice approvals
If you cannot find the answer with your tools, say so plainly rather than
guessing.\
"""

SYSTEM_PROMPT: str = _clean(os.getenv("SYSTEM_PROMPT")) or DEFAULT_SYSTEM_PROMPT


def managed_mcp_server_urls() -> List[str]:
    """Build the list of Databricks-managed MCP server URLs from config.

    Only includes a server when the resources it needs are configured, so a
    partially-configured workspace still starts (with fewer tools).
    """
    urls: List[str] = []
    if not WORKSPACE_HOST:
        return urls
    if UC_CATALOG and UC_SCHEMA:
        urls.append(f"{WORKSPACE_HOST}/api/2.0/mcp/functions/{UC_CATALOG}/{UC_SCHEMA}")
        urls.append(f"{WORKSPACE_HOST}/api/2.0/mcp/vector-search/{UC_CATALOG}/{UC_SCHEMA}")
    if GENIE_SPACE_ID:
        urls.append(f"{WORKSPACE_HOST}/api/2.0/mcp/genie/{GENIE_SPACE_ID}")
    return urls


def custom_mcp_server_urls() -> List[str]:
    """Build the list of custom MCP server URLs (your Databricks App) from config."""
    return [CUSTOM_MCP_SERVER_URL] if CUSTOM_MCP_SERVER_URL else []
