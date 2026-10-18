"""
app.py — Databricks App entrypoint that SERVES the ResponsesAgent over HTTP.

This is the modernization: instead of deploying the agent to **Model Serving**
(`databricks-agents` / `agents.deploy`), we package it as a **Databricks App**.
Databricks recommends Apps when you want full control of the code, server
config, and git-based versioning — see "Run agents on Databricks Apps":
https://docs.databricks.com/aws/en/generative-ai/agent-framework/

A thin FastAPI server exposes the agent (see `agent.py` for the
`ResponsesAgent` itself):

  GET  /             -> service info
  GET  /health       -> {"status": "ok"}         (NOT /healthz — reserved by Apps)
  POST /invocations  -> run the agent

`POST /invocations` accepts a Responses request body, e.g.:

    {"input": [{"role": "user", "content": "What is owed to suppliers?"}]}

By default it returns the full `ResponsesAgentResponse` as JSON. Add
`{"stream": true}` (or `?stream=true`) to stream Server-Sent Events instead —
each `data:` line is one `ResponsesAgentStreamEvent`, terminated by
`data: [DONE]`.

> Batteries-included alternative: `mlflow.genai.agent_server.AgentServer` (used
> by the `databricks/app-templates` → `agent-langgraph` template) wraps this
> boilerplate — an OpenAI-compatible chat proxy, trace version tracking, etc. We
> hand-roll a minimal server here to keep the example transparent and
> dependency-light; swap it in for a production app.

The agent is built lazily on the first request, so importing this module (and
starting the process) does not require a live workspace connection.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from mlflow.types.responses import ResponsesAgentRequest

app = FastAPI(title="MCP tool-calling LangGraph agent")


@lru_cache(maxsize=1)
def get_agent():
    """Import + build the agent once, on first use (keeps import side-effect-free)."""
    import agent  # noqa: PLC0415 — deferred so module import needs no workspace

    return agent.AGENT


@app.get("/")
def index() -> dict:
    return {
        "name": "MCP tool-calling LangGraph agent",
        "endpoints": {"health": "/health", "invoke": "POST /invocations"},
    }


@app.get("/health")
def health() -> dict:
    # NOTE: `/healthz` is reserved by Databricks Apps (empty 200, shadows your
    # route). Always expose app health on a non-reserved path like `/health`.
    return {"status": "ok"}


def _wants_stream(request: Request, payload: dict) -> bool:
    if str(request.query_params.get("stream", "")).lower() in {"1", "true", "yes"}:
        return True
    return bool(payload.get("stream"))


@app.post("/invocations")
async def invocations(request: Request):
    payload = await request.json()
    stream = _wants_stream(request, payload)
    # `stream` is not a field of ResponsesAgentRequest — drop it before parsing.
    payload = {k: v for k, v in payload.items() if k != "stream"}
    req = ResponsesAgentRequest(**payload)
    agent = get_agent()

    if not stream:
        response = await run_in_threadpool(agent.predict, req)
        return JSONResponse(response.model_dump())

    def event_stream():
        # agent.predict_stream is a sync generator; FastAPI iterates it in a
        # worker thread, so the MCP client's internal event loop is isolated.
        for event in agent.predict_stream(req):
            yield f"data: {json.dumps(event.model_dump())}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("DATABRICKS_APP_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
