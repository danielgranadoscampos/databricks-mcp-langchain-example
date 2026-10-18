"""
agent.py — a LangGraph tool-calling agent wrapped as an MLflow `ResponsesAgent`.

This keeps the MLflow `ResponsesAgent` authoring interface (the recommended way
to author Mosaic AI agents) but is **deployed as a Databricks App** (see
`app.py`), NOT via Model Serving / `agents.deploy`.

The agent connects to one or more MCP servers, discovers their tools, and runs a
standard tool-calling loop:
  - Databricks **managed** MCP servers (UC functions, Genie, Vector Search), and
  - a **custom** MCP server you host on a Databricks App (see
    `../custom-mcp-server/`),
all driven by `config.py` (env vars) — nothing is hardcoded.

MODERNIZATION NOTE
------------------
Both managed and custom MCP servers are reached through a single
`databricks_mcp.DatabricksMCPClient(server_url, workspace_client)`. The current
client handles OAuth for custom Databricks Apps URLs and adapts to the installed
`mcp` major version itself, so there is no hand-rolled streamable-HTTP /
`DatabricksOAuthClientProvider` plumbing here. (Reaching a custom
`*.databricksapps.com` MCP server requires the workspace client to use OAuth —
inside a Databricks App the injected service-principal credentials are OAuth, so
a no-arg `WorkspaceClient()` works, provided that service principal has
`CAN_USE` on the custom MCP app.)

Reference: Databricks "Use MCP servers in custom agents"
https://docs.databricks.com/aws/en/generative-ai/mcp/
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Generator, List, Optional, Sequence, TypedDict, Union
from uuid import uuid4

import mlflow
from pydantic import create_model

from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks
from databricks_mcp import DatabricksMCPClient

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode

from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

import config

# Enable MLflow 3 autologging/tracing: every LLM + tool step is traced.
mlflow.langchain.autolog()


#####################
## MCP tool creation
#####################
class MCPTool(BaseTool):
    """A LangChain tool that proxies a single MCP tool on a Databricks MCP server."""

    def __init__(self, name: str, description: str, args_schema: type, server_url: str, ws: WorkspaceClient):
        super().__init__(name=name, description=description, args_schema=args_schema)
        # Store non-pydantic attributes out of band.
        object.__setattr__(self, "server_url", server_url)
        object.__setattr__(self, "workspace_client", ws)

    def _run(self, **kwargs) -> str:
        client = DatabricksMCPClient(server_url=self.server_url, workspace_client=self.workspace_client)
        response = client.call_tool(self.name, kwargs)
        # Prefer text content; fall back to any structured content.
        parts = [c.text for c in response.content if getattr(c, "text", None) is not None]
        if parts:
            return "".join(parts)
        structured = getattr(response, "structuredContent", None)
        return json.dumps(structured) if structured is not None else ""


# Map JSON-schema types to Python types for input validation.
_TYPE_MAPPING = {"integer": int, "number": float, "boolean": bool}


def create_langchain_tool_from_mcp(mcp_tool, server_url: str, ws: WorkspaceClient) -> MCPTool:
    """Convert one MCP tool definition into a LangChain `MCPTool`."""
    schema = mcp_tool.inputSchema.copy()
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    field_definitions: dict[str, Any] = {}
    for field_name, field_info in properties.items():
        field_type = _TYPE_MAPPING.get(field_info.get("type", "string"), str)
        if field_name in required:
            field_definitions[field_name] = (field_type, ...)
        else:
            field_definitions[field_name] = (field_type, None)

    args_schema = create_model(f"{mcp_tool.name}Args", **field_definitions)
    return MCPTool(
        name=mcp_tool.name,
        description=mcp_tool.description or f"Tool: {mcp_tool.name}",
        args_schema=args_schema,
        server_url=server_url,
        ws=ws,
    )


def create_mcp_tools(ws: WorkspaceClient, server_urls: List[str]) -> List[MCPTool]:
    """Discover and wrap all tools from the given MCP servers (managed or custom)."""
    tools: List[MCPTool] = []
    for server_url in server_urls:
        try:
            client = DatabricksMCPClient(server_url=server_url, workspace_client=ws)
            for mcp_tool in client.list_tools():
                tools.append(create_langchain_tool_from_mcp(mcp_tool, server_url, ws))
        except Exception as e:  # noqa: BLE001 — one bad server shouldn't sink the agent
            print(f"Warning: could not load tools from MCP server {server_url}: {e}")
    return tools


#####################
## Agent logic
#####################
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    custom_inputs: Optional[dict[str, Any]]
    custom_outputs: Optional[dict[str, Any]]


def create_tool_calling_agent(
    model: LanguageModelLike,
    tools: Union[ToolNode, Sequence[BaseTool]],
    system_prompt: Optional[str] = None,
):
    """Build a LangGraph state machine that loops the LLM over tool calls."""
    model = model.bind_tools(tools)

    def should_continue(state: AgentState):
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "continue"
        return "end"

    if system_prompt:
        preprocessor = RunnableLambda(
            lambda state: [{"role": "system", "content": system_prompt}] + state["messages"]
        )
    else:
        preprocessor = RunnableLambda(lambda state: state["messages"])
    model_runnable = preprocessor | model

    def call_model(state: AgentState, config: RunnableConfig):
        return {"messages": [model_runnable.invoke(state, config)]}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", RunnableLambda(call_model))
    workflow.add_node("tools", ToolNode(tools))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
    workflow.add_edge("tools", "agent")
    return workflow.compile()


class LangGraphResponsesAgent(ResponsesAgent):
    """Wrap the compiled LangGraph agent in the MLflow Responses API interface."""

    def __init__(self, agent):
        self.agent = agent

    def _responses_to_cc(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert a Responses API input item into ChatCompletion messages."""
        msg_type = message.get("type")
        if msg_type == "function_call":
            return [
                {
                    "role": "assistant",
                    "content": "tool call",
                    "tool_calls": [
                        {
                            "id": message["call_id"],
                            "type": "function",
                            "function": {"arguments": message["arguments"], "name": message["name"]},
                        }
                    ],
                }
            ]
        elif msg_type == "message" and isinstance(message["content"], list):
            return [{"role": message["role"], "content": content["text"]} for content in message["content"]]
        elif msg_type == "reasoning":
            return [{"role": "assistant", "content": json.dumps(message["summary"])}]
        elif msg_type == "function_call_output":
            return [{"role": "tool", "content": message["output"], "tool_call_id": message["call_id"]}]
        compatible_keys = ["role", "content", "name", "tool_calls", "tool_call_id"]
        filtered = {k: v for k, v in message.items() if k in compatible_keys}
        return [filtered] if filtered else []

    def _langchain_to_responses(self, messages: list[BaseMessage]) -> list[dict[str, Any]]:
        """Convert LangChain messages into Responses output items."""
        for message in messages:
            message = message.model_dump()
            role = message["type"]
            if role == "ai":
                if tool_calls := message.get("tool_calls"):
                    return [
                        self.create_function_call_item(
                            id=message.get("id") or str(uuid4()),
                            call_id=tool_call["id"],
                            name=tool_call["name"],
                            arguments=json.dumps(tool_call["args"]),
                        )
                        for tool_call in tool_calls
                    ]
                return [
                    self.create_text_output_item(
                        text=message["content"], id=message.get("id") or str(uuid4())
                    )
                ]
            elif role == "tool":
                return [
                    self.create_function_call_output_item(
                        call_id=message["tool_call_id"], output=message["content"]
                    )
                ]
            elif role == "user":
                return [message]
        return []

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        cc_msgs = []
        for msg in request.input:
            cc_msgs.extend(self._responses_to_cc(msg.model_dump()))

        for event in self.agent.stream({"messages": cc_msgs}, stream_mode=["updates", "messages"]):
            if event[0] == "updates":
                for node_data in event[1].values():
                    if "messages" in node_data:
                        for item in self._langchain_to_responses(node_data["messages"]):
                            yield ResponsesAgentStreamEvent(type="response.output_item.done", item=item)
            elif event[0] == "messages":
                try:
                    chunk = event[1][0]
                    if isinstance(chunk, AIMessageChunk) and (content := chunk.content):
                        yield ResponsesAgentStreamEvent(
                            **self.create_text_delta(delta=content, item_id=chunk.id)
                        )
                except Exception:
                    pass


def build_workspace_client() -> WorkspaceClient:
    """The agent's workspace identity.

    No-arg `WorkspaceClient()` uses the Databricks App's injected service
    principal (OAuth) — required for calling a custom MCP app. For local runs it
    uses your `DATABRICKS_HOST`/token from the environment or a CLI profile.
    """
    return WorkspaceClient()


def build_agent() -> LangGraphResponsesAgent:
    """Assemble the agent: discover MCP tools, build the graph, wrap it."""
    ws = build_workspace_client()
    server_urls = config.managed_mcp_server_urls() + config.custom_mcp_server_urls()
    tools = create_mcp_tools(ws, server_urls)
    print(f"Agent initialized with {len(tools)} MCP tool(s) from {len(server_urls)} server(s).")
    llm = ChatDatabricks(endpoint=config.LLM_ENDPOINT_NAME)
    graph = create_tool_calling_agent(llm, tools, config.SYSTEM_PROMPT)
    return LangGraphResponsesAgent(graph)


# MLflow "model as code": building AGENT at import and calling set_model lets the
# optional `log_and_evaluate.py` log this file directly as the model. The App
# entrypoint (`app.py`) imports this module lazily, on first request.
AGENT = build_agent()
mlflow.models.set_model(AGENT)
