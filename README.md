# Databricks MCP and agent examples

This repository contains unofficial educational examples for building with Databricks Model Context Protocol (MCP) and agents. These examples are provided as-is for learning purposes; they are not an official Databricks product and are not officially supported or endorsed by Databricks.

## Examples

### [01 — MCP with an external LangChain client](01-mcp-langchain-external-client/)

Expose agents running in a Databricks App through a custom MCP server, then call those tools from an external LangChain agent that uses `ChatDatabricks` as its language-model brain. The example covers MCP tool discovery, OAuth authentication, and the tradeoffs of composing an external agent with app-hosted agents.

### [02 — MCP tool-calling agent on Databricks Apps](02-mcp-tool-calling-agent-on-apps/)

Build a LangGraph `ResponsesAgent` tool-calling agent that combines Databricks-managed MCP servers for Unity Catalog functions, Genie, and Vector Search with a custom MCP server, then deploy the agent as a Databricks App. This example is added in a separate pull request, so its directory may not yet exist on this branch.
