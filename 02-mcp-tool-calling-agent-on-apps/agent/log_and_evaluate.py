"""
log_and_evaluate.py — OPTIONAL MLflow logging + Agent Evaluation (not run in CI).

The agent is DEPLOYED as a Databricks App (see app.py / app.yaml) — this script
does NOT deploy to Model Serving and does NOT call `agents.deploy`. It exists so
you can, from a Databricks notebook or a workspace-authenticated environment:

  1. Log the agent as an MLflow model ("model as code", from agent.py) and
     optionally register it to Unity Catalog — useful for governance, tracing,
     and keeping a versioned artifact alongside the App deployment.
  2. Evaluate the agent with Mosaic AI Agent Evaluation (`mlflow.genai.evaluate`)
     using predefined LLM judges, so you can track quality as you iterate.

Requirements (beyond agent/requirements.txt):
    pip install databricks-agents==1.12.0        # mlflow.genai scorers
Run it only against a real workspace (it builds the agent, which connects to
your configured MCP servers):
    python log_and_evaluate.py

References:
  - Author agents (ResponsesAgent): https://docs.databricks.com/aws/en/generative-ai/agent-framework/
  - Agent Evaluation: https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/
"""

from __future__ import annotations

import os

import mlflow
from mlflow.models.resources import DatabricksServingEndpoint

import config


def log_model():
    """Log agent.py as an MLflow model and (optionally) register it to Unity Catalog."""
    from importlib.metadata import version

    resources = [DatabricksServingEndpoint(endpoint_name=config.LLM_ENDPOINT_NAME)]

    with mlflow.start_run():
        logged = mlflow.pyfunc.log_model(
            name="agent",
            python_model="agent.py",          # model-as-code: see agent.py set_model(AGENT)
            resources=resources,
            pip_requirements=[
                "mlflow==" + version("mlflow"),
                "databricks-langchain==" + version("databricks-langchain"),
                "databricks-mcp==" + version("databricks-mcp"),
                "langgraph==" + version("langgraph"),
            ],
        )
    print(f"Logged model: {logged.model_uri}")

    # Optional: register to Unity Catalog for a governed, versioned artifact.
    catalog = os.getenv("UC_CATALOG")
    schema = os.getenv("UC_SCHEMA")
    model_name = os.getenv("UC_MODEL_NAME", "mcp_tool_calling_agent")
    if catalog and schema:
        mlflow.set_registry_uri("databricks-uc")
        uc_name = f"{catalog}.{schema}.{model_name}"
        registered = mlflow.register_model(model_uri=logged.model_uri, name=uc_name)
        print(f"Registered to Unity Catalog: {uc_name} (version {registered.version})")
    return logged


def evaluate():
    """Evaluate the agent with Mosaic AI Agent Evaluation (predefined LLM judges)."""
    from mlflow.genai.scorers import RelevanceToQuery, Safety
    from mlflow.types.responses import ResponsesAgentRequest

    from agent import AGENT

    # Each row's `inputs` dict is splatted as kwargs into predict_fn, so predict_fn
    # receives `input=[...]`.
    eval_dataset = [
        {
            "inputs": {
                "input": [{"role": "user", "content": "How much do we owe suppliers right now?"}]
            }
        },
        {
            "inputs": {
                "input": [{"role": "user", "content": "Which customers still owe us money?"}]
            }
        },
    ]

    def predict_fn(input):
        # ResponsesAgent.predict expects a ResponsesAgentRequest (it reads
        # request.input / request.custom_inputs) — NOT a plain dict.
        request = ResponsesAgentRequest(input=input)
        return AGENT.predict(request).model_dump()

    results = mlflow.genai.evaluate(
        data=eval_dataset,
        predict_fn=predict_fn,
        scorers=[RelevanceToQuery(), Safety()],
    )
    print("Evaluation complete — review results in the MLflow UI.")
    return results


if __name__ == "__main__":
    log_model()
    evaluate()
