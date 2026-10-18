"""
define_tools.py — create the Unity Catalog objects the managed MCP servers expose.

Run this once in a Databricks notebook or job (it needs `spark` / `dbutils`). It
is fully **parameterized** (catalog / schema via widgets or env vars) and
**self-contained** (sample data generated inline — no internal tables, no
external datasets, no inline API keys). All SQL uses `CREATE OR REPLACE`, so it
is idempotent.

It creates two things the agent reaches through Databricks **managed** MCP:
  1. A Unity Catalog Python function (exposed via the UC Functions managed MCP),
     here a keyless CAGR calculator — a stand-in for any governed function.
  2. A Vector Search index over a small, inline sample-documents table (exposed
     via the Vector Search managed MCP).

Parameters (set as notebook widgets or environment variables):
    catalog           Unity Catalog catalog           (default: main)
    schema            schema                           (default: agent_demo)
    vs_endpoint       Vector Search endpoint name      (default: agent_demo_vs_endpoint)
    embedding_model   embedding model serving endpoint (default: databricks-gte-large-en)

Prereq for the Vector Search step:  %pip install databricks-vectorsearch
"""

from __future__ import annotations

import os


def _param(name: str, default: str) -> str:
    """Read a parameter from a notebook widget if present, else an env var, else default."""
    try:
        return dbutils.widgets.get(name)  # noqa: F821 — `dbutils` is a Databricks global
    except Exception:
        return os.getenv(name.upper(), default)


def create_uc_function(catalog: str, schema: str) -> None:
    """Create a keyless UC Python function (exposed via the UC Functions managed MCP)."""
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")  # noqa: F821
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")  # noqa: F821

    spark.sql(  # noqa: F821
        f"""
        CREATE OR REPLACE FUNCTION {catalog}.{schema}.annualized_growth_rate(
            start_value DOUBLE COMMENT 'Value at the start of the period',
            end_value DOUBLE COMMENT 'Value at the end of the period',
            years INT COMMENT 'Number of years in the period'
        )
        RETURNS DOUBLE
        LANGUAGE PYTHON
        COMMENT 'Compound annual growth rate (CAGR) as a decimal, e.g. 0.12 means 12%/yr.'
        AS $$
            if start_value is None or end_value is None or not years or start_value <= 0:
                return None
            return (end_value / start_value) ** (1.0 / years) - 1.0
        $$
        """
    )
    print(f"Created function {catalog}.{schema}.annualized_growth_rate")

    # --- OPTIONAL: an external-API function (e.g. web search) WITHOUT inline keys ---
    # Do NOT paste an API token into source. The secure, documented pattern is a
    # Unity Catalog HTTP *connection* that stores the credential, called via the
    # `http_request` SQL function — the token never appears in code. Alternatively,
    # host external-API tools in the custom MCP server, where the app environment
    # (a Databricks secret) holds the key at runtime. Example (fill the connection
    # name; verify the current syntax in the docs before enabling):
    #
    #   spark.sql(f'''
    #     CREATE OR REPLACE FUNCTION {catalog}.{schema}.web_search(query STRING)
    #     RETURNS STRING
    #     RETURN (
    #       SELECT text FROM http_request(
    #         conn => '<your_uc_http_connection>',   -- stores the API key securely
    #         method => 'GET',
    #         path => '/search',
    #         params => map('q', query)
    #       )
    #     )
    #   ''')


# Sample policy/contract documents — fully fictional, generated inline so there
# is no dependency on any internal or external table.
SAMPLE_DOCS = [
    (
        "doc-001",
        "Standard Contract — Cancellation",
        "Either party may cancel this agreement with 30 days written notice. "
        "Fees already paid for the current billing period are non-refundable.",
        "sample://contracts/standard.md",
    ),
    (
        "doc-002",
        "Standard Contract — Payment Terms",
        "Invoices are due within 30 days of receipt. Late payments accrue "
        "interest at 1.5% per month on the outstanding balance.",
        "sample://contracts/standard.md",
    ),
    (
        "doc-003",
        "Expense Policy — Travel",
        "Employee travel must be booked at least 14 days in advance where "
        "possible. Economy fares are the default for flights under six hours.",
        "sample://policies/expense.md",
    ),
    (
        "doc-004",
        "Data Retention Policy",
        "Operational records are retained for seven years. Personal data is "
        "deleted within 90 days of an account being closed.",
        "sample://policies/data-retention.md",
    ),
]


def create_vector_search_index(catalog: str, schema: str, vs_endpoint: str, embedding_model: str) -> None:
    """Create a sample-docs table + a Vector Search delta-sync index over it."""
    source_table = f"{catalog}.{schema}.policy_docs"
    index_name = f"{catalog}.{schema}.policy_docs_index"

    df = spark.createDataFrame(SAMPLE_DOCS, ["chunk_id", "title", "chunk_text", "doc_uri"])  # noqa: F821
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(source_table)
    # Vector Search delta-sync indexes require Change Data Feed on the source table.
    spark.sql(f"ALTER TABLE {source_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")  # noqa: F821
    print(f"Created source table {source_table}")

    from databricks.vector_search.client import VectorSearchClient

    client = VectorSearchClient()
    # Create the endpoint if it does not already exist.
    try:
        client.create_endpoint(name=vs_endpoint)
    except Exception as e:  # noqa: BLE001 — typically "already exists"
        print(f"(endpoint note: {e})")

    client.create_delta_sync_index(
        endpoint_name=vs_endpoint,
        source_table_name=source_table,
        index_name=index_name,
        pipeline_type="TRIGGERED",
        primary_key="chunk_id",
        embedding_source_column="chunk_text",
        embedding_model_endpoint_name=embedding_model,
        columns_to_sync=["title", "chunk_text", "doc_uri"],
    )
    print(f"Created Vector Search index {index_name} on endpoint {vs_endpoint}")


def main() -> None:
    catalog = _param("catalog", "main")
    schema = _param("schema", "agent_demo")
    vs_endpoint = _param("vs_endpoint", "agent_demo_vs_endpoint")
    embedding_model = _param("embedding_model", "databricks-gte-large-en")

    create_uc_function(catalog, schema)
    create_vector_search_index(catalog, schema, vs_endpoint, embedding_model)

    print(
        "\nDone. Point the agent at these managed MCP servers "
        f"(set UC_CATALOG={catalog}, UC_SCHEMA={schema}):\n"
        f"  functions:      /api/2.0/mcp/functions/{catalog}/{schema}\n"
        f"  vector-search:  /api/2.0/mcp/vector-search/{catalog}/{schema}"
    )


if __name__ == "__main__":
    main()
