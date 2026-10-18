"""
agent_shims.py — lightweight, runnable STAND-INS for your app's real agents.

WHY THIS FILE EXISTS
--------------------
The MCP server (mcp_server.py) exposes two *tools* that, in production, call
the two LangGraph agent subsystems that live inside your Databricks App:

    (a) RAG Q&A agent           -> ask_credit_question
    (b) Document Classification -> classify_document

So that this example *runs standalone* for a demo (no Databricks workspace, no
model endpoints, no vector index), each real agent is replaced here by a tiny
deterministic shim with the SAME INPUT/OUTPUT SIGNATURE as the real thing.

HOW TO GO LIVE
--------------
Each shim below has a prominent `# TODO: replace with your real ...` marker.
When you fold this into app/serve.py, swap the shim body for a call into your
actual graph (e.g. `run_rag_agent(...)`, `classification_pipeline(...)`). Keep
the function signatures identical and the MCP tools in mcp_server.py keep working
unchanged.

Nothing in this file imports Databricks or network libraries — that keeps the
smoke test (scripts/smoke.py) fully offline.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# (a) RAG Q&A agent — the SIMPLEST worked path, and the one we lead with.
#     Real signature: (query: str, client_id: str) -> cited answer
# ---------------------------------------------------------------------------
def run_rag_agent(query: str, client_id: str) -> dict[str, Any]:
    """Stand-in for your app's RAG Q&A LangGraph agent.

    In production this retrieves from the credit knowledge base (today a
    Databricks Vector Search index backed by databricks-qwen3-embedding),
    then synthesizes a grounded answer with databricks-llama-4-maverick and
    returns the answer WITH its source citations.

    # TODO: replace with your real run_rag_agent / RAG LangGraph graph.
    #        e.g.  return rag_graph.invoke({"query": query, "client_id": client_id})
    """
    return {
        "answer": (
            f"[SHIM] For client {client_id}: based on the credit file, the "
            f"answer to '{query}' is a worked example — replace this shim with "
            f"your real RAG agent to get a grounded, model-generated answer."
        ),
        # Real agents return the retrieved chunks they cited; keep this shape.
        "citations": [
            {"doc_id": "demo-doc-001", "snippet": "Demo citation snippet.", "score": 0.91},
            {"doc_id": "demo-doc-002", "snippet": "Another demo snippet.", "score": 0.84},
        ],
        "model": "databricks-llama-4-maverick",
    }


# ---------------------------------------------------------------------------
# (b) Document Classification.
#     Real signature: (file path/bytes) -> (category, confidence)
# ---------------------------------------------------------------------------
def run_classification(file_ref: str) -> dict[str, Any]:
    """Stand-in for the document classification pipeline.

    `file_ref` is a path or identifier the real pipeline can resolve (a UC
    Volume path, a signed URL, etc.). The shim just inspects the string.

    # TODO: replace with your real classification_pipeline(file_ref).
    """
    lowered = file_ref.lower()
    if "paystub" in lowered or "pay_stub" in lowered:
        category = "INCOME_PROOF"
    elif "deed" in lowered or "title" in lowered:
        category = "COLLATERAL"
    elif "bank" in lowered or "statement" in lowered:
        category = "BANK_STATEMENT"
    else:
        category = "OTHER"
    return {"category": category, "confidence": 0.88, "file_ref": file_ref}
