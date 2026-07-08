"""MCP client wrapper for Streamlit UI — connects via HTTP JSON-RPC"""

import json
import time

import httpx


MCP_URL = "http://mcp-sse:5100/call"


def call_tool(tool_name: str, arguments: dict | None = None) -> dict:
    resp = httpx.post(MCP_URL, json={"tool": tool_name, "arguments": arguments or {}}, timeout=600)
    resp.raise_for_status()
    return resp.json()


def list_datasets() -> list[str]:
    return call_tool("list_datasets").get("datasets", [])


def list_queries() -> list[dict]:
    return call_tool("list_queries").get("queries", [])


def submit_rag_job(query: str, model: str = "all", dataset: str = "programming_languages", ground_truth: str = "") -> dict:
    return call_tool("submit_rag_job", {
        "query": query,
        "model": model,
        "dataset": dataset,
        "ground_truth": ground_truth,
    })


def check_job_status(job_id: str) -> dict:
    return call_tool("check_job_status", {"job_id": job_id})


def wait_for_result(job_id: str, poll_interval: float = 2.0, timeout: float = 600.0) -> dict:
    start = time.time()
    while time.time() - start < timeout:
        status = check_job_status(job_id)
        if status.get("status") == "completed":
            return status
        if status.get("results") is not None:
            return status
        if "error" in status:
            return status
        time.sleep(poll_interval)
    return {"error": "Timeout waiting for result"}


def list_cached_results(limit: int = 50, model: str | None = None) -> list[dict]:
    args = {"limit": limit}
    if model:
        args["model"] = model
    return call_tool("list_cached_results", args).get("results", [])


def get_cached_result(job_id: str) -> dict | None:
    return call_tool("get_cached_result", {"job_id": job_id})


def get_dataset(name: str) -> dict:
    return call_tool("get_dataset", {"name": name})


def create_dataset(name: str, documents: list[str]) -> dict:
    return call_tool("create_dataset", {"name": name, "documents": documents})


def delete_dataset(name: str) -> dict:
    return call_tool("delete_dataset", {"name": name})
