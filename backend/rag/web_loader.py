"""Fetch web pages, extract text, and chunk into documents."""

import json
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter


def fetch_url(url: str) -> str:
    """Download a URL and return clean body text."""
    import requests
    from bs4 import BeautifulSoup

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; RAGBenchmark/1.0; +https://github.com/your-org/rag-benchmark)"
    }
    resp = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "iframe", "form"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = "\n".join(lines)

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text[:50_000]


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[dict]:
    """Split text into overlapping chunks, each returned as {id, text}."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_text(text)
    return [
        {"id": f"chunk-{i + 1}", "text": chunks[i]}
        for i in range(len(chunks))
    ]


def ingest_url(url: str, name: str, chunk_size: int = 500, chunk_overlap: int = 50) -> dict:
    """Full pipeline: fetch → extract → chunk → save as a dataset.

    Returns {"name": str, "chunks": int}.
    """
    text = fetch_url(url)
    docs = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Save to datasets.json
    from .experiment import DATA_DIR
    datasets_path = DATA_DIR / "datasets.json"
    data = json.loads(datasets_path.read_text()) if datasets_path.exists() else {}
    data[name] = docs
    datasets_path.write_text(json.dumps(data, indent=4))

    # Save to database if available
    from .database import is_available as db_available, save_dataset
    if db_available():
        try:
            save_dataset(name, docs)
        except Exception:
            pass

    return {"name": name, "chunks": len(docs)}
