"""PostgreSQL database layer for datasets, queries, and results"""
import json
import os
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
DATABASE_URL = os.getenv("DATABASE_URL", "")
_pool = None  # lazily initialized with ThreadedConnectionPool


def is_available() -> bool:
    return bool(DATABASE_URL)


def get_pool():
    global _pool
    if _pool is None:
        from psycopg2.pool import ThreadedConnectionPool
        _pool = ThreadedConnectionPool(1, 10, DATABASE_URL)
    return _pool


@contextmanager
def get_conn():
    conn = get_pool().getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        get_pool().putconn(conn)


def init_db():
    if not is_available():
        logger.info("DATABASE_URL not set — skipping DB init")
        return
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS datasets (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(255) UNIQUE NOT NULL,
                        created_at TIMESTAMP DEFAULT NOW()
                    );
                    CREATE TABLE IF NOT EXISTS dataset_categories (
                        id SERIAL PRIMARY KEY,
                        dataset_id INTEGER REFERENCES datasets(id) ON DELETE CASCADE,
                        name TEXT NOT NULL,
                        documents JSONB NOT NULL,
                        created_at TIMESTAMP DEFAULT NOW()
                    );
                    CREATE TABLE IF NOT EXISTS queries (
                        id SERIAL PRIMARY KEY,
                        query_id VARCHAR(255) UNIQUE,
                        query_text TEXT NOT NULL,
                        ground_truth TEXT NOT NULL,
                        category VARCHAR(255),
                        dataset_name VARCHAR(255),
                        created_at TIMESTAMP DEFAULT NOW()
                    );
                    CREATE TABLE IF NOT EXISTS results (
                        id SERIAL PRIMARY KEY,
                        job_id VARCHAR(255) UNIQUE,
                        query TEXT,
                        model VARCHAR(255),
                        dataset VARCHAR(255),
                        result_json JSONB,
                        created_at TIMESTAMP DEFAULT NOW()
                    );
                """)
        _seed_from_json()
    except Exception:
        logger.warning("DB init failed — running without database", exc_info=True)


def _seed_from_json():
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM datasets")
            if cur.fetchone()[0] > 0:
                return
            datasets_path = data_dir / "datasets.json"
            if datasets_path.exists():
                with open(datasets_path) as f:
                    datasets = json.load(f)
                for name, items in datasets.items():
                    cur.execute(
                        "INSERT INTO datasets (name) VALUES (%s) RETURNING id", (name,)
                    )
                    ds_id = cur.fetchone()[0]
                    for item in items:
                        if isinstance(item, dict):
                            doc_name = item["id"]
                            doc_data = json.dumps([item])
                        else:
                            doc_name = item
                            doc_data = json.dumps([item])
                        cur.execute(
                            "INSERT INTO dataset_categories (dataset_id, name, documents) VALUES (%s, %s, %s)",
                            (ds_id, doc_name, doc_data),
                        )
                logger.info("Seeded %d datasets from JSON", len(datasets))
            cur.execute("SELECT COUNT(*) FROM queries")
            if cur.fetchone()[0] > 0:
                return
            queries_path = data_dir / "rag_queries.json"
            if queries_path.exists():
                with open(queries_path) as f:
                    queries = json.load(f)
                for q in queries:
                    cur.execute(
                        "INSERT INTO queries (query_id, query_text, ground_truth, category, dataset_name) VALUES (%s, %s, %s, %s, %s)",
                        (q.get("id"), q["query"], q["ground_truth"], q.get("category"), q["relevant_dataset"]),
                    )
                logger.info("Seeded %d queries from JSON", len(queries))
        conn.commit()


def get_datasets() -> list[str]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT name FROM datasets ORDER BY name")
            return [row[0] for row in cur.fetchall()]


def get_dataset_documents(name: str) -> list[str] | list[dict]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT c.documents
                FROM dataset_categories c
                JOIN datasets d ON d.id = c.dataset_id
                WHERE d.name = %s
                ORDER BY c.id
            """, (name,))
            rows = cur.fetchall()
            if not rows:
                raise ValueError(f"Dataset '{name}' not found")
            docs = []
            for (doc_list,) in rows:
                docs.extend(json.loads(doc_list) if isinstance(doc_list, str) else doc_list)
            return docs


def get_queries() -> list[dict]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT query_id, query_text, ground_truth, category, dataset_name FROM queries ORDER BY id")
            return [
                {"id": r[0], "query": r[1], "ground_truth": r[2], "category": r[3], "relevant_dataset": r[4]}
                for r in cur.fetchall()
            ]


def save_result(job_id: str, query: str, model: str, dataset: str, result: dict):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO results (job_id, query, model, dataset, result_json)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (job_id) DO UPDATE SET result_json = EXCLUDED.result_json""",
                (job_id, query, model, dataset, json.dumps(result)),
            )
        conn.commit()


def get_result(job_id: str) -> Optional[dict]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT result_json FROM results WHERE job_id = %s", (job_id,))
            row = cur.fetchone()
            return row[0] if row else None


def list_results(limit: int = 50, filter_model: str = None) -> list[dict]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            if filter_model:
                cur.execute(
                    "SELECT result_json FROM results WHERE model = %s ORDER BY created_at DESC LIMIT %s",
                    (filter_model, limit),
                )
            else:
                cur.execute(
                    "SELECT result_json FROM results ORDER BY created_at DESC LIMIT %s",
                    (limit,),
                )
            return [row[0] for row in cur.fetchall()]


def save_dataset(name: str, documents: list[str] | list[dict]):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM datasets WHERE name = %s", (name,))
            row = cur.fetchone()
            if row:
                ds_id = row[0]
                cur.execute("DELETE FROM dataset_categories WHERE dataset_id = %s", (ds_id,))
            else:
                cur.execute("INSERT INTO datasets (name) VALUES (%s) RETURNING id", (name,))
                ds_id = cur.fetchone()[0]
            for doc in documents:
                if isinstance(doc, dict):
                    doc_name = doc["id"]
                    doc_data = json.dumps([doc])
                else:
                    doc_name = doc
                    doc_data = json.dumps([doc])
                cur.execute(
                    "INSERT INTO dataset_categories (dataset_id, name, documents) VALUES (%s, %s, %s)",
                    (ds_id, doc_name, doc_data),
                )
        conn.commit()


def remove_dataset(name: str):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM datasets WHERE name = %s", (name,))
        conn.commit()
