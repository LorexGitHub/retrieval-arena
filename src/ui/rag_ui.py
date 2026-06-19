import json
import os
from pathlib import Path
import streamlit as st
import requests
from src.rag.config import EMBEDDING_MODELS

st.set_page_config(page_title="RAG Embedding Experiment", layout="wide")

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --bg: #0A0A0F;
  --bg-card: #12121A;
  --bg-surface: #1A1A26;
  --border: #252533;
  --border-hover: #3A3A50;
  --text: #E8E8F0;
  --text-secondary: #9999B0;
  --text-muted: #666680;
  --accent: #6366F1;
  --accent-hover: #4F46E5;
  --accent-subtle: rgba(99,102,241,0.08);
  --accent-glow: rgba(99,102,241,0.25);
  --green: #22C55E;
  --amber: #F59E0B;
  --red: #EF4444;
  --font: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  --radius: 10px;
  --radius-sm: 6px;
}

#root > div.main { background: var(--bg); font-family: var(--font); color: var(--text); }
.block-container { max-width: 1440px; padding: 1.25rem 1.75rem !important; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }

h1 { font-size: 1.375rem !important; font-weight: 700 !important; letter-spacing: -0.03em !important; color: var(--text) !important; }
h2 { font-size: 1.0625rem !important; font-weight: 600 !important; letter-spacing: -0.02em !important; color: var(--text) !important; }
h3 { font-size: 0.875rem !important; font-weight: 600 !important; color: var(--text-secondary) !important; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.75rem !important; }
p, li, .stMarkdown { color: var(--text-secondary); line-height: 1.6; }

.card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.25rem;
  transition: border-color 200ms ease, box-shadow 200ms ease;
}
.card:hover { border-color: var(--border-hover); box-shadow: 0 2px 16px rgba(0,0,0,0.25); }
.card-header {
  font-size: 0.6875rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-muted);
  margin-bottom: 0.75rem;
  padding-bottom: 0.75rem;
  border-bottom: 1px solid var(--border);
}

.badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 3px 10px;
  border-radius: 100px;
  font-size: 0.75rem;
  font-weight: 500;
  background: var(--accent-subtle);
  color: #A5B4FC;
  border: 1px solid rgba(99,102,241,0.2);
}
.badge-dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--accent);
  animation: pulse-dot 1.5s ease-in-out infinite;
}
@keyframes pulse-dot { 0%,100%{opacity:1} 50%{opacity:0.3} }
@keyframes pulse {
  0%{opacity:0.4;transform:translateY(2px)}
  50%{opacity:1;transform:translateY(0)}
  100%{opacity:0.4;transform:translateY(2px)}
}
@keyframes fadeIn { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:translateY(0)} }
@keyframes slideUp { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:translateY(0)} }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation-duration: 0.01ms !important; } }

.stSelectbox > div > div,
.stTextInput > div > div,
.stMultiselect > div > div {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text) !important;
  font-size: 0.8125rem !important;
  transition: border-color 200ms ease, box-shadow 200ms ease !important;
}
.stSelectbox > div > div:hover,
.stTextInput > div > div:hover,
.stMultiselect > div > div:hover {
  border-color: var(--accent) !important;
}
.stSelectbox > div > div:focus-within,
.stTextInput > div > div:focus-within {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-glow) !important;
}

.stButton button {
  font-family: var(--font);
  font-weight: 600;
  font-size: 0.8125rem;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  padding: 0.5rem 1.25rem;
  background: var(--accent) !important;
  color: #FFFFFF !important;
  cursor: pointer;
  transition: all 200ms ease !important;
}
.stButton button:hover {
  background: var(--accent-hover) !important;
  box-shadow: 0 4px 20px var(--accent-glow);
  transform: translateY(-1px);
}
.stButton button:active { transform: translateY(0); }
.stButton button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

.stTabs [data-baseweb="tab-list"] {
  gap: 0;
  background: var(--bg-card);
  border-radius: var(--radius);
  padding: 3px;
  border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
  font-family: var(--font);
  font-weight: 500;
  font-size: 0.8125rem;
  color: var(--text-secondary) !important;
  border-radius: 7px !important;
  padding: 0.4rem 1rem !important;
  transition: all 200ms ease;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--text) !important; background: rgba(255,255,255,0.03); }
.stTabs [data-baseweb="tab"][aria-selected="true"] {
  background: var(--bg-surface) !important;
  color: var(--accent) !important;
  font-weight: 600;
}

.stRadio label {
  font-family: var(--font);
  font-size: 0.8125rem;
  color: var(--text-secondary);
  padding: 0.2rem 0.5rem;
  border-radius: 4px;
  transition: background 150ms ease;
  cursor: pointer;
}
.stRadio label:hover { background: rgba(255,255,255,0.04); color: var(--text); }
.stRadio input:focus-visible + div { outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 4px; }

.stMetric {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 0.75rem 1rem;
}
.stMetric label {
  font-family: var(--font);
  font-size: 0.6875rem;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-muted) !important;
}
.stMetric [data-testid="stMetricValue"] {
  font-family: var(--font);
  font-size: 1.25rem;
  font-weight: 700;
  color: var(--text) !important;
}

.stAlert { border-radius: var(--radius-sm) !important; font-family: var(--font); border: none !important; }
.stInfo { background: rgba(99,102,241,0.06) !important; border: 1px solid rgba(99,102,241,0.12) !important; border-left: 3px solid var(--accent) !important; color: #A5B4FC !important; }
.stSuccess { background: rgba(34,197,94,0.06) !important; border: 1px solid rgba(34,197,94,0.12) !important; color: #86EFAC !important; }
.stError { background: rgba(239,68,68,0.06) !important; border: 1px solid rgba(239,68,68,0.12) !important; color: #FCA5A5 !important; }
.stWarning { background: rgba(245,158,11,0.06) !important; border: 1px solid rgba(245,158,11,0.12) !important; color: #FCD34D !important; }

div[data-testid="stExpander"] {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  margin-bottom: 0.375rem;
}
div[data-testid="stExpander"] summary {
  font-family: var(--font);
  font-weight: 500;
  font-size: 0.8125rem;
  color: var(--text);
  cursor: pointer;
}
div[data-testid="stExpander"] summary:hover { color: var(--accent); }
div[data-testid="stExpander"] div[role="button"]:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; border-radius: var(--radius-sm); }

.stCaption { color: var(--text-muted); font-size: 0.75rem; }
.stCodeBlock { background: var(--bg-surface) !important; border: 1px solid var(--border); border-radius: var(--radius-sm); }

.result-bubble {
  animation: slideUp 350ms ease both;
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 0.75rem 1rem;
  margin-bottom: 0.5rem;
  transition: border-color 200ms ease;
}
.result-bubble:hover { border-color: var(--border-hover); }
.result-bubble:nth-child(1) { animation-delay: 0ms; }
.result-bubble:nth-child(2) { animation-delay: 60ms; }
.result-bubble:nth-child(3) { animation-delay: 120ms; }
.result-bubble:nth-child(4) { animation-delay: 180ms; }
.result-bubble:nth-child(5) { animation-delay: 240ms; }

.model-card {
  animation: fadeIn 400ms ease both;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1rem;
  transition: border-color 200ms ease, transform 200ms ease;
}
.model-card:hover { border-color: var(--accent); transform: translateY(-2px); }
.model-card:nth-child(1) { animation-delay: 0ms; }
.model-card:nth-child(2) { animation-delay: 80ms; }
.model-card:nth-child(3) { animation-delay: 160ms; }
.model-card:nth-child(4) { animation-delay: 240ms; }

.model-card-best {
  border-color: var(--accent);
  box-shadow: 0 0 20px var(--accent-glow);
}

.metric-value-high { color: var(--green) !important; }
.metric-value-mid { color: var(--amber) !important; }
.metric-value-low { color: var(--red) !important; }
"""

st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

API_URL = os.getenv("RAG_API_URL", "http://localhost:8002")

SAMPLE_QUERIES_PATH = Path(__file__).resolve().parents[2] / "data" / "rag_queries.json"
_sample_data = json.loads(SAMPLE_QUERIES_PATH.read_text(encoding="utf-8"))
QUERY_TO_GT = {q["query"]: q["ground_truth"] for q in _sample_data}
SAMPLE_OPTIONS = ["Custom query"] + [q["query"] for q in _sample_data]

st.markdown(
    "<div style='display:flex;align-items:center;gap:0.75rem;margin-bottom:0.25rem'>"
    "<div style='width:28px;height:28px;background:var(--accent);border-radius:7px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:0.875rem;color:#FFFFFF'>R</div>"
    "<h1 style='margin:0'>RAG Embedding Experiment</h1>"
    "</div>"
    "<p style='margin:0 0 1.5rem 2.35rem;color:var(--text-secondary);font-size:0.875rem'>"
    "Benchmark embedding models across retrieval, generation, and evaluation.</p>",
    unsafe_allow_html=True,
)

try:
    ds_resp = requests.get(f"{API_URL}/datasets", timeout=5)
    available_datasets = ds_resp.json().get("available_datasets", [])
except Exception:
    available_datasets = []

if not available_datasets:
    st.error("Could not load datasets. Is the API running?")
    st.stop()

with st.sidebar:
    st.markdown(
        "<div style='display:flex;align-items:center;gap:8px;margin-bottom:1.25rem'>"
        "<div style='width:8px;height:8px;border-radius:50%;background:var(--accent)'></div>"
        "<span style='font-weight:600;font-size:0.875rem;color:var(--text)'>Control Panel</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='card'>"
        "<div class='card-header'>Dataset</div>",
        unsafe_allow_html=True,
    )
    selected_dataset = st.selectbox("Select dataset:", available_datasets, label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    if selected_dataset:
        try:
            cat_resp = requests.get(f"{API_URL}/datasets/{selected_dataset}", timeout=5)
            cat_resp.raise_for_status()
            current_categories = cat_resp.json().get("categories", [])
        except Exception as e:
            st.error(f"Failed to load categories: {e}")
            current_categories = []

        st.markdown(
            "<div class='card' style='margin-top:0.5rem'>"
            "<div style='display:flex;justify-content:space-between;align-items:center'>"
            "<div class='card-header' style='margin:0;padding:0;border:none'>Categories</div>"
            f"<span style='font-size:0.75rem;color:var(--text-muted)'>{len(current_categories)}</span>"
            "</div>",
            unsafe_allow_html=True,
        )

        if current_categories:
            sel = st.radio(
                "Select to remove:",
                current_categories,
                key="rag_sel",
                label_visibility="collapsed",
            )
            if st.button("Remove selected", key="rag_rem_btn", use_container_width=True):
                updated = [c for c in current_categories if c != sel]
                requests.post(f"{API_URL}/datasets/{selected_dataset}", json={"categories": updated}, timeout=5)
                st.success(f"Removed '{sel}'")
                st.rerun()
        else:
            st.markdown("<div style='color:var(--text-muted);font-size:0.8125rem;padding:0.25rem 0'>No categories yet.</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;color:var(--text-muted);margin-bottom:0.375rem'>Add category</div>", unsafe_allow_html=True)
        new_cat = st.text_input("Category name:", key="rag_add", label_visibility="collapsed")
        if st.button("Add", key="rag_add_btn", use_container_width=True):
            if new_cat and new_cat not in current_categories:
                updated = current_categories + [new_cat]
                requests.post(f"{API_URL}/datasets/{selected_dataset}", json={"categories": updated}, timeout=5)
                st.success(f"Added '{new_cat}'")
                st.rerun()
            elif new_cat in current_categories:
                st.warning("Already exists!")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='card' style='margin-top:0.5rem'>"
        "<div class='card-header'>Available Models</div>",
        unsafe_allow_html=True,
    )
    for mkey, mcfg in EMBEDDING_MODELS.items():
        speed_dot = {"fast": "#22C55E", "medium": "#F59E0B", "slow": "#EF4444"}.get(mcfg.get("speed", ""), "#666")
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:0.5rem;padding:0.25rem 0;font-size:0.8125rem'>"
            f"<div style='width:6px;height:6px;border-radius:50%;background:{speed_dot}'></div>"
            f"<div style='flex:1;color:var(--text)'>{mkey}</div>"
            f"<div style='color:var(--text-muted);font-size:0.6875rem'>{mcfg.get('size','')}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='card' style='margin-top:0.5rem'>"
        "<div class='card-header'>Memory</div>"
        "<div style='font-size:0.8125rem;color:var(--text-secondary);line-height:1.6'>"
        "Models are cached in memory across runs when sufficient RAM is available. "
        "Docker containers with &lt;5&nbsp;GB available fall back to loading per-query.</div>"
        "</div>",
        unsafe_allow_html=True,
    )

_MODEL_NAMES = list(EMBEDDING_MODELS.keys())
_MODEL_OPTIONS = [
    f"{k}  ({EMBEDDING_MODELS[k]['size']}, {EMBEDDING_MODELS[k]['speed']})"
    for k in _MODEL_NAMES
]
_MODEL_MAP = dict(zip(_MODEL_OPTIONS, _MODEL_NAMES))

tab_single, tab_compare, tab_batch = st.tabs([
    "Single Run", "Model Comparison", "Batch Experiment"
])

def _score_color(val: float) -> str:
    if val >= 0.8:
        return "var(--green)"
    if val >= 0.5:
        return "var(--amber)"
    return "var(--red)"

def _render_sse_stage(stage_ph, message: str):
    stage_ph.markdown(
        f"<div style='display:flex;align-items:center;gap:8px'>"
        f"<div class='badge'><span class='badge-dot'></span>{message}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

def _render_retrieved_docs(docs, scores):
    st.markdown(
        "<div class='card' style='margin-top:1.5rem'>"
        "<div class='card-header'>Retrieved Documents</div>",
        unsafe_allow_html=True,
    )
    docs_html = ""
    for idx, (doc, score) in enumerate(zip(docs, scores)):
        docs_html += (
            f"<div class='result-bubble' style='display:flex;align-items:center;gap:0.75rem'>"
            f"<div style='width:22px;height:22px;border-radius:6px;background:var(--bg-card);"
            f"border:1px solid var(--border);display:flex;align-items:center;justify-content:center;"
            f"font-size:0.6875rem;font-weight:600;color:var(--text-muted);flex-shrink:0'>{idx+1}</div>"
            f"<div style='flex:1;font-size:0.875rem;color:var(--text)'>{doc}</div>"
            f"<div style='font-size:0.75rem;font-weight:600;color:var(--accent);font-variant-numeric:tabular-nums'>{score:.4f}</div>"
            f"</div>"
        )
    st.markdown(docs_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

def _render_metrics(ev):
    em = ev["exact_match"]
    rouge = f"{ev['rouge_l_f1']:.3f}"
    sim = f"{ev['semantic_similarity']:.3f}"
    llm = f"{ev['llm_quality_score']:.1f}" if ev["llm_quality_score"] else "N/A"
    html = f"""
    <div class='card' style='margin-top:0.75rem'>
        <div class='card-header'>Evaluation Metrics</div>
        <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:0.5rem;padding:0.75rem 1rem'>
            <div style='text-align:center'>
                <div style='font-size:0.6875rem;font-weight:600;text-transform:uppercase;letter-spacing:0.04em;color:var(--text-muted);margin-bottom:0.25rem'>Exact Match</div>
                <div style='font-size:1.25rem;font-weight:700;color:{"var(--green)" if em else "var(--red)"}'>{'Yes' if em else 'No'}</div>
            </div>
            <div style='text-align:center'>
                <div style='font-size:0.6875rem;font-weight:600;text-transform:uppercase;letter-spacing:0.04em;color:var(--text-muted);margin-bottom:0.25rem'>ROUGE-L F1</div>
                <div style='font-size:1.25rem;font-weight:700;color:var(--text)'>{rouge}</div>
            </div>
            <div style='text-align:center'>
                <div style='font-size:0.6875rem;font-weight:600;text-transform:uppercase;letter-spacing:0.04em;color:var(--text-muted);margin-bottom:0.25rem'>Semantic Similarity</div>
                <div style='font-size:1.25rem;font-weight:700;color:var(--text)'>{sim}</div>
            </div>
            <div style='text-align:center'>
                <div style='font-size:0.6875rem;font-weight:600;text-transform:uppercase;letter-spacing:0.04em;color:var(--text-muted);margin-bottom:0.25rem'>LLM Quality</div>
                <div style='font-size:1.25rem;font-weight:700;color:{"var(--text-muted)" if llm == "N/A" else "var(--text)"}'>{llm}</div>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def _handle_api_error(e, stage_ph=None):
    if stage_ph:
        stage_ph.empty()
    detail = ""
    if hasattr(e, "response") and e.response is not None:
        try:
            detail = e.response.json().get("detail", e.response.text[:500])
        except Exception:
            detail = e.response.text[:500]
    st.error(f"API Error: {e}")
    if detail:
        st.code(detail, language="text")

with tab_single:
    def _apply_sample_single():
        sel = st.session_state["sq_single"]
        if sel != "Custom query":
            st.session_state["query_single"] = sel
            st.session_state["gt_single"] = QUERY_TO_GT.get(sel, "")
    st.selectbox("Sample query:", SAMPLE_OPTIONS, key="sq_single", on_change=_apply_sample_single)
    query = st.text_input("Query:", key="query_single", placeholder="Enter your question...")
    ground_truth = st.text_input("Ground Truth:", key="gt_single", placeholder="Expected answer...")

    col1, col2 = st.columns([1, 1])
    with col1:
        model_label = st.selectbox("Embedding Model:", _MODEL_OPTIONS)
        model = _MODEL_MAP[model_label]
    with col2:
        top_k = st.slider("Top-K Retrieval:", 1, 10, 5)
        run_clicked = st.button("Run RAG", key="single_run", use_container_width=True, type="primary")

    if run_clicked:
        if not query or not ground_truth:
            st.warning("Please enter both a query and ground truth.")
        else:
            result = None
            stage_ph = st.empty()
            try:
                resp = requests.post(
                    f"{API_URL}/run",
                    json={
                        "query": query,
                        "ground_truth": ground_truth,
                        "dataset_name": selected_dataset,
                        "embedding_model": model,
                        "top_k": top_k,
                    },
                    stream=True,
                    timeout=120,
                )
                resp.raise_for_status()

                for raw in resp.iter_lines():
                    if not raw:
                        continue
                    line = raw.decode("utf-8")
                    if not line.startswith("data: "):
                        continue
                    data = json.loads(line[6:])
                    if data["type"] == "stage":
                        _render_sse_stage(stage_ph, data["message"])
                    elif data["type"] == "result":
                        result = data["result"]
                    elif data["type"] == "error":
                        st.error(data["message"])
                        break

                if result is None:
                    if not any(s in str(stage_ph) for s in ["error"]):
                        st.error("No result returned.")
                else:
                    stage_ph.empty()
                    _render_retrieved_docs(result["retrieval"]["documents"], result["retrieval"]["scores"])

                    st.markdown(
                        "<div class='card' style='margin-top:0.75rem'>"
                        "<div class='card-header'>Generated Answer</div>"
                        f"<div style='font-size:0.9375rem;color:var(--text);line-height:1.7'>{result['generation']['answer']}</div>"
                        "</div>",
                        unsafe_allow_html=True,
                    )

                    _render_metrics(result["evaluation"])

            except requests.exceptions.RequestException as e:
                _handle_api_error(e, stage_ph)

with tab_compare:
    def _apply_sample_compare():
        sel = st.session_state["sq_compare"]
        if sel != "Custom query":
            st.session_state["query_compare"] = sel
            st.session_state["gt_compare"] = QUERY_TO_GT.get(sel, "")
    st.selectbox("Sample query:", SAMPLE_OPTIONS, key="sq_compare", on_change=_apply_sample_compare)
    query_c = st.text_input("Query:", key="query_compare", placeholder="Enter your question...")
    gt_c = st.text_input("Ground Truth:", key="gt_compare", placeholder="Expected answer...")

    c1, c2 = st.columns([2, 1])
    with c1:
        compare_models = st.multiselect(
            "Models to include:",
            _MODEL_OPTIONS,
            default=_MODEL_OPTIONS[:4],
            key="compare_models",
        )
        compare_models = [_MODEL_MAP[m] for m in compare_models]
    with c2:
        top_k_c = st.slider("Top-K:", 1, 10, 5, key="c_k")
        run_compare = st.button("Compare Models", key="compare_run", use_container_width=True, type="primary")

    if run_compare:
        if not compare_models:
            st.error("Select at least one model.")
        elif not query_c or not gt_c:
            st.warning("Please enter both a query and ground truth.")
        else:
            stage_ph = st.empty()
            report = None
            try:
                resp = requests.post(
                    f"{API_URL}/compare",
                    json={
                        "query": query_c,
                        "ground_truth": gt_c,
                        "dataset_name": selected_dataset,
                        "embedding_models": compare_models,
                        "top_k": top_k_c,
                    },
                    stream=True,
                    timeout=600,
                )
                resp.raise_for_status()

                for raw in resp.iter_lines():
                    if not raw:
                        continue
                    line = raw.decode("utf-8")
                    if not line.startswith("data: "):
                        continue
                    data = json.loads(line[6:])
                    if data["type"] == "stage":
                        _render_sse_stage(stage_ph, data["message"])
                    elif data["type"] == "result":
                        report = data["result"]
                    elif data["type"] == "error":
                        st.error(data["message"])
                        break

                if report is None:
                    st.error("No result returned.")
                else:
                    stage_ph.empty()
                    best = report.get("best_model")

                    if best:
                        st.success(f"Best model overall: **{best}**")
                    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

                    models_list = list(report["results"].items())
                    n_cols = min(len(models_list), 4)
                    cols = st.columns(n_cols)
                    for idx, (model_name, result) in enumerate(models_list):
                        col = cols[idx % n_cols]
                        is_best = best == model_name
                        with col:
                            html = (
                                f"<div class='model-card{' model-card-best' if is_best else ''}' style='animation-delay:{idx*80}ms'>"
                                f"<div style='display:flex;align-items:center;gap:0.5rem;margin-bottom:0.75rem'>"
                                f"<div style='width:8px;height:8px;border-radius:50%;background:"
                                f"{'var(--accent)' if is_best else 'var(--text-muted)'}'></div>"
                                f"<span style='font-weight:600;font-size:0.875rem;color:var(--text)'>{model_name}</span>"
                                f"{'<span style=\"font-size:0.625rem;font-weight:600;text-transform:uppercase;letter-spacing:0.04em;padding:2px 6px;border-radius:4px;background:var(--accent-subtle);color:#A5B4FC;margin-left:auto\">Best</span>' if is_best else ''}"
                                f"</div>"
                            )
                            if "error" in result:
                                html += f"<div style='color:var(--red);font-size:0.8125rem'>{result['error']}</div>"
                                st.markdown(html + "</div>", unsafe_allow_html=True)
                                continue

                            ev = result["evaluation"]
                            qscore = f"{ev['llm_quality_score']:.1f}" if ev["llm_quality_score"] else "N/A"

                            docs = result["retrieval"]["documents"]
                            scores = result["retrieval"]["scores"]
                            docs_html = ""
                            for di, (doc, sc) in enumerate(zip(docs, scores)):
                                docs_html += (
                                    f"<div style='display:flex;align-items:center;gap:0.5rem;padding:0.375rem 0;"
                                    f"border-bottom:1px solid var(--border)'>"
                                    f"<div style='width:18px;height:18px;border-radius:4px;background:var(--bg-card);"
                                    f"border:1px solid var(--border);display:flex;align-items:center;justify-content:center;"
                                    f"font-size:0.625rem;font-weight:600;color:var(--text-muted);flex-shrink:0'>{di+1}</div>"
                                    f"<div style='flex:1;font-size:0.75rem;color:var(--text-secondary);line-height:1.4'>{doc}</div>"
                                    f"<div style='font-size:0.6875rem;font-weight:600;color:var(--accent);font-variant-numeric:tabular-nums'>{sc:.4f}</div>"
                                    f"</div>"
                                )

                            html += (
                                f"<div style='display:flex;flex-direction:column;gap:0.375rem'>"
                                f"<div style='font-size:0.8125rem;color:var(--text-secondary)'>"
                                f"<span style='color:var(--text-muted)'>Answer:</span> "
                                f"<span style='color:var(--text)'>{result['generation']['answer']}</span>"
                                f"</div>"
                                f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:0.375rem'>"
                                f"<div style='background:var(--bg-surface);border-radius:6px;padding:0.5rem;text-align:center'>"
                                f"<div style='font-size:0.625rem;text-transform:uppercase;letter-spacing:0.04em;color:var(--text-muted)'>EM</div>"
                                f"<div style='font-size:1rem;font-weight:700;color:var(--text)'>{'Yes' if ev['exact_match'] else 'No'}</div></div>"
                                f"<div style='background:var(--bg-surface);border-radius:6px;padding:0.5rem;text-align:center'>"
                                f"<div style='font-size:0.625rem;text-transform:uppercase;letter-spacing:0.04em;color:var(--text-muted)'>RougeL</div>"
                                f"<div style='font-size:1rem;font-weight:700;color:{_score_color(ev['rouge_l_f1'])}'>{ev['rouge_l_f1']:.3f}</div></div>"
                                f"<div style='background:var(--bg-surface);border-radius:6px;padding:0.5rem;text-align:center'>"
                                f"<div style='font-size:0.625rem;text-transform:uppercase;letter-spacing:0.04em;color:var(--text-muted)'>SemSim</div>"
                                f"<div style='font-size:1rem;font-weight:700;color:{_score_color(ev['semantic_similarity'])}'>{ev['semantic_similarity']:.3f}</div></div>"
                                f"<div style='background:var(--bg-surface);border-radius:6px;padding:0.5rem;text-align:center'>"
                                f"<div style='font-size:0.625rem;text-transform:uppercase;letter-spacing:0.04em;color:var(--text-muted)'>Quality</div>"
                                f"<div style='font-size:1rem;font-weight:700;color:var(--text)'>{qscore}</div></div>"
                                f"</div>"
                                f"<details style='margin-top:0.25rem'>"
                                f"<summary style='font-size:0.75rem;font-weight:600;cursor:pointer;color:var(--accent);"
                                f"padding:0.375rem 0;user-select:none'>"
                                f"Retrieved Documents ({len(docs)})</summary>"
                                f"<div style='margin-top:0.375rem'>{docs_html}</div>"
                                f"</details>"
                                f"</div></div>"
                            )
                            st.markdown(html, unsafe_allow_html=True)

            except requests.exceptions.RequestException as e:
                _handle_api_error(e, stage_ph)

with tab_batch:
    st.markdown(
        "<div class='card'>"
        "<div style='font-size:0.875rem;color:var(--text-secondary);line-height:1.7'>"
        "Run all 20 evaluation queries across selected embedding models. "
        "Results include exact match, ROUGE-L, semantic similarity, and overall model ranking.</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='margin-top:0.75rem;display:flex;align-items:center;gap:0.75rem;padding:0.625rem 0.875rem;"
        "background:var(--bg-surface);border-radius:var(--radius-sm);font-size:0.8125rem;color:var(--text-secondary)'>"
        "<span style='font-size:1rem'>⚡</span>"
        "Models are cached in memory. First run loads models (~8s each), subsequent queries reuse them.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    selected_models = st.multiselect(
        "Models to include:",
        _MODEL_OPTIONS,
        default=_MODEL_OPTIONS[:4],
        key="batch_models",
    )
    selected_models = [_MODEL_MAP[m] for m in selected_models]

    if st.button("Start Batch Experiment", key="batch_run", use_container_width=True, type="primary"):
        if not selected_models:
            st.error("Select at least one model.")
        else:
            with st.spinner(f"Evaluating {len(selected_models)} model(s) across 20 queries..."):
                try:
                    resp = requests.post(
                        f"{API_URL}/run-batch",
                        json={"embedding_models": selected_models},
                        timeout=3600,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    reports = data["reports"]

                    win_counts = {}
                    model_metrics = {}
                    for rep in reports:
                        best = rep.get("best_model")
                        if best:
                            win_counts[best] = win_counts.get(best, 0) + 1
                        for m_name, m_result in rep["results"].items():
                            if "error" in m_result:
                                continue
                            ev = m_result["evaluation"]
                            if m_name not in model_metrics:
                                model_metrics[m_name] = {"em": 0, "rouge": [], "semsim": [], "count": 0}
                            model_metrics[m_name]["em"] += 1 if ev["exact_match"] else 0
                            model_metrics[m_name]["rouge"].append(ev["rouge_l_f1"])
                            model_metrics[m_name]["semsim"].append(ev["semantic_similarity"])
                            model_metrics[m_name]["count"] += 1

                    for m in model_metrics:
                        mm = model_metrics[m]
                        mm["avg_rouge"] = sum(mm["rouge"]) / len(mm["rouge"]) if mm["rouge"] else 0
                        mm["avg_semsim"] = sum(mm["semsim"]) / len(mm["semsim"]) if mm["semsim"] else 0

                    st.markdown(
                        "<div class='card' style='margin-top:1rem'>"
                        "<div class='card-header'>Model Leaderboard</div>",
                        unsafe_allow_html=True,
                    )

                    sorted_models = sorted(win_counts.items(), key=lambda x: x[1], reverse=True)
                    leaderboard_rows = []
                    for i, (model, wins) in enumerate(sorted_models):
                        rank_icon = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1}"
                        mm = model_metrics.get(model, {})
                        leaderboard_rows.append({
                            "model": model,
                            "rank": rank_icon,
                            "wins": f"{wins} / {data['total_queries']}",
                            "em": f"{mm.get('em', 0)}/{mm.get('count', 1)}",
                            "avg_rouge": f"{mm.get('avg_rouge', 0):.3f}",
                            "avg_semsim": f"{mm.get('avg_semsim', 0):.3f}",
                        })

                    import pandas as pd
                    if leaderboard_rows:
                        df = pd.DataFrame(leaderboard_rows)
                        st.dataframe(
                            df,
                            column_config={
                                "rank": st.column_config.TextColumn("Rank", width="small"),
                                "model": st.column_config.TextColumn("Model", width="medium"),
                                "wins": st.column_config.TextColumn("Wins", width="small"),
                                "em": st.column_config.TextColumn("EM", width="small"),
                                "avg_rouge": st.column_config.TextColumn("Avg RougeL", width="small"),
                                "avg_semsim": st.column_config.TextColumn("Avg SemSim", width="small"),
                            },
                            hide_index=True,
                            use_container_width=True,
                        )

                    if leaderboard_rows:
                        chart_df = pd.DataFrame([
                            {"model": r["model"], "wins": int(r["wins"].split("/")[0])}
                            for r in leaderboard_rows
                        ])
                        st.bar_chart(chart_df, x="model", y="wins", height=300)

                    st.markdown("</div>", unsafe_allow_html=True)

                    st.markdown(
                        "<div class='card' style='margin-top:0.75rem'>"
                        "<div class='card-header'>Detailed Results</div>",
                        unsafe_allow_html=True,
                    )
                    for rep in reports:
                        with st.expander(f"Query: {rep['query'][:60]}..."):
                            st.markdown(
                                f"<div style='font-size:0.8125rem;color:var(--text-secondary)'>"
                                f"<span style='color:var(--text-muted)'>Ground Truth:</span> {rep['ground_truth']}</div>",
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f"<div style='font-size:0.8125rem;color:var(--text-secondary);margin-bottom:0.5rem'>"
                                f"<span style='color:var(--text-muted)'>Best Model:</span> "
                                f"<span style='color:var(--accent);font-weight:600'>{rep['best_model']}</span></div>",
                                unsafe_allow_html=True,
                            )
                            for m_name, m_result in rep["results"].items():
                                if "error" in m_result:
                                    st.markdown(
                                        f"<div style='font-size:0.8125rem;color:var(--red);padding:0.25rem 0'>- {m_name}: ERROR - {m_result['error']}</div>",
                                        unsafe_allow_html=True,
                                    )
                                    continue
                                ev = m_result["evaluation"]
                                st.markdown(
                                    f"<div style='display:flex;align-items:center;gap:0.5rem;font-size:0.8125rem;"
                                    f"color:var(--text-secondary);padding:0.25rem 0'>"
                                    f"<span style='color:var(--text)'>{m_name}</span>: "
                                    f"EM=<span style='color:{_score_color(1.0 if ev['exact_match'] else 0.0)}'>{ev['exact_match']}</span>, "
                                    f"RougeL=<span style='color:{_score_color(ev['rouge_l_f1'])}'>{ev['rouge_l_f1']:.3f}</span>, "
                                    f"SemSim=<span style='color:{_score_color(ev['semantic_similarity'])}'>{ev['semantic_similarity']:.3f}</span>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )
                    st.markdown("</div>", unsafe_allow_html=True)

                except requests.exceptions.RequestException as e:
                    _handle_api_error(e)

st.caption(
    "Start API: python -m uvicorn src.api.rag_api:app --port 8002"
)
