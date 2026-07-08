import streamlit as st

from src.rag.config import EMBEDDING_MODELS
from src.ui.mcp_client import list_datasets, list_queries, submit_rag_job, wait_for_result, get_dataset, create_dataset, delete_dataset

st.set_page_config(page_title="RAG Benchmark", layout="wide", initial_sidebar_state="expanded")

st.session_state.setdefault("query_input", "")
st.session_state.setdefault("gt_input", "")
st.session_state.setdefault("results_data", None)

T = {
    "bg": "#1b2039", "surface": "#23284A", "card": "#1b2039",
    "border": "#333D6D", "border_h": "#434D7D",
    "text": "#FFCF95", "text_sec": "#A89BC8", "text_faint": "#636992",
    "accent": "#723EC3", "accent_h": "#8550D6",
    "green": "#50D68A", "amber": "#FFCF95", "red": "#D65050",
    "metric_bg": "#23284A",
    "glow": "0 1px 8px rgba(114,62,195,0.15), 0 1px 3px rgba(0,0,0,0.2)",
}

MODEL_KEYS = list(EMBEDDING_MODELS.keys())

_datasets = list_datasets()
_queries = list_queries()
_query_gt_map = {q["query"]: q for q in _queries}

if not _datasets or not _queries:
    st.error("Could not connect to MCP server")
    st.stop()

# ── sidebar: edit existing dataset ──
with st.sidebar:
    st.markdown(f"<h3 style='color:{T['text_faint']};font-size:0.75rem;text-transform:uppercase;letter-spacing:0.04em'>Edit Dataset</h3>", unsafe_allow_html=True)
    sel = st.selectbox("Select", _datasets, label_visibility="collapsed", key="ds_sel")
    if sel:
        ds_info = get_dataset(sel)
        docs_text = "\n".join(ds_info.get("documents", []))
        new_docs = st.text_area("Documents (one per line)", value=docs_text, height=180, key="ds_docs", label_visibility="collapsed")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Save", type="primary", use_container_width=True):
                create_dataset(sel, [d.strip() for d in new_docs.split("\n") if d.strip()])
                st.rerun()
        with c2:
            if st.button("Delete", use_container_width=True):
                delete_dataset(sel)
                st.rerun()

CSS = f"""
#MainMenu, header, footer, .stAppDeployButton, div[data-testid="stToolbar"] {{ display: none !important }}
.appview-container .main .block-container {{ max-width: 920px !important; padding: 0 !important }}
html, body, [data-testid="stAppViewContainer"] {{ background: {T['bg']} !important; font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Helvetica Neue', system-ui, sans-serif; -webkit-font-smoothing: antialiased }}
h1 {{ font-size: 1.15rem !important; font-weight: 600 !important; letter-spacing: -0.015em !important; color: {T['text']} !important }}
h2 {{ font-size: 1.05rem !important; font-weight: 600 !important; letter-spacing: -0.01em !important; color: {T['text']} !important }}
h3 {{ font-size: 0.7rem !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.04em; color: {T['text_faint']} !important; margin-bottom: 0.5rem !important }}
p, li, .stMarkdown {{ color: {T['text_sec']}; font-size: 0.82rem; line-height: 1.5 }}
.card {{ background: {T['card']}; border: 1px solid {T['border']}; border-radius: 12px; padding: 1.25rem 1.5rem; box-shadow: {T['glow']} }}
::-webkit-scrollbar {{ width: 4px; height: 4px }}
::-webkit-scrollbar-track {{ background: transparent }}
::-webkit-scrollbar-thumb {{ background: {T['border']}; border-radius: 4px }}
.stSelectbox > div > div, .stTextInput > div > div, .stMultiselect > div > div {{ background: {T['surface']} !important; border: 1px solid {T['border']} !important; border-radius: 10px !important; font-size: 0.82rem !important; color: {T['text']} !important; min-height: 38px !important; caret-color: {T['accent']}; animation: fi 0.3s ease both }}
.stSelectbox > div > div:focus-within, .stTextInput > div > div:focus-within {{ border-color: {T['accent']} !important; box-shadow: 0 0 0 3px {T['accent']}22 !important }}
.stButton > button {{ font-weight: 500; font-size: 0.82rem; border-radius: 10px !important; padding: 0.35rem 1.25rem !important; transition: all 200ms cubic-bezier(0.25,0.46,0.45,0.94) !important; letter-spacing: -0.01em; height: 38px; line-height: 1; border: 1px solid {T['border']} !important }}
.stButton > button[kind="primary"] {{ background: {T['accent']} !important; color: #fff !important; border: none !important }}
.stButton > button[kind="primary"]:hover {{ background: {T['accent_h']} !important; box-shadow: 0 4px 16px {T['accent']}33 !important; transform: translateY(-1px) !important }}
.stButton > button[kind="primary"]:active {{ transform: translateY(0) !important }}
.stButton > button[kind="secondary"] {{ background: {T['surface']} !important; color: {T['text']} !important }}
.stButton > button[kind="secondary"]:hover {{ border-color: {T['border_h']} !important; transform: translateY(-1px) !important }}
.stMultiSelect [data-baseweb="tag"] {{ background: {T['metric_bg']} !important; color: {T['text_sec']} !important; border-radius: 8px !important }}
.stMultiSelect [data-baseweb="tag"] svg {{ fill: {T['text_sec']} !important }}
.stAlert {{ border-radius: 10px !important; border: none !important; font-size: 0.82rem; padding: 0.6rem 1rem !important }}
.stProgress > div > div > div {{ background: {T['accent']} !important }}

@keyframes fi {{ from{{opacity:0}}to{{opacity:1}} }}
@keyframes fu {{ from{{opacity:0;transform:translateY(14px)}}to{{opacity:1;transform:translateY(0)}} }}
@keyframes si {{ from{{opacity:0;transform:scale(0.92)}}to{{opacity:1;transform:scale(1)}} }}
@keyframes float {{ 0%,100%{{transform:translateY(0)}}50%{{transform:translateY(-8px)}} }}
.anm-fu {{ animation: fu 0.5s cubic-bezier(0.16,1,0.3,1) both }}
.anm-si {{ animation: si 0.35s cubic-bezier(0.16,1,0.3,1) both }}

/* fixed illustration behind everything */
.hero-bg {{ position:fixed; top:0; left:0; width:100%; height:100vh; z-index:-1; display:flex; flex-direction:column; align-items:center; justify-content:center; pointer-events:none; background:{T['bg']} }}
.hero-bg .vis {{ position:relative; width:200px; height:200px; margin-bottom:2rem; animation:float 4s ease-in-out infinite }}
.hero-bg .vis .c1 {{ position:absolute; inset:0; border-radius:50%; background:{T['surface']}; border:1px solid {T['border']} }}
.hero-bg .vis .c2 {{ position:absolute; top:25px; left:25px; right:25px; bottom:25px; border-radius:50%; background:{T['surface']}; border:1px solid {T['border']} }}
.hero-bg .vis .c3 {{ position:absolute; top:55px; left:55px; right:55px; bottom:55px; border-radius:50%; background:{T['accent']}22; border:1px solid {T['accent']}44; display:flex; align-items:center; justify-content:center }}
.hero-bg .vis .c3::after {{ content:''; width:16px; height:16px; border-radius:4px; background:{T['accent']}; transform:rotate(45deg) }}
.hero-bg h1 {{ font-size:2.4rem !important; font-weight:700 !important; letter-spacing:-0.03em !important; line-height:1.1 !important; margin:0 0 0.5rem !important; color:{T['text']} !important }}
.hero-bg p {{ font-size:0.95rem; color:{T['text_faint']}; margin:0; max-width:400px; text-align:center; line-height:1.5 }}

/* gradient spacer — fades the illustration out as you scroll */
.fade-spacer {{ height:50vh; background:linear-gradient(to bottom, transparent, {T['bg']}); }}

/* expandable result card */
.rwrap {{ margin-bottom:0.6rem;animation:fu 0.5s cubic-bezier(0.16,1,0.3,1) both }}
.rsum {{ list-style:none;display:flex;align-items:center;gap:0.5rem;cursor:pointer;padding:0.75rem 1rem;background:{T['card']};border:1px solid {T['border']};border-radius:10px;transition:all 150ms;user-select:none }}
.rsum:hover {{ border-color:{T['border_h']} }}
.rwrap[open] .rsum {{ border-radius:10px 10px 0 0;border-bottom-color:{T['surface']} }}
.rsum::-webkit-details-marker {{ display:none }}
.rsum::before {{ content:'▸';font-size:0.7rem;color:{T['text_faint']};transition:transform 200ms;margin-right:0.2rem }}
.rwrap[open] .rsum::before {{ transform:rotate(90deg) }}
.rdetail {{ padding:0.75rem 1rem 1rem;background:{T['card']};border:1px solid {T['border']};border-top:none;border-radius:0 0 10px 10px }}

.mr {{ display:flex;gap:0.4rem;flex-wrap:wrap;margin-bottom:0.5rem }}
.mc {{ background:{T['metric_bg']};border-radius:8px;padding:0.45rem 0.65rem;min-width:60px;flex:1;text-align:center }}
.mc-l {{ font-size:0.55rem;font-weight:600;text-transform:uppercase;letter-spacing:0.04em;color:{T['text_faint']};margin-bottom:0.1rem }}
.mc-v {{ font-size:0.9rem;font-weight:600;font-variant-numeric:tabular-nums }}
.dr {{ display:flex;align-items:center;gap:0.7rem;padding:0.3rem 0 }}
.dr+.dr {{ border-top:1px solid {T['border']} }}
.di {{ width:20px;height:20px;border-radius:6px;background:{T['surface']};display:flex;align-items:center;justify-content:center;font-size:0.6rem;font-weight:600;color:{T['text_faint']};flex-shrink:0 }}
.dt {{ flex:1;font-size:0.78rem;color:{T['text']};overflow:hidden;text-overflow:ellipsis;white-space:nowrap }}
.ds {{ font-size:0.68rem;font-weight:500;color:{T['text_sec']};font-variant-numeric:tabular-nums }}
.ab {{ background:{T['surface']};border-radius:8px;padding:0.75rem;font-size:0.82rem;color:{T['text']};line-height:1.6 }}
.bg {{ display:inline-flex;align-items:center;gap:4px;padding:1px 8px;border-radius:20px;font-size:0.6rem;font-weight:600;text-transform:uppercase;letter-spacing:0.04em;background:{T['surface']};color:{T['text_sec']};border:1px solid {T['border']} }}
.bg-b {{ background:{T['accent']}15;color:{T['accent']};border-color:{T['accent']}30 }}
.dset-tag {{ display:inline-block;font-size:0.6rem;font-weight:500;padding:1px 7px;border-radius:6px;background:{T['metric_bg']};color:{T['text_faint']};border:1px solid {T['border']};margin-left:6px }}
"""

st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

# ── fixed background illustration ──
st.markdown(
    f"<div class='hero-bg'>"
    f"<div class='vis'><div class='c1'></div><div class='c2'></div><div class='c3'></div></div>"
    f"<h1>RAG Benchmark</h1>"
    f"<p>{len(MODEL_KEYS)} models &middot; {len(_datasets)} datasets &middot; {len(_queries)} queries</p>"
    f"</div>",
    unsafe_allow_html=True,
)

# ── gradient spacer — transparent at top (shows illustration), solid at bottom (covers it) ──
st.markdown("<div class='fade-spacer'></div>", unsafe_allow_html=True)

# ── content ──
st.markdown(
    f"<div style='font-size:0.82rem;font-weight:600;letter-spacing:-0.02em;color:{T['text']};padding:0.75rem 0 0.25rem;animation:fu 0.5s 0.1s both'>Benchmark</div>"
    f"<div style='font-size:0.68rem;color:{T['text_faint']};margin-top:-4px;padding-bottom:0.5rem;animation:fu 0.5s 0.15s both'>{len(MODEL_KEYS)} models &middot; {len(_datasets)} datasets &middot; {len(_queries)} queries</div>",
    unsafe_allow_html=True,
)

st.markdown("<div class='card anm-fu'>", unsafe_allow_html=True)

c1, c2 = st.columns([2, 1])
with c2: top_k = st.selectbox("Top-K", [3, 5, 10, 20], index=1, label_visibility="collapsed")
with c1:
    selected_datasets = st.multiselect(
        "Datasets", _datasets, default=_datasets[:2] if len(_datasets) >= 2 else _datasets,
        placeholder="Select datasets", label_visibility="collapsed",
    )

st.markdown(f"<div style='color:{T['text_faint']};font-size:0.68rem;font-weight:500;margin:0.7rem 0 0.35rem'>Samples</div>", unsafe_allow_html=True)
cols = st.columns(3)
for i, q in enumerate(_queries[:9]):
    dset = q.get("relevant_dataset", "")
    with cols[i % 3]:
        if st.button(q["query"][:38], key=f"s{i}", use_container_width=True, type="secondary"):
            qdata = _query_gt_map.get(q["query"])
            st.session_state.query_input = q["query"]
            st.session_state.gt_input = qdata["ground_truth"] if qdata else ""
        st.markdown(f"<div style='font-size:0.6rem;color:{T['text_faint']};margin-top:-4px;margin-bottom:2px;padding-left:2px'><span class='dset-tag'>{dset}</span></div>", unsafe_allow_html=True)

qt = st.text_input("Query", value=st.session_state.query_input, placeholder="Type your question or pick a sample...", label_visibility="collapsed")
gt = st.text_input("Ground truth", value=st.session_state.gt_input, placeholder="Expected answer (optional)", label_visibility="collapsed")
sm = st.multiselect("Models", MODEL_KEYS, default=MODEL_KEYS[:3], placeholder="Select models", label_visibility="collapsed")

br, bc = st.columns([1, 5])
with br:
    run = st.button("Evaluate", type="primary", use_container_width=True)
with bc:
    if st.button("Clear", use_container_width=True):
        for k in ["query_input", "gt_input"]:
            st.session_state[k] = ""
        st.session_state.results_data = None
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

# ── run ──
if run:
    q = qt.strip()
    if not q or not sm or not selected_datasets:
        msgs = []
        if not q: msgs.append("Enter a query")
        if not sm: msgs.append("Select at least one model")
        if not selected_datasets: msgs.append("Select at least one dataset")
        st.warning(" and ".join(msgs))
        st.stop()

    bar = st.progress(0, text="Starting\u2026")
    total = len(sm) * len(selected_datasets)
    done = 0
    results_by_ds = {ds: [] for ds in selected_datasets}

    for di, ds_name in enumerate(selected_datasets):
        for mi, m in enumerate(sm):
            bar.progress(done / total, text=f"{ds_name} \u00b7 {m} \u2026")
            r = submit_rag_job(q, m, ds_name, gt)
            j = r.get("job_id")
            if j:
                f = wait_for_result(j)
                if f and f.get("results"):
                    r0 = f["results"][0]
                    if "error" not in r0:
                        results_by_ds[ds_name].append(r0)
            done += 1
    bar.empty()

    all_flat = [r for rs in results_by_ds.values() for r in rs]
    if not all_flat:
        st.error("No results \u2014 check MCP logs")
        st.stop()

    st.session_state.results_data = (results_by_ds, selected_datasets, top_k, q)

# ── render results ──
if st.session_state.results_data:
    results_by_ds, selected_datasets, tk, query_str = st.session_state.results_data
    all_flat = [r for rs in results_by_ds.values() for r in rs]

    best = max(all_flat, key=lambda r: (
        r.get("evaluation", {}).get("hit_rate", 0) * 50
        + r.get("evaluation", {}).get("rouge_l_f1", 0) * 25
        + r.get("evaluation", {}).get("semantic_similarity", 0) * 15
        + (r.get("evaluation", {}).get("llm_quality_score", 0) or 0) / 5 * 10
    ))
    bn = best.get("retrieval", {}).get("model_name", "?")

    st.markdown(f"<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='anm-fu' style='display:flex;align-items:center;gap:0.75rem;margin-bottom:1rem'>"
        f"<h2 style='margin:0'>Results</h2>"
        f"<span class='bg anm-si' style='animation-delay:0.1s'>top-{tk}</span>"
        f"<span class='bg bg-b anm-si' style='animation-delay:0.12s'>Best: {bn}</span>"
        f"<span style='font-size:0.75rem;color:{T['text_faint']}'>{query_str[:55]}</span></div>",
        unsafe_allow_html=True,
    )

    overall_ri = 0
    for di, ds_name in enumerate(selected_datasets):
        ds_results = results_by_ds.get(ds_name, [])
        if not ds_results:
            continue

        if di > 0:
            st.markdown(f"<div style='height:1.2rem'></div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='anm-fu' style='display:flex;align-items:center;gap:0.6rem;margin-bottom:0.6rem'>"
            f"<span style='font-weight:600;font-size:0.88rem;color:{T['text']}'>{ds_name}</span>"
            f"<span class='bg' style='font-size:0.55rem'>{len(ds_results)} models</span></div>",
            unsafe_allow_html=True,
        )

        for ri, result in enumerate(ds_results):
            ev = result.get("evaluation", {})
            ret = result.get("retrieval", {})
            gen = result.get("generation", {})
            ib = result is best
            delay = overall_ri * 0.06
            mn = ret.get("model_name", "?")

            def cv(v):
                if v is None: return T['text_faint']
                if v >= 0.8: return T['green']
                if v >= 0.5: return T['amber']
                return T['red']

            hr = ev.get("hit_rate", 0)
            em = ev.get("exact_match", False)
            ans = gen.get("answer", "") or "\u2014"
            bdg = '<span class="bg bg-b" style="font-size:0.55rem">Best</span>' if ib else ""

            rm = "".join(
                f"<div class='mc'><div class='mc-l'>{l}</div><div class='mc-v' style='color:{cv(ev.get(k,0))}'>{ev.get(k,0):.3f}</div></div>"
                for l, k in [("Hit Rate","hit_rate"),("MRR","mrr"),("Precision","prec"),("NDCG","ndcg")]
            )
            em_icon = f"<div class='mc'><div class='mc-l'>Exact</div><div class='mc-v' style='color:{T['green'] if em else T['red']}'>{'✓' if em else '✗'}</div></div>"
            gm = em_icon
            for l, k, ff in [("ROUGE-L","rouge_l_f1",".3f"),("Semantic","semantic_similarity",".3f"),
                             ("Faithful","faithfulness",".3f"),("Relevancy","answer_relevancy",".3f"),
                             ("LLM","llm_quality_score",".1f")]:
                v = ev.get(k)
                gm += f"<div class='mc'><div class='mc-l'>{l}</div><div class='mc-v' style='color:{cv(v)}'>" + ("\u2014" if v is None else f"{v:{ff}}") + "</div></div>"

            docs = ret.get("documents", [])[:tk]
            scores = ret.get("scores", [])[:tk]
            dh = "".join(
                f"<div class='dr'><div class='di'>{i+1}</div><div class='dt'>{d}</div><div class='ds'>{s:.4f}</div></div>"
                for i, (d, s) in enumerate(zip(docs, scores))
            ) if docs else f"<div style='color:{T['text_faint']};font-size:0.78rem'>\u2014</div>"

            ans_fallback = f"<span style=color:{T['text_faint']}>\u2014</span>"

            st.markdown(
                f"<details class='rwrap' style='animation-delay:{delay}s'>"
                f"<summary class='rsum'>"
                f"<span style='font-weight:600;font-size:0.85rem;color:{T['text']};min-width:120px'>{mn}</span>{bdg}"
                f"<span style='margin-left:auto;font-size:0.78rem;color:{cv(hr)};font-weight:600;font-variant-numeric:tabular-nums'>HR: {hr:.3f}</span>"
                f"<span style='font-size:0.78rem;color:{T['green'] if em else T['red']};font-weight:600'>EM: {'✓' if em else '✗'}</span>"
                f"</summary>"
                f"<div class='rdetail'>"
                f"<h3>Retrieval</h3><div class='mr'>{rm}</div>"
                f"<div style='height:0.6rem'></div>"
                f"<h3>Generation</h3><div class='mr'>{gm}</div>"
                f"<div style='height:0.6rem'></div>"
                f"<h3>Documents ({len(docs)})</h3>{dh}"
                f"<div style='height:0.6rem'></div>"
                f"<h3>Answer</h3><div class='ab'>{ans if ans else ans_fallback}</div>"
                f"</div></details>",
                unsafe_allow_html=True,
            )
            overall_ri += 1

st.markdown(
    f"<div style='text-align:center;padding:1.5rem 0 0.5rem;font-size:0.68rem;color:{T['text_faint']}'>"
    "retrieval-arena</div>",
    unsafe_allow_html=True,
)
