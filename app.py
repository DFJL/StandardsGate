"""
Standards Gate — Component 2: AI Copilot UI

A Streamlit review tool for CDISC SDTM/ADaM mapping recommendations derived
from Statistical Analysis Plans. This is an AI copilot, not an auto-generator.
All findings are presented as recommendations that require human review.
"""

import copy
import json
import os
import sys
import traceback
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

# ---------------------------------------------------------------------------
# Path setup — ensure pipeline/ is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Page configuration (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Standards Gate — CDISC Mapping Copilot",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

@st.cache_resource
def load_config() -> dict:
    """Load config.yaml once and cache for the session."""
    config_path = _PROJECT_ROOT / "config.yaml"
    with open(config_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


CONFIG = load_config()

# ---------------------------------------------------------------------------
# Download knowledge base from Google Drive on cold start
# ---------------------------------------------------------------------------
from pipeline import gdrive_sync  # noqa: E402
if gdrive_sync.is_configured():
    _kb_path = Path(CONFIG["storage"]["base_path"])
    gdrive_sync.download_knowledge_base(_kb_path)

# ---------------------------------------------------------------------------
# Helpers imported from app_utils
# ---------------------------------------------------------------------------
from app_utils import (
    get_confidence_color,
    get_confidence_label,
    lookup_from_index,
    process_uploaded_pdf,
    schema_to_adam_table,
    schema_to_export_csv,
    schema_to_open_questions_df,
    schema_to_sdtm_table,
)

# ---------------------------------------------------------------------------
# Severity badge HTML helpers
# ---------------------------------------------------------------------------

_SEVERITY_COLORS = {
    "HIGH":   ("#C62828", "#FFEBEE"),
    "MEDIUM": ("#E65100", "#FFF3E0"),
    "LOW":    ("#616161", "#F5F5F5"),
}

_CONFIDENCE_BADGE_COLORS = {
    "HIGH":   ("#2E7D32", "#E8F5E9"),
    "MEDIUM": ("#F57F17", "#FFFDE7"),
    "LOW":    ("#C62828", "#FFEBEE"),
}


def _severity_badge(severity: str) -> str:
    fg, bg = _SEVERITY_COLORS.get(severity.upper(), ("#616161", "#F5F5F5"))
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 10px;'
        f'border-radius:12px;font-size:0.78rem;font-weight:700;'
        f'border:1px solid {fg};">{severity.upper()}</span>'
    )


def _confidence_badge(label: str) -> str:
    """Badge for overall extraction confidence (HIGH/MEDIUM/LOW string)."""
    fg, bg = _CONFIDENCE_BADGE_COLORS.get(label.upper(), ("#616161", "#F5F5F5"))
    return (
        f'<span style="background:{bg};color:{fg};padding:4px 14px;'
        f'border-radius:14px;font-size:0.9rem;font-weight:700;'
        f'border:1px solid {fg};">Extraction Confidence: {label.upper()}</span>'
    )


def _adam_score_badge(score) -> str:
    """Colored pill for an ADaM confidence score."""
    color = get_confidence_color(score)
    label = get_confidence_label(score)
    return (
        f'<span style="background:{color};color:#FFF;padding:2px 10px;'
        f'border-radius:12px;font-size:0.78rem;font-weight:700;">{label}</span>'
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar():
    with st.sidebar:
        st.markdown("### 🔬 Standards Gate")
        st.caption("CDISC Mapping Copilot · v1.0")
        st.divider()

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key:
            st.success("API key detected", icon="✅")
        else:
            st.warning(
                "ANTHROPIC_API_KEY not set. PDF upload requires this key.",
                icon="⚠️",
            )

        st.divider()
        st.markdown(
            """
**Standards References**
- SDTM IG v3.4
- ADaM IG v2.1
- CDISC Library

**Disclaimer**
All outputs are *recommendations* and require validation by a qualified CDISC
programmer or biostatistician before use in study setup.
            """
        )
        st.divider()
        st.page_link("pages/about.py", label="About & Methodology", icon="ℹ️")


# ---------------------------------------------------------------------------
# Tab: Input
# ---------------------------------------------------------------------------

def _tab_input():
    st.header("SAP Input")
    st.markdown(
        "Upload a Statistical Analysis Plan PDF **or** look up a study by NCT ID "
        "if it has already been processed by the pipeline."
    )

    col_upload, col_lookup = st.columns([1, 1], gap="large")

    with col_upload:
        st.subheader("Upload SAP PDF")
        uploaded_file = st.file_uploader(
            "Select a PDF file (max 50 MB)",
            type=["pdf"],
            help="The PDF must be text-based (not scanned/image-only).",
        )
        if st.button("Process PDF", type="primary", disabled=uploaded_file is None):
            if not os.environ.get("ANTHROPIC_API_KEY"):
                st.error(
                    "ANTHROPIC_API_KEY is not set in your environment. "
                    "Please set it and restart the app before uploading a PDF."
                )
            else:
                with st.spinner(
                    "Extracting text and running AI-assisted SAP analysis — "
                    "this may take 30–90 seconds…"
                ):
                    try:
                        schema = process_uploaded_pdf(uploaded_file, CONFIG)
                        st.session_state["schema"] = schema
                        st.session_state["source"] = uploaded_file.name
                        st.success(
                            f"SAP processed successfully. Switch to the tabs above "
                            f"to review recommendations.",
                            icon="✅",
                        )
                    except RuntimeError as exc:
                        st.error(f"Processing failed: {exc}")
                    except Exception as exc:
                        st.error(f"Unexpected error: {exc}")

    with col_lookup:
        st.subheader("Look Up from Knowledge Base")

        # Build dropdown options from KB
        _kb_df = _load_master_index(CONFIG)
        if not _kb_df.empty and "nct_id" in _kb_df.columns:
            _kb_options = []
            for _, _row in _kb_df.iterrows():
                _nct = _row.get("nct_id", "")
                _title = str(_row.get("study_title", ""))[:60] if _row.get("study_title") else ""
                _ta = str(_row.get("therapeutic_area", "")) if _row.get("therapeutic_area") else ""
                _label = f"{_nct}  —  {_title}" + (f"  [{_ta}]" if _ta else "")
                _kb_options.append((_label, _nct))
            _labels = [o[0] for o in _kb_options]
            _nct_map = {o[0]: o[1] for o in _kb_options}

            selected_label = st.selectbox(
                "Select a study",
                options=["— select —"] + _labels,
                help="Studies already processed by the pipeline.",
            )
            nct_id_input = _nct_map.get(selected_label, "").strip() if selected_label != "— select —" else ""
        else:
            st.info("No studies in the knowledge base yet. Run the pipeline first.")
            nct_id_input = st.text_input("Or enter NCT ID manually", placeholder="e.g. NCT01234567")

        if st.button("Load from Knowledge Base", type="primary", disabled=not nct_id_input):
            nct_id = nct_id_input.strip().upper()
            with st.spinner(f"Loading schema for {nct_id}…"):
                try:
                    schema = lookup_from_index(nct_id, CONFIG)
                    st.session_state["schema"] = schema
                    st.session_state["source"] = nct_id
                    st.success(f"Schema loaded for {nct_id}. Switch to the tabs above to review.", icon="✅")
                except FileNotFoundError as exc:
                    st.warning(str(exc))
                except Exception as exc:
                    st.error(f"Failed to load schema: {exc}")

    # If a schema is loaded, show a quick status bar
    if "schema" in st.session_state:
        schema = st.session_state["schema"]
        source = st.session_state.get("source", "Unknown")
        extraction_conf = schema.get("extraction_confidence", "LOW")

        st.divider()
        st.markdown("### Currently Loaded SAP")

        meta = schema.get("metadata", {})
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        r1c1.metric("NCT ID", meta.get("nct_id", source))
        r1c2.metric("Phase", meta.get("phase") or "—")
        r1c3.metric(
            "Endpoints",
            sum(
                len(schema.get("endpoints", {}).get(k, []))
                for k in ("primary", "secondary", "exploratory")
            ),
        )
        r1c4.metric(
            "Open Questions",
            len(schema.get("open_questions", [])),
        )

        st.markdown(
            _confidence_badge(extraction_conf),
            unsafe_allow_html=True,
        )

        high_count = sum(
            1
            for q in schema.get("open_questions", [])
            if (q.get("severity") or "").upper() == "HIGH"
        )
        if high_count:
            st.markdown(
                f"""
<div style="background:#FFEBEE;border-left:4px solid #C62828;padding:12px 18px;
border-radius:4px;margin-top:12px;">
<strong style="color:#C62828;">⚠ Human Review Required</strong><br/>
<span style="color:#4A1A1A;">{high_count} HIGH severity item(s) detected that require
immediate review before proceeding with study setup.</span>
</div>
""",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Tab: SAP Summary
# ---------------------------------------------------------------------------
# Tab: Knowledge Model
# ---------------------------------------------------------------------------

# ADaM → SDTM source domain mapping (deterministic CDISC relationships)
_ADAM_TO_SDTM: dict[str, list[str]] = {
    "ADSL": ["DM", "DS"],
    "ADAE": ["AE", "CM", "DM"],
    "ADTTE": ["AE", "DS", "DM"],
    "ADQS": ["QS", "DM"],
    "ADEFF": ["VS", "LB", "QS", "DM"],
    "ADVS": ["VS", "DM"],
    "ADLB": ["LB", "DM"],
    "ADPC": ["PC", "EX", "DM"],
    "ADPP": ["PP", "PC", "DM"],
    "ADEG": ["EG", "DM"],
    "ADRS": ["RS", "TR", "TU", "DM"],
    "ADBM": ["LB", "DM"],
    "ADCM": ["CM", "DM"],
    "ADMH": ["MH", "DM"],
}

# Endpoint keyword → likely ADaM candidates
_EP_HINTS: list[tuple[list[str], list[str]]] = [
    (["TTE", "TIME TO EVENT", "SURVIVAL", "EVENT-FREE", "RELAPSE-FREE", "PFS", "OS "],
     ["ADTTE"]),
    (["PRO", "PATIENT REPORTED", "QUESTIONNAIRE", "SCALE", "SCORE", "CDR", "ADAS", "MMSE",
      "COGNITIVE", "BEHAVIORAL", "FUNCTIONAL"],
     ["ADQS"]),
    (["PK", "PHARMACOKINETIC", "CONCENTRATION", "AUC", "CMAX", "TMAX", "T1/2"],
     ["ADPC", "ADPP"]),
    (["TUMOR", "ORR", "DCR", "RECIST", "RADIOLOG", "IMAGING", "PET", "MRI"],
     ["ADRS"]),
    (["ECG", "QTC", "QT INTERVAL"],
     ["ADEG"]),
    (["LAB", "BIOMARKER", "HEMATOL", "CHEMISTRY", "SERUM", "PLASMA LEVEL"],
     ["ADLB"]),
    (["VITAL", "BLOOD PRESSURE", "WEIGHT", "BMI", "HEIGHT"],
     ["ADVS"]),
    (["ADVERSE", "SAFETY", "AE ", "TOLERABILITY"],
     ["ADAE"]),
    (["CFB", "CHANGE FROM BASELINE", "MEAN CHANGE"],
     ["ADEFF", "ADQS"]),
]


def _methods_summary(methods) -> str:
    if not methods:
        return "—"
    if isinstance(methods, dict):
        parts = []
        if methods.get("primary_analysis_method"):
            parts.append(str(methods["primary_analysis_method"]))
        if methods.get("estimand_framework"):
            parts.append("Estimand framework")
        if methods.get("multiplicity_adjustment"):
            parts.append("Multiplicity adjustment")
        return ", ".join(parts) or "—"
    return ", ".join(str(m) for m in methods[:3]) or "—"


def _match_adam_for_endpoint(ep: dict, adam_names: list[str]) -> list[str]:
    """Infer which ADaM datasets relate to an endpoint using type/description keywords."""
    text = ((ep.get("type") or "") + " " + (ep.get("description") or "")).upper()
    for keywords, candidates in _EP_HINTS:
        if any(k in text for k in keywords):
            matched = [a for a in candidates if a in adam_names]
            if matched:
                return matched[:2]
    # fallback: first non-core ADaM
    non_core = [a for a in adam_names if a not in ("ADSL",)]
    return non_core[:1] if non_core else []


def _build_knowledge_graph(schema: dict) -> tuple:
    """
    Build streamlit-agraph nodes/edges + a node registry for drill-down.
    Returns (nodes, edges, node_registry) where node_registry maps
    node_id → dict with keys: type, label, data (schema fragment).
    """
    from streamlit_agraph import Node, Edge

    meta = schema.get("metadata", {})
    study_title = meta.get("study_title") or "Study"
    short_title = study_title[:32] + "…" if len(study_title) > 32 else study_title

    pops       = schema.get("analysis_populations") or []
    estimands  = schema.get("estimands") or []
    eps_prim   = (schema.get("endpoints", {}).get("primary") or [])
    eps_sec    = (schema.get("endpoints", {}).get("secondary") or [])
    methods    = schema.get("statistical_methods") or {}
    sdtm_list  = schema.get("sdtm_domains_expected") or []
    adam_list  = schema.get("adam_datasets_expected") or []
    open_qs    = schema.get("open_questions") or []

    adam_names = [(d.get("dataset") or str(d)) if isinstance(d, dict) else str(d) for d in adam_list]
    sdtm_names = [(d.get("domain") or str(d)) if isinstance(d, dict) else str(d) for d in sdtm_list]

    nodes, edges, registry = [], [], {}
    _counter = [0]

    def uid(prefix):
        _counter[0] += 1
        return f"{prefix}_{_counter[0]}"

    def n(nid, label, color, size, shape="dot", node_type="group", data=None):
        nodes.append(Node(
            id=nid, label=label, size=size, shape=shape,
            color={"background": color, "border": "#ffffff",
                   "highlight": {"background": "#FFD700", "border": "#FFA500"}},
            font={"size": 11, "color": "#ffffff", "face": "Inter,sans-serif"},
        ))
        registry[nid] = {"type": node_type, "label": label, "data": data or {}}

    def e(src, dst, color="#3A5A7C", dashes=False):
        edges.append(Edge(source=src, target=dst, color=color,
                          width=1.5, dashes=dashes,
                          arrows="to"))

    # ── Level 0 ───────────────────────────────────────────────────────────
    n("study_root", short_title, "#1B3A5C", 40, shape="box",
      node_type="study", data=meta)

    # ── Level 1 ───────────────────────────────────────────────────────────
    n("ctx",  "Study Context",      "#1A5276", 28, shape="box", node_type="group",
      data={"populations": pops, "estimands": estimands, "methods": methods})
    n("plan", "Analysis Plan",      "#1A3A5C", 28, shape="box", node_type="group",
      data={"primary": eps_prim, "secondary": eps_sec})
    n("oq",   f"Open Questions ({len(open_qs)})", "#4A235A", 24, shape="box",
      node_type="group", data={"questions": open_qs})
    e("study_root", "ctx",  "#2E86C1")
    e("study_root", "plan", "#2E86C1")
    e("study_root", "oq",   "#8E44AD")

    # ── Level 2: Context ──────────────────────────────────────────────────
    pop_grp = uid("pop_grp")
    n(pop_grp, f"Populations ({len(pops)})", "#1E8449", 22, node_type="pop_group",
      data={"populations": pops})
    e("ctx", pop_grp, "#27AE60")

    est_grp = uid("est_grp")
    n(est_grp, f"Estimands ({len(estimands)})", "#1A6690", 20, node_type="est_group",
      data={"estimands": estimands})
    e("ctx", est_grp, "#5B8DB8")

    meth_grp = uid("meth_grp")
    n(meth_grp, "Statistical Methods", "#BA5A00", 20, node_type="methods",
      data=methods)
    e("ctx", meth_grp, "#E67E22")

    # ── Level 3: Populations ──────────────────────────────────────────────
    for p in pops[:6]:
        abbr = p.get("abbreviation") or p.get("name") or "Pop"
        pid = uid("pop")
        n(pid, abbr, "#A9DFBF", 14, node_type="population", data=p)
        e(pop_grp, pid, "#27AE60")

    # ── Level 3: Estimands ────────────────────────────────────────────────
    for i, est in enumerate(estimands[:5]):
        lbl = (est.get("treatment_comparison") or est.get("population") or f"Estimand {i+1}")[:26]
        eid = uid("est")
        n(eid, lbl, "#AED6F1", 14, node_type="estimand", data=est)
        e(est_grp, eid, "#5B8DB8")

    # ── Level 3: Method details ───────────────────────────────────────────
    if isinstance(methods, dict):
        mdetails = [
            ("primary_analysis_method", "Primary method"),
            ("multiplicity_adjustment", "Multiplicity adj."),
            ("estimand_framework",      "Estimand framework"),
            ("bayesian_elements",       "Bayesian elements"),
            ("missing_data_handling",   "Missing data"),
        ]
        for key, label in mdetails:
            val = methods.get(key)
            if val and val is not False:
                mid = uid("meth")
                short = str(val)[:24] if isinstance(val, str) else label
                n(mid, short, "#FDEBD0", 12, node_type="method_detail",
                  data={"field": label, "value": str(val)})
                e(meth_grp, mid, "#E67E22")

    # ── Level 2: Analysis Plan ────────────────────────────────────────────
    prim_grp = uid("prim_grp")
    n(prim_grp, f"Primary ({len(eps_prim)})", "#7D6608", 22, node_type="ep_group",
      data={"endpoints": eps_prim, "tier": "Primary"})
    e("plan", prim_grp, "#F0A500")

    sec_grp = uid("sec_grp")
    n(sec_grp, f"Secondary ({len(eps_sec)})", "#5D6D7E", 18, node_type="ep_group",
      data={"endpoints": eps_sec, "tier": "Secondary"})
    e("plan", sec_grp, "#AAB7B8")

    # ── Levels 3-5: Endpoint → ADaM → SDTM ───────────────────────────────
    def add_ep_lineage(ep, parent_id, ep_color):
        ep_type = ep.get("type") or "Endpoint"
        ep_id = uid("ep")
        n(ep_id, ep_type[:26], ep_color, 16, node_type="endpoint", data=ep)
        e(parent_id, ep_id, ep_color)

        for adam_name in (_match_adam_for_endpoint(ep, adam_names) or adam_names[:1]):
            adam_rat = next(
                ((d.get("rationale") or "") for d in adam_list
                 if isinstance(d, dict) and d.get("dataset") == adam_name), "")
            a_id = uid(f"adam")
            n(a_id, adam_name, "#8E44AD", 18, node_type="adam",
              data={"dataset": adam_name, "rationale": adam_rat,
                    "endpoint_type": ep_type,
                    "sdtm_sources": _ADAM_TO_SDTM.get(adam_name, [])})
            e(ep_id, a_id, "#8E44AD")

            for sdtm_name in [s for s in _ADAM_TO_SDTM.get(adam_name, []) if s in sdtm_names][:4]:
                sdtm_rat = next(
                    ((d.get("rationale") or "") for d in sdtm_list
                     if isinstance(d, dict) and d.get("domain") == sdtm_name), "")
                s_id = uid("sdtm")
                n(s_id, sdtm_name, "#D7BDE2", 14, node_type="sdtm",
                  data={"domain": sdtm_name, "rationale": sdtm_rat,
                        "adam_parent": adam_name})
                e(a_id, s_id, "#C0A0D8", dashes=True)

    for ep in eps_prim[:4]:
        add_ep_lineage(ep, prim_grp, "#F0A500")
    for ep in eps_sec[:3]:
        add_ep_lineage(ep, sec_grp, "#AAB7B8")

    # ── Open Questions ────────────────────────────────────────────────────
    high_qs  = [q for q in open_qs if (q.get("severity") or "").upper() == "HIGH"]
    other_qs = [q for q in open_qs if (q.get("severity") or "").upper() != "HIGH"]

    if high_qs:
        hg = uid("high_grp")
        n(hg, f"HIGH ({len(high_qs)})", "#922B21", 18, node_type="oq_group",
          data={"questions": high_qs, "severity": "HIGH"})
        e("oq", hg, "#E74C3C")
        for q in high_qs[:4]:
            qid = uid("oq_h")
            lbl = (q.get("category") or "Question")[:26]
            n(qid, lbl, "#F1948A", 12, node_type="open_question", data=q)
            e(hg, qid, "#E74C3C")

    if other_qs:
        og = uid("other_grp")
        n(og, f"MED/LOW ({len(other_qs)})", "#5D6D7E", 16, node_type="oq_group",
          data={"questions": other_qs, "severity": "MEDIUM/LOW"})
        e("oq", og, "#AAB7B8")
        for q in other_qs[:3]:
            qid = uid("oq_m")
            lbl = (q.get("category") or "Question")[:26]
            n(qid, lbl, "#CCD1D1", 11, node_type="open_question", data=q)
            e(og, qid, "#AAB7B8")

    return nodes, edges, registry


def _render_node_detail(node_id: str, registry: dict, schema: dict):
    """Render a detailed drill-down panel for the clicked node."""
    info = registry.get(node_id)
    if not info:
        return

    ntype = info["type"]
    data  = info["data"]
    label = info["label"]

    st.markdown(
        f'<div style="background:#1C2B3A;border-left:4px solid #3A9BDC;'
        f'padding:14px 20px;border-radius:6px;margin-top:8px;">'
        f'<span style="color:#7EC8E3;font-size:0.75rem;text-transform:uppercase;'
        f'letter-spacing:.08em;">{ntype.replace("_"," ")}</span>'
        f'<h4 style="color:#FFFFFF;margin:4px 0 0;">{label}</h4></div>',
        unsafe_allow_html=True,
    )
    st.write("")

    if ntype == "study":
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Title:** {data.get('study_title') or '—'}")
            st.markdown(f"**NCT ID:** {data.get('nct_id') or '—'}")
            st.markdown(f"**Sponsor:** {data.get('sponsor') or '—'}")
        with col2:
            st.markdown(f"**Phase:** {data.get('phase') or '—'}")
            st.markdown(f"**Therapeutic Area:** {data.get('therapeutic_area') or '—'}")
            st.markdown(f"**Indication:** {data.get('indication') or '—'}")

    elif ntype == "population":
        st.markdown(f"**Abbreviation:** {data.get('abbreviation') or data.get('name') or '—'}")
        st.markdown(f"**Full Name:** {data.get('name') or '—'}")
        if data.get("description"):
            st.markdown(f"**Definition:** {data['description']}")
        if data.get("sap_section"):
            st.caption(f"SAP Section: {data['sap_section']}")

    elif ntype == "estimand":
        for field, label_txt in [
            ("population", "Population"),
            ("treatment_comparison", "Treatment Comparison"),
            ("variable", "Variable / Endpoint"),
            ("intercurrent_events", "Intercurrent Events Strategy"),
            ("population_level_summary", "Summary Measure"),
        ]:
            if data.get(field):
                st.markdown(f"**{label_txt}:** {data[field]}")

    elif ntype == "methods":
        for field, label_txt in [
            ("primary_analysis_method", "Primary Method"),
            ("covariates", "Covariates"),
            ("visit_windowing", "Visit Windowing"),
            ("missing_data_handling", "Missing Data"),
            ("multiplicity_adjustment", "Multiplicity Adjustment"),
            ("estimand_framework", "Estimand Framework"),
            ("bayesian_elements", "Bayesian Elements"),
        ]:
            val = data.get(field)
            if val is not None and val is not False:
                st.markdown(f"**{label_txt}:** {val}")

    elif ntype == "method_detail":
        st.markdown(f"**{data.get('field','Field')}:** {data.get('value','—')}")

    elif ntype == "endpoint":
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Type:** {data.get('type') or '—'}")
            st.markdown(f"**Timepoint:** {data.get('timepoint') or '—'}")
        with col2:
            matched = _match_adam_for_endpoint(
                data,
                [(d.get("dataset") or str(d)) if isinstance(d, dict) else str(d)
                 for d in (schema.get("adam_datasets_expected") or [])]
            )
            st.markdown(f"**Recommended ADaM:** {', '.join(matched) or '—'}")
        if data.get("description"):
            st.markdown(f"**Description:** {data['description']}")
        if data.get("sap_section"):
            st.caption(f"SAP Section: {data['sap_section']}")

    elif ntype == "adam":
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Dataset:** {data.get('dataset','—')}")
            st.markdown(f"**Drives endpoint:** {data.get('endpoint_type','—')}")
        with col2:
            sdtm_src = data.get("sdtm_sources") or []
            st.markdown(f"**SDTM Sources:** {', '.join(sdtm_src) or '—'}")
        if data.get("rationale"):
            st.markdown(f"**Rationale:** {data['rationale']}")
        conf = (schema.get("component_confidence") or {}).get(data.get("dataset",""))
        if conf is not None:
            st.markdown(f"**Confidence:** {get_confidence_label(conf)}")

    elif ntype == "sdtm":
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Domain:** {data.get('domain','—')}")
            st.markdown(f"**Feeds into:** {data.get('adam_parent','—')}")
        with col2:
            st.markdown(f"**Standard:** SDTM IG v3.4")
        if data.get("rationale"):
            st.markdown(f"**Rationale:** {data['rationale']}")

    elif ntype == "open_question":
        sev = data.get("severity", "MEDIUM")
        color, _ = _SEVERITY_COLORS.get(sev, ("#616161", "#F5F5F5"))
        st.markdown(
            f'{_severity_badge(sev)} **{data.get("category","Question")}**',
            unsafe_allow_html=True,
        )
        st.markdown(f"> {data.get('question','')}")
        if data.get("sap_section"):
            st.caption(f"SAP Section: {data['sap_section']}")
        if data.get("sap_excerpt"):
            with st.expander("SAP excerpt", expanded=True):
                st.markdown(
                    f'<div style="font-size:0.85rem;font-family:Georgia,serif;'
                    f'line-height:1.6;color:#444;">{data["sap_excerpt"]}</div>',
                    unsafe_allow_html=True,
                )

    elif ntype in ("group", "pop_group", "est_group", "ep_group", "oq_group"):
        # Summary card for a group node
        if "populations" in data:
            for p in data["populations"][:5]:
                abbr = p.get("abbreviation") or p.get("name") or "—"
                st.markdown(f"- **{abbr}** — {(p.get('description') or '')[:80]}")
        if "endpoints" in data:
            tier = data.get("tier", "")
            for ep in data["endpoints"][:5]:
                st.markdown(f"- **{ep.get('type','?')}** — {(ep.get('description') or '')[:80]}")
        if "estimands" in data:
            for est in data["estimands"][:4]:
                st.markdown(f"- {est.get('treatment_comparison') or est.get('population') or '—'}")
        if "questions" in data:
            for q in data["questions"][:5]:
                sev = q.get("severity","?")
                st.markdown(f"- {_severity_badge(sev)} {q.get('question','')[:80]}",
                            unsafe_allow_html=True)


def _tab_knowledge_model(schema: dict):
    from streamlit_agraph import agraph, Config

    st.header("Study Knowledge Model")
    st.caption(
        "5-level CDISC lineage: Study → Context / Analysis Plan → "
        "Endpoints → ADaM Datasets → SDTM Source Domains. "
        "**Click any node** to see its details below the graph."
    )

    try:
        nodes, edges, registry = _build_knowledge_graph(schema)
    except Exception as exc:
        st.error(f"Could not build knowledge graph: {exc}")
        return

    config = Config(
        width="100%",
        height=700,
        directed=True,
        physics=False,
        hierarchical=True,
        hierarchical_sort_method="directed",
        direction="LR",
        levelSeparation=210,
        nodeSpacing=85,
        treeSpacing=130,
        blockShifting=True,
        edgeMinimization=True,
        parentCentralization=True,
        node={"labelProperty": "label"},
        link={"renderLabel": False},
    )

    clicked = agraph(nodes=nodes, edges=edges, config=config)

    # ── Drill-down panel ──────────────────────────────────────────────────
    if clicked:
        st.divider()
        _render_node_detail(clicked, registry, schema)
    else:
        st.caption("↑ Click any node in the graph to inspect its details here.")


# ---------------------------------------------------------------------------

def _tab_sap_summary(schema: dict):
    st.header("Detected SAP Summary")
    st.caption(
        "The following information was detected by AI-assisted extraction. "
        "Review for accuracy before use."
    )

    meta = schema.get("metadata", {})

    # --- Metadata card ---
    with st.expander("Study Metadata", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Study Title:** {meta.get('study_title') or '—'}")
            st.markdown(f"**NCT ID:** {meta.get('nct_id') or '—'}")
            st.markdown(f"**Sponsor:** {meta.get('sponsor') or '—'}")
            st.markdown(f"**Phase:** {meta.get('phase') or '—'}")
        with col2:
            st.markdown(f"**Therapeutic Area:** {meta.get('therapeutic_area') or '—'}")
            st.markdown(f"**Indication:** {meta.get('indication') or '—'}")
            st.markdown(f"**Extraction Date:** {meta.get('extraction_date') or '—'}")
            st.markdown(f"**Source Document:** {meta.get('source_document') or '—'}")

    extraction_notes = schema.get("extraction_notes", "")
    if extraction_notes:
        with st.expander("Extraction Notes (AI)", expanded=False):
            st.info(extraction_notes)

    # --- Endpoints ---
    st.subheader("Detected Endpoints")
    endpoints = schema.get("endpoints", {})

    for ep_type, label, color in [
        ("primary",     "Primary Endpoints",     "#0066CC"),
        ("secondary",   "Secondary Endpoints",   "#5C6BC0"),
        ("exploratory", "Exploratory Endpoints", "#78909C"),
    ]:
        ep_list = endpoints.get(ep_type, [])
        if not ep_list:
            continue
        st.markdown(
            f'<span style="color:{color};font-weight:700;font-size:1rem;">'
            f'{label} ({len(ep_list)})</span>',
            unsafe_allow_html=True,
        )
        rows = []
        for ep in ep_list:
            rows.append({
                "Description": ep.get("description") or "—",
                "Type": ep.get("type") or "—",
                "Timepoint": ep.get("timepoint") or "—",
            })
        st.dataframe(
            pd.DataFrame(rows),
            width="stretch",
            hide_index=True,
        )

    # --- Analysis Populations ---
    st.subheader("Detected Analysis Populations")
    pops = schema.get("analysis_populations", [])
    if pops:
        pop_rows = []
        for p in pops:
            pop_rows.append({
                "Name": p.get("name") or "—",
                "Abbreviation": p.get("abbreviation") or "—",
                "Definition": p.get("definition") or "—",
            })
        st.dataframe(pd.DataFrame(pop_rows), width="stretch", hide_index=True)
    else:
        st.info("No analysis populations detected. Review SAP manually.")

    # --- Statistical Methods ---
    st.subheader("Detected Statistical Methods")
    sm = schema.get("statistical_methods", {})
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Primary Analysis Method:** {sm.get('primary_analysis_method') or '—'}")
        st.markdown(f"**Covariates:** {sm.get('covariates') or '—'}")
        st.markdown(f"**Visit Windowing:** {sm.get('visit_windowing') or '—'}")
        st.markdown(f"**Missing Data Handling:** {sm.get('missing_data_handling') or '—'}")
    with col2:
        st.markdown(f"**Multiplicity Adjustment:** {'Yes' if sm.get('multiplicity_adjustment') else 'No / Not detected'}")
        st.markdown(f"**Estimand Framework:** {'Yes' if sm.get('estimand_framework') else 'No / Not detected'}")
        st.markdown(f"**Bayesian Elements:** {'Yes' if sm.get('bayesian_elements') else 'No / Not detected'}")
        st.markdown(f"**SAP Section Reference:** {sm.get('sap_section') or '—'}")

    # --- Special Assessments ---
    st.subheader("Detected Special Assessments")
    sa = schema.get("special_assessments", {})
    _bool_fields = [
        ("pharmacokinetics",       "Pharmacokinetics (PK)"),
        ("tumor_response",         "Tumor Response"),
        ("patient_reported_outcomes", "Patient-Reported Outcomes (PRO)"),
        ("biomarkers",             "Biomarkers"),
        ("imaging",                "Imaging"),
        ("ecg",                    "ECG"),
        ("ophthalmology",          "Ophthalmology"),
    ]
    assess_rows = [
        {"Assessment": label, "Detected": "Yes" if sa.get(key) else "No"}
        for key, label in _bool_fields
    ]
    nse = sa.get("non_standard_endpoints", [])
    if nse:
        assess_rows.append({"Assessment": "Non-standard Endpoints", "Detected": ", ".join(nse)})

    st.dataframe(pd.DataFrame(assess_rows), width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Tab: SDTM Mapping
# ---------------------------------------------------------------------------

def _tab_sdtm(schema: dict):
    st.header("Recommended SDTM Domains")
    st.markdown(
        "> **Review required.** The domains below are *recommended* based on the "
        "detected SAP content. Confirm alignment with your study protocol and SDTMIG v3.4."
    )

    df = schema_to_sdtm_table(schema)
    if df.empty:
        st.warning("No SDTM domain recommendations available. Check the SAP Summary tab.")
        return

    # Summary metrics
    always_count = int(df["Always Present"].sum())
    conditional_count = len(df) - always_count
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Recommended Domains", len(df))
    c2.metric("Standard (Always Present)", always_count)
    c3.metric("Conditional (Study-specific)", conditional_count)

    st.divider()

    # Display table (without the raw boolean Always Present col — use a badge instead)
    display_df = df.copy()
    display_df["Type"] = display_df["Always Present"].map(
        lambda v: "Standard" if v else "Conditional"
    )
    display_df = display_df[["Domain", "Description", "Type", "Rationale", "SAP Section"]]

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Domain": st.column_config.TextColumn("Domain", width="small"),
            "Description": st.column_config.TextColumn("Description", width="medium"),
            "Type": st.column_config.TextColumn("Type", width="small"),
            "Rationale": st.column_config.TextColumn("Rationale", width="large"),
            "SAP Section": st.column_config.TextColumn("SAP Section", width="small"),
        },
    )

    st.caption(
        "Traceability: 'SAP Section' column shows the document section that triggered "
        "each conditional domain recommendation. '—' indicates standard domains always expected."
    )


# ---------------------------------------------------------------------------
# Tab: ADaM Mapping
# ---------------------------------------------------------------------------

def _tab_adam(schema: dict):
    st.header("Recommended ADaM Datasets")
    st.markdown(
        "> **Review required.** The datasets below are *recommended* based on the "
        "detected endpoints, populations, and special assessments. Validate against "
        "ADaM IG v2.1 and study-specific requirements."
    )

    df = schema_to_adam_table(schema)
    if df.empty:
        st.warning("No ADaM dataset recommendations available. Check the SAP Summary tab.")
        return

    # Summary metrics
    high_conf = int((df["Confidence Score"] >= 80).sum())
    mid_conf  = int(((df["Confidence Score"] >= 60) & (df["Confidence Score"] < 80)).sum())
    low_conf  = int((df["Confidence Score"] < 60).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Recommended Datasets", len(df))
    c2.metric("High Confidence (≥80%)", high_conf)
    c3.metric("Medium Confidence (60–79%)", mid_conf)
    c4.metric("Low Confidence (<60%)", low_conf)

    st.divider()

    # Render each dataset as a card with colored confidence badge
    for _, row in df.iterrows():
        score = row["Confidence Score"]
        color = get_confidence_color(score)

        with st.container():
            header_col, badge_col = st.columns([5, 1])
            with header_col:
                st.markdown(
                    f"### {row['Dataset']} &nbsp; <span style='font-size:0.9rem;"
                    f"color:#555;font-weight:400;'>{row['Description']}</span>",
                    unsafe_allow_html=True,
                )
            with badge_col:
                st.markdown(
                    _adam_score_badge(score),
                    unsafe_allow_html=True,
                )

            detail_c1, detail_c2, detail_c3 = st.columns([2, 2, 1])
            with detail_c1:
                st.markdown(f"**Key Variables:**  \n`{row['Key Variables']}`")
            with detail_c2:
                st.markdown(f"**Depends On:**  \n{row['Depends On']}")
            with detail_c3:
                st.markdown(f"**SAP Section:**  \n{row['SAP Section']}")

            st.divider()


# ---------------------------------------------------------------------------
# Tab: Open Questions
# ---------------------------------------------------------------------------

def _load_sap_text(schema: dict) -> str | None:
    """Return extracted SAP text for a schema, or None if unavailable."""
    meta = schema.get("metadata", {})
    nct_id = meta.get("nct_id", "")

    # Try the stored text_path first (pipeline run in same session)
    text_path_str = meta.get("text_path")
    if text_path_str:
        p = Path(text_path_str)
        if p.exists():
            return p.read_text(encoding="utf-8")

    # Fall back to KB extracted_text directory
    if nct_id:
        kb_text = Path(CONFIG["storage"]["base_path"]) / "extracted_text" / f"{nct_id}.txt"
        if kb_text.exists():
            return kb_text.read_text(encoding="utf-8")

    return None


def _find_sap_passage(text: str, query: str, window: int = 400) -> str | None:
    """
    Return a window of `window` chars around the best keyword match for `query`
    in `text`. Returns None if no reasonable match found.
    """
    if not text or not query:
        return None
    # Build search terms from the query (drop common stop words)
    stop = {"the", "a", "an", "in", "of", "for", "to", "is", "are", "be",
            "was", "not", "this", "that", "with", "and", "or", "how", "what",
            "which", "whether", "does", "should"}
    words = [w.strip("?.,:;") for w in query.lower().split() if w.lower() not in stop and len(w) > 3]
    if not words:
        return None

    text_lower = text.lower()
    best_pos, best_hits = 0, 0
    # Slide a window and count hits
    step = max(1, len(text) // 300)
    for i in range(0, len(text) - window, step):
        chunk = text_lower[i: i + window]
        hits = sum(1 for w in words if w in chunk)
        if hits > best_hits:
            best_hits, best_pos = hits, i

    if best_hits == 0:
        return None

    start = max(0, best_pos)
    end = min(len(text), start + window)
    excerpt = text[start:end].strip()
    # Trim to sentence boundaries if possible
    if ". " in excerpt[20:]:
        excerpt = excerpt[excerpt.index(". ", 20) + 2:]
    if ". " in excerpt[50:]:
        last = excerpt.rindex(". ", 0, -10)
        excerpt = excerpt[:last + 1]
    return f"…{excerpt}…" if start > 0 else excerpt


def _tab_open_questions(schema: dict):
    sap_text = _load_sap_text(schema)
    oqs_df = schema_to_open_questions_df(schema)
    high_count = int((oqs_df["Severity"] == "HIGH").sum()) if not oqs_df.empty else 0

    st.header("Open Questions & Flags")
    st.markdown(
        "The items below represent ambiguities or missing information detected in the SAP. "
        "Each should be reviewed and resolved with the study team **before** programming begins."
    )

    if high_count:
        st.markdown(
            f"""
<div style="background:#FFEBEE;border-left:4px solid #C62828;padding:14px 20px;
border-radius:4px;margin-bottom:16px;">
<strong style="color:#C62828;font-size:1.05rem;">⚠ Human Review Required</strong><br/>
<span style="color:#4A1A1A;">{high_count} HIGH severity item(s) detected. These represent significant
gaps in the SAP that must be clarified before CDISC implementation can proceed.</span>
</div>
""",
            unsafe_allow_html=True,
        )

    if oqs_df.empty:
        st.success(
            "No open questions or flags detected. This may indicate a well-specified SAP "
            "or limited extraction coverage — verify manually.",
            icon="✅",
        )
        return

    # Severity filter
    severities = ["ALL"] + sorted(oqs_df["Severity"].unique().tolist(),
                                   key=lambda s: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(s, 9))
    selected_sev = st.selectbox("Filter by Severity", severities)

    filtered = oqs_df if selected_sev == "ALL" else oqs_df[oqs_df["Severity"] == selected_sev]

    st.markdown(f"**Showing {len(filtered)} of {len(oqs_df)} items**")
    st.divider()

    raw_qs = schema.get("open_questions") or []
    q_by_question = {q.get("question", ""): q for q in raw_qs}

    for _, row in filtered.iterrows():
        sev = row["Severity"]
        cat = row["Category"]
        question = row["Question"]

        _, bg = _SEVERITY_COLORS.get(sev, ("#616161", "#F5F5F5"))
        border = _SEVERITY_COLORS.get(sev, ("#616161", "#F5F5F5"))[0]

        raw = q_by_question.get(question, {})
        sap_section = raw.get("sap_section") or ""
        sap_excerpt_stored = raw.get("sap_excerpt") or ""

        section_html = (
            f'<span style="font-size:0.75rem;color:#888;margin-left:8px;">'
            f'📄 {sap_section}</span>' if sap_section else ""
        )

        st.markdown(
            f"""<div style="background:{bg};border-left:4px solid {border};
padding:12px 18px;border-radius:4px;margin-bottom:6px;">
{_severity_badge(sev)}&nbsp;&nbsp;<span style="font-size:0.8rem;color:#555;">{cat}</span>
{section_html}<br/>
<span style="margin-top:6px;display:block;font-weight:500;">{question}</span>
</div>""",
            unsafe_allow_html=True,
        )

        # SAP excerpt — prefer stored, fall back to fuzzy search
        excerpt = sap_excerpt_stored
        if not excerpt and sap_text:
            excerpt = _find_sap_passage(sap_text, question) or ""

        if excerpt:
            with st.expander("View SAP excerpt", expanded=False):
                st.markdown(
                    f'<div style="font-size:0.85rem;color:#444;background:#F8F9FA;'
                    f'border-left:3px solid #BDBDBD;padding:10px 14px;border-radius:3px;'
                    f'font-family:Georgia,serif;line-height:1.6;">{excerpt}</div>',
                    unsafe_allow_html=True,
                )
        st.write("")


# ---------------------------------------------------------------------------
# Tab: SAP Text
# ---------------------------------------------------------------------------

def _tab_sap_text(schema: dict):
    st.header("SAP Text Viewer")
    st.caption(
        "Extracted plain text from the source SAP PDF. "
        "Use the search box to locate specific passages."
    )

    sap_text = _load_sap_text(schema)
    if not sap_text:
        st.info(
            "SAP text is not available for this study. "
            "Text is stored only when a study is processed through the pipeline in this session "
            "or when the extracted_text directory is present in the knowledge base.",
            icon="ℹ️",
        )
        return

    meta = schema.get("metadata", {})
    open_qs = schema.get("open_questions") or []

    st.markdown(
        f"**{len(sap_text):,} characters** extracted  ·  "
        f"Source: `{meta.get('source_document') or 'SAP PDF'}`"
    )

    # ── Jump-to buttons from open questions ──────────────────────────────
    if open_qs:
        st.markdown("**Jump to open question context:**")
        cols = st.columns(min(4, len(open_qs)))
        for i, q in enumerate(open_qs[:8]):
            label = (q.get("sap_section") or q.get("category") or f"Q{i+1}")[:22]
            if cols[i % 4].button(f"📌 {label}", key=f"jump_{i}"):
                st.session_state["sap_search"] = (q.get("question") or "")[:60]
        st.divider()

    # ── Search box ───────────────────────────────────────────────────────
    search = st.text_input(
        "Search in SAP text",
        value=st.session_state.get("sap_search", ""),
        placeholder="Enter keyword or phrase…",
        key="sap_search_input",
    )

    if search and search.strip():
        # Find all occurrences
        text_lower = sap_text.lower()
        query_lower = search.lower().strip()
        matches = []
        start = 0
        while True:
            pos = text_lower.find(query_lower, start)
            if pos == -1:
                break
            matches.append(pos)
            start = pos + 1

        if matches:
            st.success(f"Found **{len(matches)}** match(es). Showing up to 5.", icon="🔍")
            for pos in matches[:5]:
                snippet_start = max(0, pos - 300)
                snippet_end = min(len(sap_text), pos + 400)
                before = sap_text[snippet_start:pos].replace("\n", " ")
                match_text = sap_text[pos: pos + len(search)]
                after = sap_text[pos + len(search): snippet_end].replace("\n", " ")
                st.markdown(
                    f'<div style="font-size:0.85rem;background:#FFFDE7;border-left:3px solid #F9A825;'
                    f'padding:10px 14px;border-radius:3px;margin-bottom:8px;'
                    f'font-family:Georgia,serif;line-height:1.65;">'
                    f'{"…" if snippet_start > 0 else ""}{before}'
                    f'<mark style="background:#FFD54F;padding:1px 3px;border-radius:2px;">{match_text}</mark>'
                    f'{after}{"…" if snippet_end < len(sap_text) else ""}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.warning(f'No matches found for "{search}".', icon="⚠️")
        st.divider()

    # ── Full text display ─────────────────────────────────────────────────
    with st.expander("Full SAP Text", expanded=not bool(search)):
        # Split into chunks of ~3000 chars to avoid huge single blocks
        chunk_size = 3000
        chunks = [sap_text[i: i + chunk_size] for i in range(0, len(sap_text), chunk_size)]
        for chunk in chunks:
            st.markdown(
                f'<div style="font-size:0.82rem;font-family:Georgia,serif;'
                f'line-height:1.7;color:#333;white-space:pre-wrap;">{chunk}</div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Tab: Export
# ---------------------------------------------------------------------------

def _tab_export(schema: dict):
    st.header("Export Mapping Package")
    st.markdown(
        "Download the full CDISC mapping recommendations for use in study setup, "
        "specification authoring, and handoff to the programming team."
    )

    st.info(
        "All exported materials are **recommendations** that require review and "
        "sign-off by a qualified CDISC programmer or biostatistician.",
        icon="ℹ️",
    )

    meta = schema.get("metadata", {})
    nct_id = meta.get("nct_id", "STUDY")
    study_title = meta.get("study_title", "")
    if study_title:
        st.markdown(f"**Study:** {study_title}")
    st.markdown(f"**NCT ID:** {nct_id}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Mapping CSV")
        st.caption(
            "A flat CSV containing metadata, endpoints, populations, "
            "SDTM domains, ADaM datasets, and open questions. "
            "Suitable for Excel, project trackers, and specification templates."
        )
        csv_str = schema_to_export_csv(schema)
        st.download_button(
            label="Download Mapping CSV",
            data=csv_str.encode("utf-8"),
            file_name=f"{nct_id}_CDISC_mapping.csv",
            mime="text/csv",
            type="primary",
        )

    with col2:
        st.subheader("Full JSON Schema")
        st.caption(
            "The complete extracted and enriched schema in JSON format. "
            "Includes all raw fields, confidence scores, section references, "
            "and open questions. Suitable for programmatic downstream use."
        )
        json_str = json.dumps(schema, indent=2, ensure_ascii=False)
        st.download_button(
            label="Download Full JSON",
            data=json_str.encode("utf-8"),
            file_name=f"{nct_id}_schema.json",
            mime="application/json",
            type="primary",
        )

    st.divider()
    st.subheader("Package Contents Preview")

    sdtm_df = schema_to_sdtm_table(schema)
    adam_df = schema_to_adam_table(schema)
    oq_df   = schema_to_open_questions_df(schema)

    col_a, col_b, col_c, col_d = st.columns(4)
    eps = schema.get("endpoints", {})
    col_a.metric(
        "Endpoints",
        sum(len(eps.get(k, [])) for k in ("primary", "secondary", "exploratory")),
    )
    col_b.metric("SDTM Domains", len(sdtm_df))
    col_c.metric("ADaM Datasets", len(adam_df))
    col_d.metric("Open Questions", len(oq_df))


# ---------------------------------------------------------------------------
# Tab: Pipeline
# ---------------------------------------------------------------------------

_TA_DISPLAY_OPTIONS = [
    "Oncology",
    "CNS / Alzheimer",
    "Cardiovascular",
    "Immunology / Autoimmune",
    "Respiratory",
    "Rare Disease",
    "Diabetes / Metabolic",
]

_TA_QUERY_MAP = {
    "Oncology": "oncology",
    "CNS / Alzheimer": "alzheimer",
    "Cardiovascular": "cardiovascular",
    "Immunology / Autoimmune": "autoimmune",
    "Respiratory": "respiratory",
    "Rare Disease": "rare disease",
    "Diabetes / Metabolic": "diabetes",
}


def _load_master_index(config: dict) -> pd.DataFrame:
    """Load master_index.csv from the knowledge base. Returns empty DataFrame if not found."""
    try:
        storage_cfg = config["storage"]
        base_path = Path(storage_cfg["base_path"])
        index_path = base_path / storage_cfg["master_index_file"]
        if not index_path.exists():
            return pd.DataFrame()
        df = pd.read_csv(index_path)
        return df
    except (FileNotFoundError, KeyError, Exception):
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Thread-safe pipeline state — cached so it survives Streamlit reruns
# (Streamlit re-executes the script top-to-bottom on every interaction;
#  @st.cache_resource ensures this dict is only created once per process)
# ---------------------------------------------------------------------------
@st.cache_resource
def _get_pipeline_state() -> dict:
    return {
        "running": False,
        "pct": 0,
        "msg": "",
        "log": [],
        "error": None,
        "start_time": 0.0,
        "results": [],
        "stop_requested": False,
    }

_PIPELINE_STATE = _get_pipeline_state()


def _run_pipeline_thread(config_override: dict, api_key: str):
    """Pipeline worker that runs in a background thread, writing progress to _PIPELINE_STATE."""
    from pipeline import step1_query, step2_download, step3_extract, step4_parse, step5_index, rule_engine
    import json as _json

    def _set(pct: int, msg: str, log: str = ""):
        _PIPELINE_STATE["pct"] = pct
        _PIPELINE_STATE["msg"] = msg
        if log:
            _PIPELINE_STATE["log"].append(log)

    def _stopped() -> bool:
        if _PIPELINE_STATE.get("stop_requested"):
            _set(_PIPELINE_STATE["pct"], "⛔ Pipeline stopped by user.", "⛔ Stopped by user.")
            return True
        return False

    try:
        _PIPELINE_STATE["running"] = True
        _PIPELINE_STATE["stop_requested"] = False
        _PIPELINE_STATE["log"] = []
        _PIPELINE_STATE["error"] = None

        _set(0, "Step 1/5 — Querying ClinicalTrials.gov…", "Querying ClinicalTrials.gov…")
        query_results = step1_query.run(config_override)
        _set(20, f"Step 1/5 complete — {len(query_results)} studies found.",
             f"Found {len(query_results)} studies with SAP documents.")
        if _stopped(): return

        if not query_results:
            _set(100, "Done — no studies found.", "No studies matched the selected filters.")
        else:
            _set(20, "Step 2/5 — Downloading SAP PDFs…", "Downloading SAP PDFs…")
            download_results = step2_download.run(config_override, query_results)
            _set(40, f"Step 2/5 complete — {len(download_results)} PDFs downloaded.",
                 f"Downloaded {len(download_results)} PDFs.")
            if _stopped(): return

            _set(40, "Step 3/5 — Extracting text from PDFs…", "Extracting text from PDFs…")
            extraction_results = step3_extract.run(config_override, download_results)
            _set(60, f"Step 3/5 complete — {len(extraction_results)} documents extracted.",
                 f"Extracted text from {len(extraction_results)} documents.")
            if _stopped(): return

            if api_key:
                _set(60, "Step 4/5 — Parsing with Claude API…", "Parsing with Claude API…")
                parse_results = step4_parse.run(config_override, extraction_results)
                _set(75, f"Step 4/5 complete — {len(parse_results)} schemas parsed.",
                     f"Parsed {len(parse_results)} schemas.")

                _set(75, "Step 4b/5 — Generating CDISC recommendations…",
                     "Generating CDISC recommendations…")
                enriched_results = []
                for idx, result in enumerate(parse_results):
                    pct = 75 + int(15 * (idx + 1) / max(len(parse_results), 1))
                    _set(pct, f"Step 4b/5 — Recommending for study {idx+1}/{len(parse_results)}…")
                    schema_obj = result.get("schema")
                    if schema_obj:
                        enriched = rule_engine.run_rules(schema_obj, config_override)
                        schema_path = result.get("schema_path")
                        if schema_path:
                            try:
                                with open(schema_path, "w", encoding="utf-8") as fh:
                                    _json.dump(enriched, fh, indent=2, ensure_ascii=False)
                            except Exception:
                                pass
                        enriched_results.append({**result, "schema": enriched})
                    else:
                        enriched_results.append(result)
                _set(90, f"Step 4b/5 complete — {len(enriched_results)} recommendations generated.",
                     f"CDISC recommendations generated for {len(enriched_results)} schemas.")
            else:
                _set(62, "Skipping Steps 4/4b — no API key.", "Skipping Steps 4 and 4b (no API key).")
                enriched_results = []

            _set(90, "Step 5/5 — Updating knowledge base index…", "Updating knowledge base index…")
            step5_index.run(config_override, enriched_results)
            added = len(enriched_results)
            _set(100, f"Done — {added} SAPs added to knowledge base.",
                 f"✅ Pipeline complete. {added} SAPs added to knowledge base.")

            # Store result rows for display in status section
            result_rows = []
            for r in enriched_results:
                schema = r.get("schema") or {}
                result_rows.append({
                    "NCT ID": r.get("nct_id", ""),
                    "Title": r.get("study_title", schema.get("study_title", ""))[:80],
                    "TA": r.get("therapeutic_area", ""),
                    "Phase": r.get("phase", ""),
                    "Primary Endpoint": schema.get("primary_endpoint_type", ""),
                    "SDTM Domains": ", ".join(
                        d if isinstance(d, str) else d.get("domain", str(d))
                        for d in schema.get("sdtm_domains_expected", [])
                    ),
                    "ADaM Datasets": ", ".join(
                        d if isinstance(d, str) else d.get("dataset", str(d))
                        for d in schema.get("adam_datasets_expected", [])
                    ),
                    "Confidence": schema.get("extraction_confidence", ""),
                })
            _PIPELINE_STATE["results"] = result_rows

            # Sync to Google Drive
            if gdrive_sync.is_configured():
                from pathlib import Path as _Path
                _kb_path = _Path(config_override["storage"]["base_path"])
                _PIPELINE_STATE["log"].append("☁️ Connecting to Google Drive…")
                try:
                    uploaded = gdrive_sync.upload_knowledge_base(_kb_path)
                    if uploaded:
                        _PIPELINE_STATE["log"].append("☁️ Knowledge base synced to Google Drive.")
                    else:
                        _PIPELINE_STATE["log"].append("⚠️ Google Drive upload returned False — check Streamlit logs for details.")
                except Exception as _gdrive_exc:
                    _PIPELINE_STATE["log"].append(f"⚠️ Google Drive sync exception: {_gdrive_exc}")

    except Exception:
        _PIPELINE_STATE["error"] = traceback.format_exc()
        _PIPELINE_STATE["msg"] = "❌ Pipeline failed — see error below."
        _PIPELINE_STATE["pct"] = 0
    finally:
        _PIPELINE_STATE["running"] = False


@st.fragment(run_every=3)
def _render_pipeline_status():
    """Pipeline status — fragment auto-refreshes every 3s while running."""
    import time as _time
    running = _PIPELINE_STATE["running"]
    pct = _PIPELINE_STATE["pct"]
    msg = _PIPELINE_STATE["msg"]
    if not (running or pct > 0 or msg):
        return

    log = list(_PIPELINE_STATE["log"])
    err = _PIPELINE_STATE["error"]
    start = _PIPELINE_STATE.get("start_time") or _time.time()
    elapsed = _time.time() - start
    mins, secs = divmod(int(elapsed), 60)
    timer_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"

    st.subheader("⚙️ Pipeline Status")

    if running:
        bar_val = max(pct / 100, 0.01)
        st.progress(bar_val, text=f"{msg}  ·  ⏱ {timer_str} elapsed")
        col_cap, col_stop = st.columns([5, 1])
        col_cap.caption("🔄 Refreshing every 3 seconds…")
        if col_stop.button("⛔ Stop", key="stop_pipeline"):
            _PIPELINE_STATE["stop_requested"] = True
    elif err:
        st.error("❌ Pipeline failed.")
        with st.expander("Error details", expanded=True):
            st.code(err)
    else:
        st.progress(pct / 100, text=f"✅ {msg}  ·  ⏱ Total: {timer_str}")

        # Results table
        results = _PIPELINE_STATE.get("results", [])
        if results:
            st.markdown("**SAPs processed this run:**")
            results_df = pd.DataFrame(results)
            st.dataframe(
                results_df,
                hide_index=True,
                width="stretch",
                column_config={
                    "NCT ID": st.column_config.TextColumn("NCT ID", width="small"),
                    "Title": st.column_config.TextColumn("Title", width="large"),
                    "TA": st.column_config.TextColumn("TA", width="small"),
                    "Phase": st.column_config.TextColumn("Phase", width="small"),
                    "Primary Endpoint": st.column_config.TextColumn("Primary Endpoint", width="small"),
                    "SDTM Domains": st.column_config.TextColumn("SDTM Domains"),
                    "ADaM Datasets": st.column_config.TextColumn("ADaM Datasets"),
                    "Confidence": st.column_config.TextColumn("Confidence", width="small"),
                },
            )

    if log:
        with st.expander("Pipeline log", expanded=running):
            for line in reversed(log):
                st.write(line)

    if not running and (pct == 100 or err):
        if st.button("Clear status", key="clear_pipeline"):
            _PIPELINE_STATE.update({"running": False, "pct": 0, "msg": "", "log": [], "error": None, "results": [], "stop_requested": False})
            st.rerun()


def _tab_pipeline():
    import threading
    from pipeline import step1_query  # noqa: imported for type reference only

    st.header("Pipeline — Knowledge Base Builder")
    st.markdown(
        "Query ClinicalTrials.gov for SAP documents, download PDFs, extract content "
        "with Claude, and populate the knowledge base — all from this tab."
    )

    # ------------------------------------------------------------------
    # Section 1: Knowledge Base Status
    # ------------------------------------------------------------------
    st.subheader("Knowledge Base Status")

    index_df = _load_master_index(CONFIG)

    total_saps = len(index_df)
    if total_saps > 0 and "therapeutic_area" in index_df.columns:
        unique_tas = index_df["therapeutic_area"].dropna().nunique()
    else:
        unique_tas = 0

    if total_saps > 0 and "extraction_date" in index_df.columns:
        last_run = index_df["extraction_date"].dropna().max()
        last_run = str(last_run) if last_run else "—"
    else:
        last_run = "—"

    kb_c1, kb_c2, kb_c3 = st.columns(3)
    kb_c1.metric("Total SAPs in Knowledge Base", total_saps)
    kb_c2.metric("Therapeutic Areas Covered", unique_tas)
    kb_c3.metric("Last Run Date", last_run)

    if total_saps > 0:
        # ── Filters ──────────────────────────────────────────────────────────
        f1, f2, f3 = st.columns(3)
        with f1:
            ta_options = sorted(index_df["therapeutic_area"].dropna().unique().tolist()) if "therapeutic_area" in index_df.columns else []
            ta_filter = st.multiselect("Filter by Therapeutic Area", options=ta_options, default=[])
        with f2:
            phase_options = sorted(index_df["phase"].dropna().unique().tolist()) if "phase" in index_df.columns else []
            phase_filter = st.multiselect("Filter by Phase", options=phase_options, default=[])
        with f3:
            conf_options = ["HIGH", "MEDIUM", "LOW"]
            conf_filter = st.multiselect("Filter by Confidence", options=conf_options, default=[])

        filtered_df = index_df.copy()
        if ta_filter:
            filtered_df = filtered_df[filtered_df["therapeutic_area"].isin(ta_filter)]
        if phase_filter:
            filtered_df = filtered_df[filtered_df["phase"].isin(phase_filter)]
        if conf_filter:
            filtered_df = filtered_df[filtered_df["extraction_confidence"].isin(conf_filter)]

        display_cols = [c for c in ["nct_id", "study_title", "therapeutic_area", "phase",
                                     "primary_endpoint_type", "extraction_confidence", "extraction_date"]
                        if c in filtered_df.columns]
        display_df = filtered_df[display_cols].copy() if display_cols else filtered_df.copy()

        # Keep original NCT IDs for display, add separate URL column
        if "nct_id" in display_df.columns:
            display_df.insert(1, "ct_url", display_df["nct_id"].apply(
                lambda x: f"https://clinicaltrials.gov/study/{x}" if pd.notna(x) else x
            ))

        st.caption(f"Showing {len(filtered_df)} of {total_saps} SAPs · Select rows to delete")
        selection = st.dataframe(
            display_df,
            width="stretch",
            hide_index=True,
            on_select="rerun",
            selection_mode="multi-row",
            column_config={
                "nct_id": st.column_config.TextColumn("NCT ID", width="small"),
                "ct_url": st.column_config.LinkColumn("↗", display_text="View", width="small"),
                "study_title": st.column_config.TextColumn("Study Title", width="large"),
                "therapeutic_area": st.column_config.TextColumn("Therapeutic Area"),
                "phase": st.column_config.TextColumn("Phase", width="small"),
                "primary_endpoint_type": st.column_config.TextColumn("Primary Endpoint", width="small"),
                "extraction_confidence": st.column_config.TextColumn("Confidence", width="small"),
                "extraction_date": st.column_config.TextColumn("Date", width="small"),
            },
        )

        selected_rows = selection.selection.rows if selection and selection.selection else []
        selected_nct_ids = display_df.iloc[selected_rows]["nct_id"].tolist() if selected_rows and "nct_id" in display_df.columns else []

        dl_col, del_col, _ = st.columns([1, 1, 3])
        with del_col:
            if selected_nct_ids:
                if st.button(f"🗑 Delete {len(selected_nct_ids)} selected", type="primary"):
                    storage_cfg = CONFIG["storage"]
                    base_path = Path(storage_cfg["base_path"])
                    index_path = base_path / storage_cfg["master_index_file"]
                    # Remove rows from CSV
                    new_df = index_df[~index_df["nct_id"].isin(selected_nct_ids)]
                    new_df.to_csv(index_path, index=False)
                    # Delete schema files
                    schemas_dir = base_path / "parsed_schemas"
                    for nct_id in selected_nct_ids:
                        schema_file = schemas_dir / f"{nct_id}.json"
                        if schema_file.exists():
                            schema_file.unlink()
                    # Sync to Drive
                    if gdrive_sync.is_configured():
                        gdrive_sync.upload_knowledge_base(base_path)
                    st.success(f"Deleted {len(selected_nct_ids)} SAP(s): {', '.join(selected_nct_ids)}")
                    st.rerun()
            else:
                st.button("🗑 Delete selected", disabled=True)
        with dl_col:
            storage_cfg = CONFIG["storage"]
            base_path = Path(storage_cfg["base_path"])
            index_path = base_path / storage_cfg["master_index_file"]
            try:
                csv_bytes = index_path.read_bytes()
                st.download_button(
                    label="⬇ Download master_index.csv",
                    data=csv_bytes,
                    file_name="master_index.csv",
                    mime="text/csv",
                )
            except FileNotFoundError:
                pass
    else:
        st.info("No SAPs in the knowledge base yet. Run the pipeline below to populate it.")

    st.divider()

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Section 2: Run Pipeline
    # ------------------------------------------------------------------
    st.subheader("Run Pipeline")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        st.warning(
            "ANTHROPIC_API_KEY not configured — Steps 4 and 4b will be skipped. "
            "Set the key in Streamlit secrets to enable Claude extraction."
        )

    with st.form("pipeline_form"):
        selected_tas = st.multiselect(
            "Therapeutic Areas to query",
            options=_TA_DISPLAY_OPTIONS,
            default=["Oncology", "CNS / Alzheimer"],
        )

        selected_phases = st.multiselect(
            "Study Phases",
            options=["PHASE1", "PHASE2", "PHASE3", "PHASE4"],
            default=["PHASE2", "PHASE3"],
        )

        selected_statuses = st.multiselect(
            "Study Status",
            options=["COMPLETED", "ACTIVE_NOT_RECRUITING", "TERMINATED"],
            default=["COMPLETED"],
        )

        col_n1, col_n2 = st.columns(2)
        with col_n1:
            max_per_ta = st.number_input(
                "Max SAPs per TA", min_value=1, max_value=20, value=3, step=1
            )
        with col_n2:
            max_total = st.number_input(
                "Max total SAPs", min_value=1, max_value=50, value=7, step=1
            )

        overwrite = st.checkbox(
            "Overwrite existing",
            value=False,
            help="Re-process NCT IDs already in the knowledge base",
        )

        run_clicked = st.form_submit_button(
            "▶ Run Pipeline",
            width="stretch",
            type="primary",
        )

    if run_clicked:
        if not selected_tas:
            st.error("Please select at least one Therapeutic Area.")
        elif not selected_phases:
            st.error("Please select at least one Study Phase.")
        elif not selected_statuses:
            st.error("Please select at least one Study Status.")
        elif _PIPELINE_STATE["running"]:
            st.warning("Pipeline is already running — wait for it to finish.")
        else:
            config_override = copy.deepcopy(CONFIG)
            query_terms = [_TA_QUERY_MAP[ta] for ta in selected_tas if ta in _TA_QUERY_MAP]
            config_override["filters"]["therapeutic_areas"] = query_terms
            config_override["filters"]["phases"] = selected_phases
            config_override["filters"]["study_status"] = selected_statuses
            config_override["sampling"]["max_per_ta"] = int(max_per_ta)
            config_override["sampling"]["max_total"] = int(max_total)
            config_override["storage"]["overwrite_existing"] = overwrite

            # Set running state BEFORE starting thread to avoid race condition
            import time as _time
            _PIPELINE_STATE.update({
                "running": True,
                "pct": 0,
                "msg": "Starting pipeline…",
                "log": ["Pipeline started."],
                "error": None,
                "start_time": _time.time(),
            })

            t = threading.Thread(
                target=_run_pipeline_thread,
                args=(config_override, api_key),
                daemon=False,
            )
            t.start()
            st.toast("Pipeline started!", icon="🚀")
            st.rerun()

    st.divider()

    # ── Backfill SAP excerpts (no API calls) ─────────────────────────────
    st.subheader("Backfill SAP Excerpts")
    st.markdown(
        "Populate `sap_section` and `sap_excerpt` fields on existing KB schemas "
        "using local text matching — **no API calls required**. "
        "Affects studies that have a text file in `extracted_text/`."
    )
    if st.button("Run Backfill (local, free)", key="backfill_btn"):
        storage_cfg = CONFIG["storage"]
        schemas_dir = Path(storage_cfg["base_path"]) / "parsed_schemas"
        text_dir = Path(storage_cfg["base_path"]) / storage_cfg.get("extracted_text", "extracted_text")
        schema_files = sorted(schemas_dir.glob("*.json")) if schemas_dir.exists() else []
        if not schema_files:
            st.warning("No schemas found in knowledge base.")
        else:
            results = []
            for sf in schema_files:
                nct_id = sf.stem
                text_file = text_dir / f"{nct_id}.txt"
                if not text_file.exists():
                    results.append(f"⏭ {nct_id} — no text file")
                    continue
                try:
                    with open(sf, encoding="utf-8") as fh:
                        schema = json.load(fh)
                    sap_text = text_file.read_text(encoding="utf-8")
                    open_qs = schema.get("open_questions") or []
                    updated = 0
                    for q in open_qs:
                        if q.get("sap_excerpt") and q.get("sap_section"):
                            continue
                        question = q.get("question") or q.get("category") or ""
                        passage = _find_sap_passage(sap_text, question)
                        if passage:
                            if not q.get("sap_excerpt"):
                                q["sap_excerpt"] = passage
                            if not q.get("sap_section"):
                                # Find nearest section header
                                import re as _re
                                sec_re = _re.compile(
                                    r"(?:^|\n)\s*(\d+(?:\.\d+)*\.?\s+[A-Z][^\n]{3,60})",
                                    _re.MULTILINE)
                                words = [w for w in question.lower().split() if len(w) > 5]
                                pos = sap_text.lower().find(words[0]) if words else 0
                                best = ""
                                for m in sec_re.finditer(sap_text):
                                    if m.start() <= pos:
                                        best = m.group(1).strip()
                                if best:
                                    q["sap_section"] = best
                            updated += 1
                    if updated:
                        with open(sf, "w", encoding="utf-8") as fh:
                            json.dump(schema, fh, ensure_ascii=False, indent=2)
                        results.append(f"✅ {nct_id} — {updated} question(s) updated")
                    else:
                        results.append(f"⏭ {nct_id} — already complete or no matches")
                except Exception as exc:
                    results.append(f"❌ {nct_id} — {exc}")
            st.success(f"Backfill complete: {len(schema_files)} schema(s) processed.")
            for r in results:
                st.markdown(f"- {r}")

    st.divider()
    _render_pipeline_status()


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    _render_sidebar()

    st.title("Standards Gate — CDISC Mapping Copilot")
    st.markdown(
        "_A biometrics knowledge layer that converts SAPs into structured delivery metadata — "
        "all outputs are **recommendations** that require human validation._"
    )

    tabs = st.tabs([
        "Input",
        "Knowledge Model",
        "SAP Summary",
        "SDTM Mapping",
        "ADaM Mapping",
        "Open Questions",
        "SAP Text",
        "Export",
        "Pipeline",
    ])

    with tabs[0]:
        _tab_input()

    schema = st.session_state.get("schema")

    # Pipeline tab always available (index 8)
    with tabs[8]:
        _tab_pipeline()

    if schema is None:
        for tab in tabs[1:8]:
            with tab:
                st.info(
                    "No SAP loaded. Use the **Input** tab to upload a PDF or look up a study.",
                    icon="👈",
                )
        return

    with tabs[1]:
        _tab_knowledge_model(schema)

    with tabs[2]:
        _tab_sap_summary(schema)

    with tabs[3]:
        _tab_sdtm(schema)

    with tabs[4]:
        _tab_adam(schema)

    with tabs[5]:
        _tab_open_questions(schema)

    with tabs[6]:
        _tab_sap_text(schema)

    with tabs[7]:
        _tab_export(schema)



if __name__ == "__main__":
    main()
