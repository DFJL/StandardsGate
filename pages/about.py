"""
Study Standards Lens — About Page
"""

import streamlit as st

st.set_page_config(
    page_title="About — Study Standards Lens",
    page_icon="🔬",
    layout="wide",
)

st.title("About Study Standards Lens")
st.caption("AI-Assisted Review for SAP Completeness and CDISC Alignment · From Draft to Final")

st.divider()

# ---------------------------------------------------------------------------
# Purpose
# ---------------------------------------------------------------------------
st.header("What Is Study Standards Lens?")
st.markdown(
    """
**Study Standards Lens** is an AI-powered copilot that reviews Statistical Analysis Plans (SAPs)
for completeness and CDISC alignment — at any stage of the study lifecycle, from first draft
through protocol amendment to final sign-off.

It reads a SAP and surfaces:
- **Completeness flags** — ambiguities, missing specifications, and gaps that must be resolved
  before CDISC implementation can begin (reducing costly rework cycles)
- **SDTM domain alignment** — which domains the study design calls for, with rationale traceable
  back to specific SAP content
- **ADaM dataset alignment** — inferred from detected endpoints, analysis populations, and
  special assessments, with per-dataset confidence scores
- **An interactive knowledge model** — a visual map of how study design elements connect to
  CDISC deliverables

The goal is to shift CDISC alignment review earlier in the study lifecycle, so that
standards gaps are caught while the SAP is still being written — not after programming has started.

**Study Standards Lens does NOT:**
- Auto-generate specifications or programming code
- Replace review by a qualified CDISC programmer or biostatistician
- Make final compliance or submission decisions
- Guarantee completeness — AI extraction quality depends on SAP structure and clarity
"""
)

st.divider()

# ---------------------------------------------------------------------------
# Human-in-the-loop
# ---------------------------------------------------------------------------
st.header("Human-in-the-Loop Design")
st.markdown(
    """
Every output is framed as a **recommendation, not a decision**. The tool is designed to
augment expert judgement, not replace it.

| We say | We never say |
|---|---|
| Recommended | Generated |
| Detected | Auto-mapped |
| Review required | Approved |
| Confidence: 72% | Correct |

Before any recommendation is acted upon, it should be reviewed by:
1. A **biostatistician** familiar with the study protocol and SAP
2. A **CDISC programmer** with implementation experience in the relevant therapeutic area
3. Optionally, a **data standards lead** for consistency with sponsor standards

Confidence scores are *signals*, not verdicts. A score of 85% means the SAP provided rich
information for that dataset — it does not mean the recommendation is 85% likely to be correct.
"""
)

st.divider()

# ---------------------------------------------------------------------------
# Lifecycle framing
# ---------------------------------------------------------------------------
st.header("Designed for the Full Study Lifecycle")
st.markdown(
    """
Standards alignment is not a one-time activity. Study Standards Lens is designed to add
value at every stage:

| Stage | How to use Study Standards Lens |
|---|---|
| **SAP Draft** | Catch specification gaps early — before the team invests in programming |
| **SAP Review** | Surface open questions for the study team to resolve prior to sign-off |
| **Protocol Amendment** | Re-run on updated SAP to identify new or changed CDISC requirements |
| **Pre-submission** | Cross-check expected domains and datasets against final SAP |

Re-running the tool as the SAP evolves provides a version-aware view of completeness over time.
"""
)

st.divider()

# ---------------------------------------------------------------------------
# Standards
# ---------------------------------------------------------------------------
st.header("CDISC Standards Referenced")

col1, col2 = st.columns(2)

with col1:
    st.subheader("SDTM — Study Data Tabulation Model")
    st.markdown(
        """
- **Version:** SDTM IG v3.4
- Standard domains expected in all Phase 2/3 studies: DM, EX, AE, MH, CM, VS, LB, DS, SV, SE
- Conditional domains triggered by detected study features: PK, tumor response, PRO, imaging, ECG, ophthalmology
- Each conditional recommendation is traceable to the SAP section that triggered it
"""
    )

with col2:
    st.subheader("ADaM — Analysis Data Model")
    st.markdown(
        """
- **Version:** ADaM IG v2.1
- **References:** ADIMIG (general), ADTTEG (time-to-event), OCCDS (occurrence data)
- Standard datasets in all studies: ADSL, ADAE
- Conditional datasets based on endpoint types: ADTTE, ADEFF, ADQS, ADPC/ADPP, ADRS, ADBM, ADEG
- Per-dataset confidence scoring reflects: population definitions, method specification,
  covariate listing, visit windowing, and open questions
"""
    )

st.divider()

# ---------------------------------------------------------------------------
# How it works
# ---------------------------------------------------------------------------
st.header("How It Works")
st.markdown(
    """
**Two input modes:**

1. **Upload a SAP PDF** — The tool extracts text (pdfplumber, with pypdf fallback), then calls
   Claude (Anthropic) to extract a structured schema. Best for draft SAPs not yet in the knowledge base.

2. **Load from Knowledge Base** — Studies pre-processed by the pipeline are available instantly,
   without repeating the AI extraction step.

**Processing stages:**

```
PDF text extraction  →  AI SAP parsing  →  AI CDISC recommendations  →  Review UI
    (pdfplumber)         (Claude: extract      (Claude: derive SDTM/        (this app)
                          structured schema)    ADaM recommendations)
```

Two separate Claude API calls: the first extracts structured schema from the SAP text;
the second reasons over that schema to produce SDTM/ADaM recommendations with clinical rationale.

**Confidence scoring** reflects how well the SAP specifies each component:
- **≥80% (green):** SAP explicitly described the key design elements for this dataset
- **60–79% (yellow):** Most elements present; minor inference was required
- **<60% (red):** Significant inference required; flag for discussion before programming
"""
)

st.divider()

# ---------------------------------------------------------------------------
# Limitations
# ---------------------------------------------------------------------------
st.header("Known Limitations")
st.markdown(
    """
- **Image-based PDFs** (scanned documents) cannot be processed. Use a text-based PDF.
- **Very long SAPs** (>400,000 characters) use a head+tail extraction strategy; the middle
  of the document may not be reflected in the analysis.
- **Non-standard endpoint types** may not be fully recognised; always review the
  Detected Special Assessments section carefully.
- **Extraction quality** depends on SAP structure and language clarity. Poorly structured
  or non-standard SAPs yield lower confidence scores.
- The tool does not validate against sponsor-specific standards or controlled terminology.
- **Open questions flagged as truncation artifacts** (marked ⚠️) may be false positives
  caused by the document length limit — verify against the full SAP text.
"""
)

st.divider()

st.caption(
    "Study Standards Lens is a decision-support tool. All recommendations require validation "
    "by qualified clinical data standards professionals. CDISC SDTM IG v3.4 and "
    "ADaM IG v2.1 are the authoritative references for implementation decisions."
)
