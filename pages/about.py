"""
StandardsGate — About Page

Explains the tool's purpose, methodology, and how to interpret results.
Emphasises the human-in-the-loop design philosophy.
"""

import streamlit as st

st.set_page_config(
    page_title="About — StandardsGate",
    page_icon="ℹ️",
    layout="wide",
)

st.title("About StandardsGate")
st.caption("CDISC Mapping Copilot · Component 2")

st.divider()

# ---------------------------------------------------------------------------
# Purpose
# ---------------------------------------------------------------------------
st.header("Purpose & Scope")
st.markdown(
    """
StandardsGate is an **AI-assisted review copilot** for clinical trial data standards planning.
It processes Statistical Analysis Plans (SAPs) and surfaces *recommendations* for SDTM domain
and ADaM dataset requirements — accelerating the study setup process while keeping the
biostatistician and CDISC programmer firmly in control.

**What StandardsGate does:**
- Extracts key SAP content (endpoints, analysis populations, statistical methods, special assessments)
  using **Claude (Anthropic)** for AI-assisted text analysis
- Applies a **second LLM call** to derive SDTM domain and ADaM dataset recommendations —
  using clinical reasoning, not deterministic rules, to align with CDISC standards
- Assigns per-dataset confidence scores based on how well the SAP specifies each component
- Surfaces open questions and ambiguities so they can be resolved before programming begins
- Produces exportable mapping packages for use in specification authoring and team handoff

**What StandardsGate does NOT do:**
- Auto-generate specifications or programming code
- Replace review by a qualified CDISC programmer or biostatistician
- Make final compliance decisions
- Guarantee completeness — the quality of recommendations depends on the quality of the SAP
"""
)

st.divider()

# ---------------------------------------------------------------------------
# Human-in-the-loop
# ---------------------------------------------------------------------------
st.header("Human-in-the-Loop Design")
st.markdown(
    """
Every output in StandardsGate is framed as a **recommendation, not a decision**. The tool is
designed to augment expert judgment, not replace it.

The language throughout is intentional:

| We say | We never say |
|---|---|
| Recommended | Generated |
| Detected | Auto-mapped |
| Review required | Approved |
| Confidence: 72% | Correct |

Before any recommendation from this tool is acted upon, it should be reviewed by:
1. A **biostatistician** familiar with the study protocol and SAP
2. A **CDISC programmer** with implementation experience in the relevant therapeutic area
3. Optionally, a **data standards lead** for consistency with sponsor standards

Confidence scores are signals, not verdicts. A score of 85% means the SAP provided rich
information for that dataset — it does not mean the recommendation is 85% likely to be correct.
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
- **Reference:** CDISC SDTM Implementation Guide v3.4
- Standard domains included in all Phase 2/3 studies: DM, EX, AE, MH, CM, VS, LB, DS, SV, SE
- Conditional domains triggered by detected study features (PK, tumor response, PRO, imaging, ECG, ophthalmology)
- Domain traceability: each conditional recommendation references the SAP section that triggered it
"""
    )

with col2:
    st.subheader("ADaM — Analysis Data Model")
    st.markdown(
        """
- **Version:** ADaM IG v2.1
- **References:** ADIMIG (general), ADTTEG (time-to-event), OCCDS (occurrence data)
- Standard datasets in all studies: ADSL, ADAE
- Conditional datasets based on endpoint types: ADTTE (TTE), ADEFF (CFB), ADQS (PRO),
  ADPC/ADPP (PK), ADRS (tumor response), ADBM (biomarkers), ADEG (ECG)
- Per-dataset confidence scoring accounts for: population definitions, method specification,
  covariate listing, visit windowing, and open questions
"""
    )

st.divider()

# ---------------------------------------------------------------------------
# Processing pipeline
# ---------------------------------------------------------------------------
st.header("How It Works")
st.markdown(
    """
**Two input modes:**

1. **PDF Upload** — Upload a SAP PDF directly. The tool extracts text using pdfplumber
   (with pypdf fallback for compatibility), then sends it to Claude (Anthropic) for
   structured extraction. The document must be text-based; scanned/image PDFs will
   yield limited results.

2. **NCT ID Lookup** — If a study has been processed by the StandardsGate pipeline
   (Component 1), its parsed schema can be loaded from the local knowledge base instantly,
   without repeating the AI extraction step.

**Processing stages:**

```
PDF text extraction  →  AI SAP parsing  →  AI CDISC recommendations  →  Display
    (pdfplumber/pypdf)    (Claude: extract    (Claude: derive SDTM        (this UI)
                           structured schema)   domains + ADaM datasets)
```

Two separate Claude API calls: the first extracts a structured schema from the SAP text;
the second reasons over that schema to produce SDTM/ADaM recommendations with clinical rationale.

**Confidence scoring** reflects how well the SAP specifies each component — assessed by the
recommendation LLM based on available SAP content:
- **≥80% (green):** SAP explicitly described the key design elements for this dataset
- **60–79% (yellow):** Most elements present; minor inference was required
- **<60% (red):** Significant inference required; SAP was vague on key aspects — flag for discussion
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
- **Very long SAPs** (>120,000 characters) are truncated before AI extraction; the tail of
  the document (often appendices and shells) may not be reflected.
- **Non-standard endpoint types** may not be fully recognised by the AI; always review
  the Detected Special Assessments section carefully.
- **Extraction quality** depends on SAP structure and language clarity. Poorly structured
  or non-standard SAPs will yield lower confidence scores.
- The tool does not currently validate against sponsor-specific standards or controlled terminology.
"""
)

st.divider()

st.caption(
    "StandardsGate is a decision-support tool. All recommendations require validation "
    "by qualified clinical data standards professionals. CDISC SDTM IG v3.4 and "
    "ADaM IG v2.1 are the authoritative references for implementation decisions."
)
