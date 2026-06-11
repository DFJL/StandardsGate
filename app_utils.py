"""
app_utils.py — Helper functions for the Standards Gate Streamlit UI (Component 2).

These utilities bridge the pipeline modules (step3, step4, rule_engine) to the
display layer.  None of the pipeline source files are modified; we import and
call them directly.
"""

import csv
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# SDTM domain metadata
# Used to augment the raw rule-engine output with a human-readable description.
# ---------------------------------------------------------------------------

_SDTM_DOMAIN_DESCRIPTIONS = {
    "DM": "Demographics",
    "EX": "Exposure",
    "AE": "Adverse Events",
    "MH": "Medical History",
    "CM": "Concomitant/Prior Medications",
    "VS": "Vital Signs",
    "LB": "Laboratory Test Results",
    "DS": "Disposition",
    "SV": "Subject Visits",
    "SE": "Subject Elements",
    "PC": "Pharmacokinetics Concentrations",
    "PP": "Pharmacokinetics Parameters",
    "RS": "Disease Response",
    "TU": "Tumor/Lesion Identification",
    "QS": "Questionnaires",
    "MI": "Microscopic Findings",
    "EG": "ECG Test Results",
    "OE": "Ophthalmic Examinations",
}

# ADaM dataset descriptions and standard key variables / dependencies
_ADAM_DATASET_META = {
    "ADSL": {
        "description": "Subject-Level Analysis Dataset",
        "key_vars": "USUBJID, STUDYID, SITEID, ARM, ACTARM, TRT01P, TRT01A, SAFFL, ITTFL, RANDFL, AGE, SEX, RACE",
        "depends_on": "DM, DS, EX, randomization data",
    },
    "ADAE": {
        "description": "Adverse Events Analysis Dataset",
        "key_vars": "USUBJID, AEDECOD, AEBODSYS, AESEV, AESER, AESTDTC, AEENDTC, TRTEMFL, AETOXGR",
        "depends_on": "ADSL, AE",
    },
    "ADTTE": {
        "description": "Time-to-Event Analysis Dataset",
        "key_vars": "USUBJID, PARAMCD, PARAM, AVAL, CNSR, STARTDT, ADT, EVNTDESC, CNSDTDSC",
        "depends_on": "ADSL, DS, AE, RS (if applicable)",
    },
    "ADEFF": {
        "description": "Efficacy Analysis Dataset (Change from Baseline)",
        "key_vars": "USUBJID, PARAMCD, PARAM, AVISIT, AVISITN, ADT, AVAL, BASE, CHG, PCHG, ANL01FL",
        "depends_on": "ADSL, primary efficacy SDTM domain (e.g. QS, LB, RS)",
    },
    "ADQS": {
        "description": "Questionnaires Analysis Dataset",
        "key_vars": "USUBJID, PARAMCD, PARAM, AVISIT, AVISITN, AVAL, BASE, CHG, QSCAT, QSSCAT",
        "depends_on": "ADSL, QS",
    },
    "ADPC": {
        "description": "Pharmacokinetics Concentrations Analysis Dataset",
        "key_vars": "USUBJID, PARAMCD, PARAM, AVAL, ARRLT, ANRRLT, ATPT, ATPTN, ATPTREF",
        "depends_on": "ADSL, PC",
    },
    "ADPP": {
        "description": "Pharmacokinetics Parameters Analysis Dataset",
        "key_vars": "USUBJID, PARAMCD, PARAM, AVAL, PPSPEC, PPREASND",
        "depends_on": "ADSL, PP, ADPC",
    },
    "ADRS": {
        "description": "Disease Response Analysis Dataset",
        "key_vars": "USUBJID, PARAMCD, PARAM, AVAL, AVALC, ADT, AVISIT, AVISITN, RSEVAL",
        "depends_on": "ADSL, RS, TU",
    },
    "ADBM": {
        "description": "Biomarker Analysis Dataset",
        "key_vars": "USUBJID, PARAMCD, PARAM, AVISIT, AVAL, BASE, CHG, PCHG, LBSPEC",
        "depends_on": "ADSL, LB",
    },
    "ADEG": {
        "description": "ECG Analysis Dataset",
        "key_vars": "USUBJID, PARAMCD, PARAM, AVISIT, AVAL, BASE, CHG, EGSTRESC",
        "depends_on": "ADSL, EG",
    },
}


# ---------------------------------------------------------------------------
# Confidence color helper
# ---------------------------------------------------------------------------

def get_confidence_color(score) -> str:
    """
    Return a CSS hex color string for a given confidence score (0-100).

    - GREEN  (#2E7D32):  score >= 80
    - YELLOW (#F57F17):  60 <= score < 80
    - RED    (#C62828):  score < 60 or None
    """
    if score is None:
        return "#9E9E9E"   # grey — not applicable
    try:
        score = int(score)
    except (TypeError, ValueError):
        return "#9E9E9E"

    if score >= 80:
        return "#2E7D32"   # green
    elif score >= 60:
        return "#F57F17"   # amber/yellow
    else:
        return "#C62828"   # red


def get_confidence_label(score) -> str:
    """Return an evidence label for a confidence score — interpretable by senior biometricians."""
    if score is None:
        return "Missing SAP Support"
    try:
        score = int(score)
    except (TypeError, ValueError):
        return "Missing SAP Support"
    if score >= 80:
        return "Direct SAP Support"
    elif score >= 60:
        return "Inferred from Standards"
    else:
        return "Limited SAP Evidence"


# ---------------------------------------------------------------------------
# Processing functions
# ---------------------------------------------------------------------------

def process_uploaded_pdf(uploaded_file, config: dict) -> dict:
    """
    Accept a Streamlit UploadedFile, save it to a temp location, run the
    step3 text extraction → step4 Claude parse → rule_engine pipeline, and
    return the fully populated schema dict.

    Raises RuntimeError on failure with a human-readable message.
    """
    from pipeline import step3_extract, step4_parse, rule_engine

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Set it in your environment before "
            "uploading a PDF for AI-assisted extraction."
        )

    # Write uploaded bytes to a temporary PDF file
    suffix = Path(uploaded_file.name).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_pdf:
        tmp_pdf.write(uploaded_file.read())
        tmp_pdf_path = Path(tmp_pdf.name)

    try:
        # --- Step 3: extract text ---
        with tempfile.TemporaryDirectory() as tmp_text_dir:
            text_dir = Path(tmp_text_dir)
            nct_id = "UPLOAD_001"

            txt_path = step3_extract.extract_study_text(
                config=config,
                nct_id=nct_id,
                pdf_path=tmp_pdf_path,
                text_dir=text_dir,
            )
            if txt_path is None:
                raise RuntimeError(
                    "Text extraction failed. The PDF may be image-based or corrupted. "
                    "Please try a text-based PDF."
                )

            # --- Step 4: parse with Claude ---
            with tempfile.TemporaryDirectory() as tmp_schema_dir:
                schema_dir = Path(tmp_schema_dir)
                meta = {
                    "nct_id": nct_id,
                    "study_title": uploaded_file.name,
                    "original_filename": uploaded_file.name,
                }
                schema = step4_parse.parse_study_sap(
                    config=config,
                    nct_id=nct_id,
                    text_path=txt_path,
                    schema_dir=schema_dir,
                    meta=meta,
                )
                if schema is None:
                    raise RuntimeError(
                        "Claude parsing failed. Check the extraction_notes field "
                        "or verify the document is a Statistical Analysis Plan."
                    )

                # --- Rule engine: derive SDTM / ADaM recommendations ---
                schema = rule_engine.run_rules(schema, config)
                return schema
    finally:
        try:
            tmp_pdf_path.unlink()
        except Exception:
            pass


def lookup_from_index(nct_id: str, config: dict) -> dict:
    """
    Attempt to load a previously-processed SAP schema for the given NCT ID.

    Looks for:
      1. <base_path>/parsed_schemas/<NCT_ID>.json  (direct match)
      2. <base_path>/parsed_schemas/<NCT_ID>_SAP_001.json  (pipeline naming convention)

    If found, loads the schema and re-runs the rule engine (idempotent).
    If not found, raises a FileNotFoundError with a helpful message.
    """
    from pipeline import rule_engine

    storage_cfg = config["storage"]
    base_path = Path(storage_cfg["base_path"])
    schema_dir = base_path / storage_cfg["subdirs"]["parsed_schemas"]

    # Try both naming conventions
    candidates = [
        schema_dir / f"{nct_id}.json",
        schema_dir / f"{nct_id}_SAP_001.json",
    ]

    schema_path = None
    for candidate in candidates:
        if candidate.exists():
            schema_path = candidate
            break

    if schema_path is None:
        # Fallback: reconstruct a partial schema from master_index.csv
        schema = _schema_from_index_row(nct_id, base_path, storage_cfg)
        if schema is not None:
            schema = rule_engine.run_rules(schema, config)
            return schema
        searched = [str(c) for c in candidates]
        raise FileNotFoundError(
            f"No processed schema found for NCT ID '{nct_id}'. "
            f"Run the pipeline first to populate the knowledge base. "
            f"Searched: {', '.join(searched)}"
        )

    try:
        with open(schema_path, encoding="utf-8") as fh:
            schema = json.load(fh)
    except Exception as exc:
        raise RuntimeError(f"Failed to read schema file {schema_path}: {exc}")

    # Re-run rule engine to ensure SDTM/ADaM recommendations are current
    schema = rule_engine.run_rules(schema, config)
    return schema


def _schema_from_index_row(nct_id: str, base_path: Path, storage_cfg: dict) -> Optional[dict]:
    """
    Reconstruct a minimal canonical schema from a master_index.csv row.
    Used as fallback when the full JSON schema file is not available locally.
    """
    import pandas as pd
    index_path = base_path / storage_cfg["master_index_file"]
    if not index_path.exists():
        return None

    try:
        df = pd.read_csv(index_path, dtype=str).fillna("")
        row = df[df["nct_id"] == nct_id]
        if row.empty:
            return None
        r = row.iloc[0].to_dict()
    except Exception:
        return None

    def _bool(v):
        return str(v).lower() in ("true", "1", "yes")

    primary_ep = {}
    if r.get("primary_endpoint_type"):
        primary_ep = {
            "description": r.get("primary_endpoint_desc", ""),
            "type": r.get("primary_endpoint_type", "OTHER"),
            "timepoint": None,
            "population": None,
            "sap_section": None,
        }

    pops = []
    for abbr in str(r.get("analysis_populations", "")).split("|"):
        abbr = abbr.strip()
        if abbr:
            pops.append({"name": abbr, "abbreviation": abbr, "definition": None, "sap_section": None})

    schema = {
        "metadata": {
            "nct_id": nct_id,
            "study_title": r.get("study_title", ""),
            "sponsor": r.get("sponsor", ""),
            "therapeutic_area": r.get("therapeutic_area", ""),
            "indication": r.get("indication", ""),
            "phase": r.get("phase", ""),
            "study_design": r.get("study_design", ""),
            "source_document": r.get("source_pdf", ""),
            "extraction_date": r.get("extraction_date", ""),
        },
        "endpoints": {
            "primary": [primary_ep] if primary_ep else [],
            "secondary": [],
            "exploratory": [],
        },
        "analysis_populations": pops,
        "special_assessments": {
            "pharmacokinetics": _bool(r.get("has_pk")),
            "tumor_response": _bool(r.get("has_tumor_response")),
            "patient_reported_outcomes": _bool(r.get("has_pro")),
            "biomarkers": False,
            "imaging": _bool(r.get("has_imaging")),
            "ecg": _bool(r.get("has_ecg")),
            "ophthalmology": False,
            "non_standard_endpoints": [],
        },
        "statistical_methods": {
            "primary_analysis_method": None,
            "covariates": None,
            "visit_windowing": None,
            "multiplicity_adjustment": _bool(r.get("multiplicity_adjustment")),
            "missing_data_handling": None,
            "estimand_framework": _bool(r.get("estimand_framework")),
            "bayesian_elements": _bool(r.get("bayesian")),
            "sap_section": None,
        },
        "open_questions": [],
        "flags": [],
        "sap_section_refs": {},
        "sdtm_domains_expected": [],
        "adam_datasets_expected": [],
        "component_confidence": {},
        "extraction_confidence": r.get("extraction_confidence", "LOW"),
        "extraction_notes": "Schema reconstructed from master_index.csv — full JSON not available locally. Re-run pipeline to restore complete schema.",
        "non_standard_approaches": [],
    }
    return schema


# ---------------------------------------------------------------------------
# Schema → Display DataFrame converters
# ---------------------------------------------------------------------------

def schema_to_sdtm_table(schema: dict) -> pd.DataFrame:
    """
    Convert the rule-engine SDTM output to a display-ready DataFrame.

    Columns: Domain | Description | Rationale | SAP Section | Confidence | Always Present
    """
    domains = schema.get("sdtm_domains_expected", [])
    component_conf = schema.get("component_confidence", {})

    rows = []
    for d in domains:
        domain_code = d.get("domain", "")
        conf_score = component_conf.get(domain_code)
        rows.append({
            "Domain": domain_code,
            "Description": _SDTM_DOMAIN_DESCRIPTIONS.get(domain_code, "—"),
            "Rationale": d.get("rationale", ""),
            "SAP Section": d.get("sap_section") or "—",
            "Confidence": conf_score,
            "Always Present": d.get("always_present", False),
        })

    if not rows:
        return pd.DataFrame(columns=["Domain", "Description", "Rationale", "SAP Section", "Confidence", "Always Present"])

    return pd.DataFrame(rows)


def schema_to_adam_table(schema: dict) -> pd.DataFrame:
    """
    Convert the rule-engine ADaM output to a display-ready DataFrame.

    Columns: Dataset | Description | Key Variables | Depends On | SAP Section | Confidence
    """
    datasets = schema.get("adam_datasets_expected", [])
    component_conf = schema.get("component_confidence", {})

    rows = []
    for d in datasets:
        ds_name = d.get("dataset", "")
        meta = _ADAM_DATASET_META.get(ds_name, {})
        conf_score = component_conf.get(ds_name)
        rows.append({
            "Dataset": ds_name,
            "Description": meta.get("description", "—"),
            "Key Variables": meta.get("key_vars", "—"),
            "Depends On": meta.get("depends_on", "—"),
            "SAP Section": d.get("sap_section") or "—",
            "Confidence Score": conf_score,
            "Confidence Label": get_confidence_label(conf_score),
        })

    if not rows:
        return pd.DataFrame(columns=[
            "Dataset", "Description", "Key Variables", "Depends On",
            "SAP Section", "Confidence Score", "Confidence Label",
        ])

    return pd.DataFrame(rows)


def schema_to_open_questions_df(schema: dict) -> pd.DataFrame:
    """
    Convert the open_questions list to a display-ready DataFrame.

    Columns: Severity | Category | Question
    Sorted by severity: HIGH → MEDIUM → LOW
    """
    oqs = schema.get("open_questions", [])

    severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}

    rows = []
    for q in oqs:
        rows.append({
            "Severity": (q.get("severity") or "MEDIUM").upper(),
            "Category": q.get("category") or "General",
            "Question": q.get("question") or str(q),
        })

    if not rows:
        return pd.DataFrame(columns=["Severity", "Category", "Question"])

    df = pd.DataFrame(rows)
    df["_sort"] = df["Severity"].map(lambda s: severity_order.get(s, 99))
    df = df.sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)
    return df


def schema_to_export_csv(schema: dict) -> str:
    """
    Flatten the full schema into a CSV-formatted string suitable for download.

    Returns the CSV text as a string (UTF-8).
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # ---- Metadata section ----
    writer.writerow(["## METADATA"])
    writer.writerow(["Field", "Value"])
    meta = schema.get("metadata", {})
    for k, v in meta.items():
        writer.writerow([k, v])
    writer.writerow([])

    # ---- Endpoints section ----
    writer.writerow(["## ENDPOINTS"])
    writer.writerow(["Type", "Description", "Timepoint", "Endpoint Type"])
    eps = schema.get("endpoints", {})
    for ep_type, ep_list in [("Primary", eps.get("primary", [])),
                              ("Secondary", eps.get("secondary", [])),
                              ("Exploratory", eps.get("exploratory", []))]:
        for ep in ep_list:
            writer.writerow([
                ep_type,
                ep.get("description", ""),
                ep.get("timepoint", ""),
                ep.get("type", ""),
            ])
    writer.writerow([])

    # ---- Analysis Populations ----
    writer.writerow(["## ANALYSIS POPULATIONS"])
    writer.writerow(["Name", "Abbreviation", "Definition"])
    for pop in schema.get("analysis_populations", []):
        writer.writerow([
            pop.get("name", ""),
            pop.get("abbreviation", ""),
            pop.get("definition", ""),
        ])
    writer.writerow([])

    # ---- SDTM Domains ----
    writer.writerow(["## RECOMMENDED SDTM DOMAINS"])
    writer.writerow(["Domain", "Description", "Rationale", "SAP Section", "Always Present"])
    for d in schema.get("sdtm_domains_expected", []):
        writer.writerow([
            d.get("domain", ""),
            _SDTM_DOMAIN_DESCRIPTIONS.get(d.get("domain", ""), ""),
            d.get("rationale", ""),
            d.get("sap_section") or "",
            d.get("always_present", False),
        ])
    writer.writerow([])

    # ---- ADaM Datasets ----
    writer.writerow(["## RECOMMENDED ADAM DATASETS"])
    writer.writerow(["Dataset", "Description", "Key Variables", "Depends On", "SAP Section", "Confidence"])
    comp_conf = schema.get("component_confidence", {})
    for d in schema.get("adam_datasets_expected", []):
        ds_name = d.get("dataset", "")
        meta_ds = _ADAM_DATASET_META.get(ds_name, {})
        writer.writerow([
            ds_name,
            meta_ds.get("description", ""),
            meta_ds.get("key_vars", ""),
            meta_ds.get("depends_on", ""),
            d.get("sap_section") or "",
            comp_conf.get(ds_name, ""),
        ])
    writer.writerow([])

    # ---- Open Questions ----
    writer.writerow(["## OPEN QUESTIONS & FLAGS"])
    writer.writerow(["Severity", "Category", "Question"])
    for q in schema.get("open_questions", []):
        writer.writerow([
            (q.get("severity") or "MEDIUM").upper(),
            q.get("category") or "",
            q.get("question") or str(q),
        ])

    return output.getvalue()
