"""
Rule Engine: Deterministic SDTM/ADaM derivation from extracted SAP parameters.

This module is intentionally separate from the LLM extraction step. By keeping
the CDISC domain/dataset recommendations in a deterministic rule engine:
  1. Recommendations are auditable and reproducible
  2. Each recommendation is traceable to the SAP section that triggered it
  3. Biostatisticians can inspect and override individual rules
  4. No hallucination risk for the standards mapping layer

Rule logic follows CDISC SDTM Implementation Guide (SDTMIG) and ADaM
Implementation Guides (ADIMIG, ADTTEG, etc.) conventions.

Confidence scoring:
  - Per-dataset confidence reflects how much information was available in the
    SAP to correctly implement that dataset
  - 90-100: SAP explicitly described all key design elements for this dataset
  - 70-89:  Most elements present; minor inference required
  - 50-69:  Significant inference; SAP was vague on key aspects
  - 0-49:   Very little SAP guidance; high implementation risk
"""

from loguru import logger
from typing import Optional


# ---------------------------------------------------------------------------
# SDTM domain rules
# ---------------------------------------------------------------------------

# These domains are ALWAYS expected regardless of study type
_ALWAYS_SDTM_DOMAINS = ["DM", "EX", "AE", "MH", "CM"]

# Mapping from SAP feature → additional SDTM domains it implies
# Each entry is (domain_code, rationale, sap_feature_key)
_CONDITIONAL_SDTM_RULES = [
    # Pharmacokinetics → concentration and parameter domains
    ("PC", "PK sampling present → plasma concentration data", "pharmacokinetics"),
    ("PP", "PK sampling present → PK parameters derived", "pharmacokinetics"),
    # Tumor response → response/tumor domains
    ("RS", "Tumor response assessment present → response domain", "tumor_response"),
    ("TU", "Tumor response assessment present → tumor measurability domain", "tumor_response"),
    # Patient-reported outcomes → questionnaire response domain
    ("QS", "PRO/questionnaire assessments present → QS domain", "patient_reported_outcomes"),
    # Imaging assessments → findings domain
    ("MI", "Imaging assessments present → MI domain", "imaging"),
    # ECG assessments
    ("EG", "ECG assessments present → EG domain", "ecg"),
    # Biomarkers → lab results domain (LB covers most biomarkers)
    ("LB", "Biomarkers present → LB domain for laboratory results", "biomarkers"),
    # Ophthalmology
    ("OE", "Ophthalmology assessments present → OE domain", "ophthalmology"),
]

# Domains almost always present in Phase 2/3 trials (not in _ALWAYS but nearly universal)
_STANDARD_ADDITIONAL_SDTM = ["VS", "LB", "DS", "SV", "SE"]


# ---------------------------------------------------------------------------
# ADaM dataset rules
# ---------------------------------------------------------------------------

# These ADaM datasets are ALWAYS expected
_ALWAYS_ADAM_DATASETS = ["ADSL", "ADAE"]

# Mapping from SAP feature → ADaM dataset(s) implied
# Each entry is (dataset_name, rationale, sap_feature_key_or_endpoint_type)
_CONDITIONAL_ADAM_RULES = [
    # Time-to-event endpoints → ADTTE
    ("ADTTE", "Time-to-event endpoint present → ADTTE dataset", "endpoint_tte"),
    # Change-from-baseline endpoints → ADEFF (efficacy)
    ("ADEFF", "Change-from-baseline endpoint present → ADEFF dataset", "endpoint_cfb"),
    # PRO/questionnaire endpoints → ADQS
    ("ADQS", "Patient-reported outcomes present → ADQS dataset", "patient_reported_outcomes"),
    # Pharmacokinetics
    ("ADPC", "PK sampling present → ADPC dataset", "pharmacokinetics"),
    ("ADPP", "PK sampling present → ADPP (PK parameters) dataset", "pharmacokinetics"),
    # Tumor response
    ("ADRS", "Tumor response assessment present → ADRS dataset", "tumor_response"),
    # Biomarkers
    ("ADBM", "Biomarker assessments present → ADBM dataset", "biomarkers"),
    # ECG
    ("ADEG", "ECG assessments present → ADEG dataset", "ecg"),
]


# ---------------------------------------------------------------------------
# Confidence scoring helpers
# ---------------------------------------------------------------------------

def _score_adsl_confidence(schema: dict) -> int:
    """
    Score confidence for ADSL implementation.

    ADSL confidence depends on:
    - Population definitions being clearly specified
    - Randomization/stratification factors described
    - Treatment assignment information available
    """
    score = 50  # baseline

    pops = schema.get("analysis_populations", [])
    if len(pops) >= 2:
        score += 20  # multiple populations well-defined
    elif len(pops) == 1:
        score += 10

    # Check for ITT/randomized population (essential for ADSL)
    pop_abbrevs = [p.get("abbreviation", "").upper() for p in pops]
    pop_names = [p.get("name", "").lower() for p in pops]
    has_randomized = any(
        a in ["ITT", "FAS", "MITT", "RAND"] for a in pop_abbrevs
    ) or any("randomiz" in n or "intent" in n for n in pop_names)

    if has_randomized:
        score += 15

    # If populations have definitions (not just names), higher confidence
    with_definitions = sum(1 for p in pops if p.get("definition"))
    if with_definitions == len(pops) and len(pops) > 0:
        score += 10

    # Section references improve confidence (SAP was well-structured)
    if schema.get("sap_section_refs", {}).get("analysis_populations"):
        score += 5

    return min(score, 100)


def _score_adae_confidence(schema: dict) -> int:
    """
    Score confidence for ADAE implementation.

    ADAE is relatively standard; lower confidence if safety population
    is not defined or AE grading criteria not mentioned.
    """
    score = 60  # baseline — AE analysis is well-standardized

    # Safety population defined?
    pops = schema.get("analysis_populations", [])
    pop_abbrevs = [p.get("abbreviation", "").upper() for p in pops]
    pop_names = [p.get("name", "").lower() for p in pops]
    has_safety = any(a in ["SAF", "SAFETY", "SS"] for a in pop_abbrevs) or any(
        "safety" in n for n in pop_names
    )
    if has_safety:
        score += 20

    # If AE section is referenced
    if schema.get("sap_section_refs", {}).get("safety_analyses"):
        score += 10

    # Flags about AE handling issues lower confidence
    flags = schema.get("flags", [])
    if any("adverse" in f.lower() or "safety" in f.lower() for f in flags):
        score -= 10

    return min(max(score, 0), 100)


def _score_adtte_confidence(schema: dict) -> int:
    """
    Score confidence for ADTTE implementation.

    High confidence requires:
    - TTE endpoint clearly defined with timepoint
    - Analysis method (Cox PH or KM) specified
    - Censoring rules described
    """
    primary_eps = schema.get("endpoints", {}).get("primary", [])
    tte_eps = [ep for ep in primary_eps if ep.get("type") == "TTE"]

    if not tte_eps:
        # Check secondary endpoints
        secondary_eps = schema.get("endpoints", {}).get("secondary", [])
        tte_eps = [ep for ep in secondary_eps if ep.get("type") == "TTE"]

    if not tte_eps:
        return None  # Not applicable

    score = 40  # baseline for TTE

    # Endpoint has description
    if tte_eps[0].get("description"):
        score += 15

    # Endpoint has timepoint
    if tte_eps[0].get("timepoint"):
        score += 10

    # Analysis method is Cox or KM
    method = (
        schema.get("statistical_methods", {}).get("primary_analysis_method", "") or ""
    ).lower()
    if "cox" in method or "kaplan" in method or "log-rank" in method or "tte" in method:
        score += 20

    # Covariates specified
    if schema.get("statistical_methods", {}).get("covariates"):
        score += 10

    # Missing data / censoring described
    missing = (
        schema.get("statistical_methods", {}).get("missing_data_handling", "") or ""
    ).lower()
    if "censor" in missing:
        score += 10

    # Open questions about TTE lower confidence
    oqs = schema.get("open_questions", [])
    tte_oqs = [q for q in oqs if "censor" in q.get("question", "").lower()
               or "event" in q.get("question", "").lower()
               or "survival" in q.get("question", "").lower()]
    score -= len(tte_oqs) * 5

    return min(max(score, 0), 100)


def _score_adeff_confidence(schema: dict) -> int:
    """
    Score confidence for ADEFF (efficacy change-from-baseline) implementation.

    High confidence requires:
    - CFB endpoint clearly defined
    - Analysis method (MMRM, ANCOVA) specified
    - Covariates described
    - Visit structure implied
    """
    primary_eps = schema.get("endpoints", {}).get("primary", [])
    cfb_eps = [ep for ep in primary_eps if ep.get("type") == "CFB"]

    if not cfb_eps:
        secondary_eps = schema.get("endpoints", {}).get("secondary", [])
        cfb_eps = [ep for ep in secondary_eps if ep.get("type") == "CFB"]

    if not cfb_eps:
        return None  # Not applicable

    score = 35  # baseline

    if cfb_eps[0].get("description"):
        score += 15
    if cfb_eps[0].get("timepoint"):
        score += 10

    method = (
        schema.get("statistical_methods", {}).get("primary_analysis_method", "") or ""
    ).lower()
    if any(m in method for m in ["mmrm", "ancova", "mixed model", "repeated"]):
        score += 20

    if schema.get("statistical_methods", {}).get("covariates"):
        score += 10

    # Visit windowing specified — very important for CFB datasets
    if schema.get("statistical_methods", {}).get("visit_windowing"):
        score += 10

    # Open questions about missing data or visit windows lower confidence
    oqs = schema.get("open_questions", [])
    cfb_oqs = [q for q in oqs if any(
        kw in q.get("question", "").lower()
        for kw in ["visit window", "missing data", "baseline", "imputation"]
    )]
    score -= len(cfb_oqs) * 7

    return min(max(score, 0), 100)


# ---------------------------------------------------------------------------
# Traceability helper
# ---------------------------------------------------------------------------

def _get_section_ref(schema: dict, key: str) -> Optional[str]:
    """
    Look up the SAP section reference for a given key from sap_section_refs.
    Used to attach traceability to each derived recommendation.
    """
    return schema.get("sap_section_refs", {}).get(key)


# ---------------------------------------------------------------------------
# Main rule engine functions
# ---------------------------------------------------------------------------

def derive_sdtm_domains(schema: dict) -> list[dict]:
    """
    Derive expected SDTM domains from the extracted SAP schema.

    Returns a list of dicts:
      {"domain": "DM", "rationale": "...", "sap_section": "...", "always_present": True/False}
    """
    special = schema.get("special_assessments", {})
    domains = []

    # Always-present domains
    for domain in _ALWAYS_SDTM_DOMAINS:
        domains.append({
            "domain": domain,
            "rationale": "Standard domain — always expected in Phase 2/3 clinical trials",
            "sap_section": None,
            "always_present": True,
        })

    # Standard additional domains nearly always present in Phase 2/3 trials
    for domain in _STANDARD_ADDITIONAL_SDTM:
        if domain not in _ALWAYS_SDTM_DOMAINS:
            domains.append({
                "domain": domain,
                "rationale": "Standard domain for Phase 2/3 trials (vital signs, labs, disposition, visits)",
                "sap_section": None,
                "always_present": True,
            })

    # Conditional domains based on special assessments
    already_added = {d["domain"] for d in domains}
    for domain_code, rationale, feature_key in _CONDITIONAL_SDTM_RULES:
        if domain_code in already_added:
            continue
        if special.get(feature_key, False):
            sap_section = _get_section_ref(schema, feature_key)
            domains.append({
                "domain": domain_code,
                "rationale": rationale,
                "sap_section": sap_section,
                "always_present": False,
            })
            already_added.add(domain_code)

    return sorted(domains, key=lambda d: (not d["always_present"], d["domain"]))


def derive_adam_datasets(schema: dict) -> list[dict]:
    """
    Derive expected ADaM datasets from the extracted SAP schema.

    Returns a list of dicts:
      {"dataset": "ADSL", "rationale": "...", "sap_section": "...", "always_present": True/False}
    """
    special = schema.get("special_assessments", {})
    endpoints = schema.get("endpoints", {})
    datasets = []

    # Always-present datasets
    for dataset in _ALWAYS_ADAM_DATASETS:
        datasets.append({
            "dataset": dataset,
            "rationale": "Standard ADaM dataset — always expected in clinical trials",
            "sap_section": None,
            "always_present": True,
        })

    already_added = {d["dataset"] for d in datasets}

    # Determine endpoint types present
    all_endpoints = (
        endpoints.get("primary", []) +
        endpoints.get("secondary", []) +
        endpoints.get("exploratory", [])
    )
    endpoint_types = {ep.get("type") for ep in all_endpoints if ep.get("type")}

    # Feature flags derived from endpoint types
    features = {
        "endpoint_tte": "TTE" in endpoint_types,
        "endpoint_cfb": "CFB" in endpoint_types,
        "patient_reported_outcomes": special.get("patient_reported_outcomes", False),
        "pharmacokinetics": special.get("pharmacokinetics", False),
        "tumor_response": special.get("tumor_response", False),
        "biomarkers": special.get("biomarkers", False),
        "ecg": special.get("ecg", False),
    }

    # Map endpoints section references for traceability
    section_map = {
        "endpoint_tte": _get_section_ref(schema, "endpoints_primary"),
        "endpoint_cfb": _get_section_ref(schema, "endpoints_primary"),
        "patient_reported_outcomes": _get_section_ref(schema, "endpoints_secondary"),
        "pharmacokinetics": None,
        "tumor_response": _get_section_ref(schema, "endpoints_secondary"),
        "biomarkers": None,
        "ecg": None,
    }

    for dataset_name, rationale, feature_key in _CONDITIONAL_ADAM_RULES:
        if dataset_name in already_added:
            continue
        if features.get(feature_key, False):
            datasets.append({
                "dataset": dataset_name,
                "rationale": rationale,
                "sap_section": section_map.get(feature_key),
                "always_present": False,
            })
            already_added.add(dataset_name)

    return sorted(datasets, key=lambda d: (not d["always_present"], d["dataset"]))


def compute_component_confidence(schema: dict, adam_datasets: list[dict]) -> dict:
    """
    Compute per-component confidence scores for each ADaM dataset.

    Scores reflect how much information was available in the SAP to correctly
    implement each dataset. This replaces the placeholder nulls from extraction.

    Returns a dict: {"ADSL": 85, "ADAE": 72, "ADTTE": 61, ...}
    """
    dataset_names = {d["dataset"] for d in adam_datasets}
    confidence = {}

    # Always score ADSL and ADAE since they're always present
    confidence["ADSL"] = _score_adsl_confidence(schema)
    confidence["ADAE"] = _score_adae_confidence(schema)

    # Only score ADTTE / ADEFF if they are in the derived dataset list
    if "ADTTE" in dataset_names:
        confidence["ADTTE"] = _score_adtte_confidence(schema)
    else:
        confidence["ADTTE"] = None

    if "ADEFF" in dataset_names:
        confidence["ADEFF"] = _score_adeff_confidence(schema)
    else:
        confidence["ADEFF"] = None

    # ADQS (PRO data)
    if "ADQS" in dataset_names:
        # Reasonable default; PRO datasets often lack windowing specs
        oqs = schema.get("open_questions", [])
        pro_oqs = [q for q in oqs if "pro" in q.get("question", "").lower()
                   or "questionnaire" in q.get("question", "").lower()]
        confidence["ADQS"] = max(55 - len(pro_oqs) * 5, 20)
    else:
        confidence["ADQS"] = None

    # PK datasets — confidence based on PK section references
    if "ADPC" in dataset_names or "ADPP" in dataset_names:
        pk_score = 60  # PK usually well-specified when present
        if _get_section_ref(schema, "pharmacokinetics"):
            pk_score += 15
        confidence["ADPC"] = pk_score
        confidence["ADPP"] = pk_score
    else:
        confidence["ADPC"] = None
        confidence["ADPP"] = None

    # Tumor response
    if "ADRS" in dataset_names:
        rs_score = 55
        if schema.get("special_assessments", {}).get("tumor_response"):
            rs_score += 20
        if _get_section_ref(schema, "tumor_response"):
            rs_score += 10
        # Open questions about tumor response lower confidence
        oqs = schema.get("open_questions", [])
        rs_oqs = [q for q in oqs if "tumor" in q.get("question", "").lower()
                  or "response" in q.get("question", "").lower()
                  or "recist" in q.get("question", "").lower()]
        rs_score -= len(rs_oqs) * 8
        confidence["ADRS"] = min(max(rs_score, 10), 100)
    else:
        confidence["ADRS"] = None

    return confidence


def run_rules(schema: dict) -> dict:
    """
    Run the full rule engine on an extracted SAP schema.

    Populates:
    - schema["sdtm_domains_expected"]: list of SDTM domain recommendation dicts
    - schema["adam_datasets_expected"]: list of ADaM dataset recommendation dicts
    - schema["component_confidence"]: per-dataset confidence scores

    Returns the updated schema.
    """
    nct_id = schema.get("metadata", {}).get("nct_id", "UNKNOWN")
    logger.info(f"[{nct_id}] Running rule engine")

    sdtm_domains = derive_sdtm_domains(schema)
    adam_datasets = derive_adam_datasets(schema)
    component_confidence = compute_component_confidence(schema, adam_datasets)

    # Store structured recommendations back into the schema
    schema["sdtm_domains_expected"] = sdtm_domains
    schema["adam_datasets_expected"] = adam_datasets
    schema["component_confidence"] = component_confidence

    logger.info(
        f"[{nct_id}] Rule engine derived: "
        f"{len(sdtm_domains)} SDTM domains, "
        f"{len(adam_datasets)} ADaM datasets"
    )

    # Log the confidence scores for visibility
    non_null_scores = {k: v for k, v in component_confidence.items() if v is not None}
    if non_null_scores:
        scores_str = ", ".join(f"{k}: {v}%" for k, v in sorted(non_null_scores.items()))
        logger.info(f"[{nct_id}] Component confidence: {scores_str}")

    return schema
