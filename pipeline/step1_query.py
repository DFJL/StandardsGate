"""
Step 1: Query ClinicalTrials.gov API v2 for studies with SAP documents.

Handles:
- Filtering by therapeutic area (condition), phase, and study status
- Pagination via nextPageToken
- Identifying studies that have a Statistical Analysis Plan large document
- Random sampling with a fixed seed for reproducibility
"""

import time
import random
from typing import Optional
from loguru import logger
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
import logging


def _build_query_params(config: dict, ta: str, page_token: Optional[str] = None) -> dict:
    """Build query parameters for the ClinicalTrials.gov API v2 request."""
    api_cfg = config["clinicaltrials_api"]
    filters = config["filters"]

    # Status filter — filter.overallStatus accepts comma-separated values
    status_filter = ",".join(filters["study_status"]) if filters.get("study_status") else None

    # Keep query minimal — phase is filtered client-side after fetching
    params = {
        "query.cond": ta,
        "query.term": "Statistical Analysis Plan",
        "pageSize": api_cfg["page_size"],
        "format": "json",
    }

    if status_filter:
        params["filter.overallStatus"] = status_filter

    if page_token:
        params["pageToken"] = page_token

    return params


def _has_sap_document(study: dict, doc_type_keyword: str) -> bool:
    """
    Return True if the study has a large document whose typeAbbrev or label
    matches the SAP document type keyword (case-insensitive substring match).
    """
    try:
        large_docs = (
            study.get("protocolSection", {})
            .get("ipdSharingStatementModule", {})
        )
        # LargeDocModule lives under documentSection in API v2
        doc_section = study.get("documentSection", {})
        large_doc_module = doc_section.get("largeDocumentModule", {})
        large_docs_list = large_doc_module.get("largeDocs", [])
    except (KeyError, AttributeError):
        return False

    keyword_lower = doc_type_keyword.lower()
    for doc in large_docs_list:
        label = doc.get("label", "").lower()
        type_abbrev = doc.get("typeAbbrev", "").lower()
        if keyword_lower in label or "sap" in type_abbrev or "statistical analysis" in label:
            return True
    return False


def _extract_study_record(study: dict, ta: str) -> dict:
    """
    Extract relevant fields from a raw API study record into a flat dict.
    Returns None if NCT ID is missing.
    """
    protocol = study.get("protocolSection", {})
    id_module = protocol.get("identificationModule", {})
    status_module = protocol.get("statusModule", {})
    sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
    conditions_module = protocol.get("conditionsModule", {})
    design_module = protocol.get("designModule", {})
    doc_section = study.get("documentSection", {})
    large_doc_module = doc_section.get("largeDocumentModule", {})

    nct_id = id_module.get("nctId", "")
    if not nct_id:
        return None

    # Collect all large doc entries for this study — step 2 will pick the SAP
    large_docs = large_doc_module.get("largeDocs", [])
    sap_docs = []
    for doc in large_docs:
        label = doc.get("label", "").lower()
        type_abbrev = doc.get("typeAbbrev", "").lower()
        if "statistical analysis" in label or "sap" in type_abbrev:
            sap_docs.append(doc)

    return {
        "nct_id": nct_id,
        "study_title": (
            id_module.get("officialTitle")
            or id_module.get("briefTitle", "")
        ),
        "sponsor": sponsor_module.get("leadSponsor", {}).get("name", ""),
        "condition": ", ".join(conditions_module.get("conditions", [])),
        "therapeutic_area": ta,  # the TA we queried for
        "phase": ", ".join(design_module.get("phases", [])),
        "status": status_module.get("overallStatus", ""),
        "large_doc_info": sap_docs,  # list of SAP doc dicts with filename etc.
    }


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    before_sleep=before_sleep_log(logging.getLogger("tenacity"), logging.WARNING),
)
def _fetch_page(session: requests.Session, url: str, params: dict) -> dict:
    """Fetch a single page from the API with retry logic."""
    response = session.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def query_studies_for_ta(config: dict, ta: str) -> list[dict]:
    """
    Query ClinicalTrials.gov for all studies matching the given therapeutic area
    that have SAP documents. Handles full pagination.

    Returns a list of study record dicts.
    """
    api_cfg = config["clinicaltrials_api"]
    filters = config["filters"]
    doc_type_kw = filters["document_type"]
    delay = api_cfg["request_delay_seconds"]

    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    studies_with_sap = []
    page_token = None
    page_num = 0

    logger.info(f"Querying ClinicalTrials.gov for TA='{ta}'")

    while True:
        page_num += 1
        params = _build_query_params(config, ta, page_token)

        logger.debug(f"  Fetching page {page_num} (pageToken={page_token})")
        data = _fetch_page(session, api_cfg["base_url"], params)

        studies = data.get("studies", [])
        logger.debug(f"  Got {len(studies)} studies on page {page_num}")

        allowed_phases = set(filters.get("phases", []))
        for study in studies:
            if _has_sap_document(study, doc_type_kw):
                record = _extract_study_record(study, ta)
                if record:
                    # Filter phase client-side
                    if allowed_phases:
                        study_phases = set(
                            study.get("protocolSection", {})
                            .get("designModule", {})
                            .get("phases", [])
                        )
                        if not study_phases.intersection(allowed_phases):
                            continue
                    studies_with_sap.append(record)

        # Pagination: API v2 returns nextPageToken when more pages exist
        page_token = data.get("nextPageToken")
        if not page_token:
            break

        # Respect rate limit
        time.sleep(delay)

    logger.info(
        f"  Found {len(studies_with_sap)} studies with SAP docs for TA='{ta}'"
    )
    return studies_with_sap


def run(config: dict) -> list[dict]:
    """
    Main entry point for Step 1.

    Queries each configured therapeutic area, then applies sampling to stay
    within max_per_ta and max_total limits (reproducible via seed).

    Returns a deduplicated list of study record dicts ready for Step 2.
    """
    filters = config["filters"]
    sampling = config["sampling"]
    therapeutic_areas = filters["therapeutic_areas"]

    rng = random.Random(sampling["seed"])
    all_selected: list[dict] = []
    seen_nct_ids: set[str] = set()

    for ta in therapeutic_areas:
        ta_studies = query_studies_for_ta(config, ta)

        # Deduplicate within this batch (a study may appear for multiple TAs)
        unique_ta = [s for s in ta_studies if s["nct_id"] not in seen_nct_ids]

        # Sample up to max_per_ta
        max_per = sampling["max_per_ta"]
        if len(unique_ta) > max_per:
            unique_ta = rng.sample(unique_ta, max_per)
            logger.info(f"  Sampled {max_per} studies from TA='{ta}'")

        for s in unique_ta:
            seen_nct_ids.add(s["nct_id"])

        all_selected.extend(unique_ta)

        # Check global cap
        if len(all_selected) >= sampling["max_total"]:
            all_selected = all_selected[: sampling["max_total"]]
            logger.info(f"Reached max_total={sampling['max_total']}, stopping query phase.")
            break

        time.sleep(config["clinicaltrials_api"]["request_delay_seconds"])

    logger.info(
        f"Step 1 complete. Selected {len(all_selected)} studies total across all TAs."
    )
    return all_selected
