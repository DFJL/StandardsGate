"""
Step 2: Download SAP PDFs from ClinicalTrials.gov CDN.

The ClinicalTrials.gov CDN URL pattern is:
  {cdn_base}/{nct_id[-2:]}/{nct_id}/{filename}

For example, NCT12345678 → last 2 digits of the NCT number portion = "78":
  https://cdn.clinicaltrials.gov/large-docs/78/NCT12345678/SAP_001.pdf

Each study's large_doc_info list (from Step 1) contains the exact filename(s).
We pick the first SAP document and download it.

Alongside each PDF we save a metadata.json with provenance info.
"""

import json
import time
import os
from pathlib import Path
from typing import Optional
from loguru import logger
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
import logging


def _cdn_url_for_doc(cdn_base: str, nct_id: str, filename: str) -> str:
    """
    Build the CDN URL for a large document.

    ClinicalTrials.gov CDN organises files by the last two digits of the
    numeric portion of the NCT ID to reduce directory size.

    NCT12345678 → numeric portion = 12345678 → last 2 chars = "78"
    """
    # Strip the "NCT" prefix to get the numeric portion
    numeric = nct_id.upper().replace("NCT", "")
    last_two = numeric[-2:]  # last 2 digits of numeric ID
    return f"{cdn_base.rstrip('/')}/{last_two}/{nct_id}/{filename}"


def _pick_sap_doc(large_doc_info: list[dict]) -> Optional[dict]:
    """
    From the list of large_doc_info dicts, pick the best candidate for the
    SAP PDF. Prefer docs where typeAbbrev is 'SAP' or label contains
    'Statistical Analysis Plan'.
    """
    if not large_doc_info:
        return None

    # Prefer explicit SAP type abbreviation
    for doc in large_doc_info:
        if doc.get("typeAbbrev", "").upper() == "SAP":
            return doc

    # Fall back to label keyword match
    for doc in large_doc_info:
        label = doc.get("label", "").lower()
        if "statistical analysis" in label:
            return doc

    # Last resort: return first item
    return large_doc_info[0]


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    before_sleep=before_sleep_log(logging.getLogger("tenacity"), logging.WARNING),
)
def _download_file(session: requests.Session, url: str, dest_path: Path) -> int:
    """
    Stream-download a file from url into dest_path.
    Returns the number of bytes written.
    """
    response = session.get(url, stream=True, timeout=120)
    response.raise_for_status()

    bytes_written = 0
    with open(dest_path, "wb") as fh:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                fh.write(chunk)
                bytes_written += len(chunk)

    return bytes_written


def download_study_sap(
    config: dict,
    study: dict,
    pdf_dir: Path,
) -> Optional[dict]:
    """
    Download the SAP PDF for a single study.

    Returns a dict with download metadata, or None if download failed / skipped.

    Args:
        config:   Full pipeline config dict.
        study:    Study record dict from Step 1 (must have nct_id, large_doc_info).
        pdf_dir:  Directory to save PDFs into.
    """
    nct_id = study["nct_id"]
    storage_cfg = config["storage"]
    api_cfg = config["clinicaltrials_api"]
    cdn_base = api_cfg["cdn_base_url"]
    overwrite = storage_cfg["overwrite_existing"]

    large_doc_info = study.get("large_doc_info", [])
    sap_doc = _pick_sap_doc(large_doc_info)

    if sap_doc is None:
        logger.warning(f"[{nct_id}] No SAP document found in large_doc_info, skipping download.")
        return None

    filename = sap_doc.get("filename") or sap_doc.get("fileName")
    if not filename:
        logger.warning(f"[{nct_id}] SAP doc entry has no filename field: {sap_doc}")
        return None

    pdf_path = pdf_dir / f"{nct_id}.pdf"
    meta_path = pdf_dir / f"{nct_id}_metadata.json"

    # Idempotency check
    if pdf_path.exists() and not overwrite:
        logger.info(f"[{nct_id}] PDF already exists, skipping (overwrite_existing=false).")
        # Still return metadata if the companion file exists
        if meta_path.exists():
            with open(meta_path) as fh:
                return json.load(fh)
        return {"nct_id": nct_id, "pdf_path": str(pdf_path), "skipped": True}

    url = _cdn_url_for_doc(cdn_base, nct_id, filename)
    logger.info(f"[{nct_id}] Downloading SAP from {url}")

    session = requests.Session()
    session.headers.update({"User-Agent": "StandardsGate/1.0 (clinical trial research pipeline)"})

    try:
        bytes_written = _download_file(session, url, pdf_path)
    except Exception as exc:
        logger.error(f"[{nct_id}] Download failed: {exc}")
        # Clean up partial file
        if pdf_path.exists():
            pdf_path.unlink()
        return None

    logger.info(f"[{nct_id}] Downloaded {bytes_written:,} bytes → {pdf_path.name}")

    # Save provenance metadata alongside the PDF
    metadata = {
        "nct_id": nct_id,
        "study_title": study.get("study_title", ""),
        "sponsor": study.get("sponsor", ""),
        "condition": study.get("condition", ""),
        "therapeutic_area": study.get("therapeutic_area", ""),
        "phase": study.get("phase", ""),
        "status": study.get("status", ""),
        "pdf_path": str(pdf_path),
        "source_url": url,
        "original_filename": filename,
        "sap_doc_label": sap_doc.get("label", ""),
        "sap_doc_type_abbrev": sap_doc.get("typeAbbrev", ""),
        "file_size_bytes": bytes_written,
    }

    with open(meta_path, "w") as fh:
        json.dump(metadata, fh, indent=2)

    return metadata


def run(config: dict, studies: list[dict]) -> list[dict]:
    """
    Main entry point for Step 2.

    Downloads SAP PDFs for all studies in the list.

    Returns a list of metadata dicts for successfully downloaded PDFs.
    """
    storage_cfg = config["storage"]
    base_path = Path(storage_cfg["base_path"])
    pdf_dir = base_path / storage_cfg["subdirs"]["raw_pdfs"]
    pdf_dir.mkdir(parents=True, exist_ok=True)

    delay = config["clinicaltrials_api"]["request_delay_seconds"]
    results = []

    logger.info(f"Step 2: Downloading SAP PDFs for {len(studies)} studies → {pdf_dir}")

    for i, study in enumerate(studies, 1):
        nct_id = study.get("nct_id", "UNKNOWN")
        logger.info(f"  [{i}/{len(studies)}] {nct_id}")

        meta = download_study_sap(config, study, pdf_dir)
        if meta:
            results.append(meta)

        # Polite delay between downloads
        if i < len(studies):
            time.sleep(delay)

    logger.info(
        f"Step 2 complete. Successfully obtained PDFs for {len(results)}/{len(studies)} studies."
    )
    return results
