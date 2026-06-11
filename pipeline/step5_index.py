"""
Step 5: Update the master_index.csv with new study entries.

The master index is a flat CSV that provides a quick summary of all studies
in the knowledge base. It enables fast filtering, searching, and reporting
without having to open individual JSON schema files.

Design principles:
- Idempotent: if a study already exists in the index and overwrite_existing=false,
  it is skipped silently
- Additive: new studies are appended; existing studies are never deleted
- Atomic writes: we write to a temp file and rename to avoid partial CSV corruption
- Schema-stable: all columns are always present, even if some values are null/empty
"""

import csv
import json
import os
from datetime import date
from pathlib import Path
from typing import Optional
from loguru import logger


# Canonical column order for master_index.csv — matches the spec exactly.
MASTER_INDEX_COLUMNS = [
    "nct_id",
    "study_title",
    "sponsor",
    "therapeutic_area",
    "indication",
    "phase",
    "study_design",
    "primary_endpoint_type",
    "primary_endpoint_desc",
    "has_pk",
    "has_tumor_response",
    "has_pro",
    "has_imaging",
    "has_ecg",
    "analysis_populations",
    "multiplicity_adjustment",
    "estimand_framework",
    "bayesian",
    "sdtm_domains_expected",
    "adam_datasets_expected",
    "open_questions_count",
    "high_severity_flags",
    "extraction_confidence",
    "source_pdf",
    "extraction_date",
]


def _schema_to_index_row(schema: dict, source_pdf: str) -> dict:
    """
    Flatten a canonical SAP schema dict into a flat CSV row dict.

    All values are converted to strings suitable for CSV storage.
    List fields are serialized as pipe-delimited strings for easy parsing.
    """
    meta = schema.get("metadata", {})
    endpoints = schema.get("endpoints", {})
    special = schema.get("special_assessments", {})
    methods = schema.get("statistical_methods", {})
    pops = schema.get("analysis_populations", [])
    open_qs = schema.get("open_questions", [])
    flags = schema.get("flags", [])
    sdtm = schema.get("sdtm_domains_expected", [])
    adam = schema.get("adam_datasets_expected", [])

    # Primary endpoint info
    primary_eps = endpoints.get("primary", [])
    primary_type = (primary_eps[0].get("type") or "") if primary_eps else ""
    primary_desc = (primary_eps[0].get("description") or "") if primary_eps else ""

    # Analysis populations as pipe-delimited abbreviations
    pop_abbrevs = "|".join(
        (p.get("abbreviation") or p.get("name") or "") for p in pops
    )

    # SDTM domains: extract domain codes from the structured dicts
    sdtm_codes = "|".join(
        (d.get("domain") or str(d)) if isinstance(d, dict) else str(d)
        for d in sdtm
    )

    # ADaM datasets: extract dataset names
    adam_names = "|".join(
        (d.get("dataset") or str(d)) if isinstance(d, dict) else str(d)
        for d in adam
    )

    # Count high-severity open questions
    high_oq_count = sum(
        1 for q in open_qs if q.get("severity", "").upper() == "HIGH"
    )

    # Count total flags (separate from open questions — flags are qualitative)
    total_flags = len(flags)

    return {
        "nct_id": meta.get("nct_id", ""),
        "study_title": meta.get("study_title", ""),
        "sponsor": meta.get("sponsor", ""),
        "therapeutic_area": meta.get("therapeutic_area", ""),
        "indication": meta.get("indication", ""),
        "phase": meta.get("phase", ""),
        "study_design": meta.get("study_design", ""),
        "primary_endpoint_type": primary_type,
        "primary_endpoint_desc": primary_desc[:200],  # truncate long descriptions
        "has_pk": str(special.get("pharmacokinetics", False)).lower(),
        "has_tumor_response": str(special.get("tumor_response", False)).lower(),
        "has_pro": str(special.get("patient_reported_outcomes", False)).lower(),
        "has_imaging": str(special.get("imaging", False)).lower(),
        "has_ecg": str(special.get("ecg", False)).lower(),
        "analysis_populations": pop_abbrevs,
        "multiplicity_adjustment": str(methods.get("multiplicity_adjustment", False)).lower(),
        "estimand_framework": str(methods.get("estimand_framework", False)).lower(),
        "bayesian": str(methods.get("bayesian_elements", False)).lower(),
        "sdtm_domains_expected": sdtm_codes,
        "adam_datasets_expected": adam_names,
        "open_questions_count": str(len(open_qs)),
        "high_severity_flags": str(high_oq_count + total_flags),
        "extraction_confidence": schema.get("extraction_confidence", ""),
        "source_pdf": source_pdf,
        "extraction_date": meta.get("extraction_date", date.today().isoformat()),
    }


def _load_existing_index(index_path: Path) -> dict[str, dict]:
    """
    Load an existing master_index.csv into a dict keyed by nct_id.
    Returns an empty dict if the file does not exist or is empty.
    """
    if not index_path.exists():
        return {}

    existing = {}
    try:
        with open(index_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                nct_id = row.get("nct_id", "").strip()
                if nct_id:
                    existing[nct_id] = row
    except Exception as exc:
        logger.warning(f"Could not read existing master index: {exc}. Starting fresh.")

    return existing


def _write_index(index_path: Path, rows: dict[str, dict]) -> None:
    """
    Write all rows to master_index.csv atomically (temp file + rename).

    rows: dict keyed by nct_id, values are row dicts with MASTER_INDEX_COLUMNS.
    """
    # Sort by nct_id for deterministic output
    sorted_rows = sorted(rows.values(), key=lambda r: r.get("nct_id", ""))

    # Write to temp file first
    tmp_path = index_path.with_suffix(".csv.tmp")
    try:
        with open(tmp_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=MASTER_INDEX_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(sorted_rows)

        # Atomic rename
        tmp_path.replace(index_path)

    except Exception as exc:
        # Clean up temp file on error
        if tmp_path.exists():
            tmp_path.unlink()
        raise exc


def update_index_for_study(
    config: dict,
    schema: dict,
    source_pdf: str,
    index_path: Path,
) -> bool:
    """
    Add or update a single study's entry in master_index.csv.

    Returns True if the study was added/updated, False if skipped.
    """
    overwrite = config["storage"]["overwrite_existing"]
    nct_id = schema.get("metadata", {}).get("nct_id", "")

    if not nct_id:
        logger.warning("Schema has no nct_id in metadata — cannot index.")
        return False

    existing = _load_existing_index(index_path)

    if nct_id in existing and not overwrite:
        logger.info(f"[{nct_id}] Already in master index, skipping (overwrite_existing=false).")
        return False

    row = _schema_to_index_row(schema, source_pdf)
    existing[nct_id] = row
    _write_index(index_path, existing)

    action = "Updated" if nct_id in existing else "Added"
    logger.info(f"[{nct_id}] {action} in master index.")
    return True


def run(config: dict, parse_results: list[dict]) -> Path:
    """
    Main entry point for Step 5.

    Updates master_index.csv with all successfully parsed studies.

    Returns the Path to the master_index.csv file.

    Args:
        config:        Full pipeline config dict.
        parse_results: List of enriched result dicts from Step 4 (must contain
                       'schema' and 'schema_path' keys).
    """
    storage_cfg = config["storage"]
    base_path = Path(storage_cfg["base_path"])
    base_path.mkdir(parents=True, exist_ok=True)

    index_path = base_path / storage_cfg["master_index_file"]

    logger.info(
        f"Step 5: Updating master index ({index_path}) "
        f"with {len(parse_results)} studies."
    )

    added = 0
    skipped = 0

    for result in parse_results:
        nct_id = result.get("nct_id", "UNKNOWN")
        schema = result.get("schema")

        if schema is None:
            # Try loading from disk if schema wasn't carried through
            schema_path_str = result.get("schema_path")
            if schema_path_str and Path(schema_path_str).exists():
                try:
                    with open(schema_path_str) as fh:
                        schema = json.load(fh)
                except Exception as exc:
                    logger.warning(f"[{nct_id}] Could not load schema from {schema_path_str}: {exc}")
                    skipped += 1
                    continue
            else:
                logger.warning(f"[{nct_id}] No schema available, skipping index update.")
                skipped += 1
                continue

        source_pdf = result.get("pdf_path", result.get("source_pdf", ""))

        was_added = update_index_for_study(config, schema, source_pdf, index_path)
        if was_added:
            added += 1
        else:
            skipped += 1

    logger.info(
        f"Step 5 complete. Added/updated {added} entries; skipped {skipped}. "
        f"Master index: {index_path}"
    )

    # Report current total count in the index
    existing = _load_existing_index(index_path)
    logger.info(f"Master index now contains {len(existing)} studies total.")

    return index_path
