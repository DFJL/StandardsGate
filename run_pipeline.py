"""
StandardsGate Component 1 — SAP Knowledge Base Pipeline
Main orchestrator script.

Usage:
    python run_pipeline.py
    python run_pipeline.py --dry-run
    python run_pipeline.py --nct-ids NCT12345678 NCT98765432
    python run_pipeline.py --ta "Breast Cancer"
    python run_pipeline.py --steps 1 2 3  # run only specified steps
    python run_pipeline.py --config path/to/custom_config.yaml

Pipeline steps:
    1. Query ClinicalTrials.gov API for studies with SAP documents
    2. Download SAP PDFs
    3. Extract text from PDFs
    4. Parse SAPs into canonical JSON schema (Claude API)
    5. Update master_index.csv

All parameters come from config.yaml — nothing is hardcoded here.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml
from loguru import logger
try:
    from dotenv import load_dotenv  # optional — graceful fallback if not installed
except ImportError:
    def load_dotenv(*args, **kwargs): pass  # no-op if dotenv not installed

# Import pipeline steps
from pipeline import step1_query, step2_download, step3_extract, step4_parse, step5_index
from pipeline.rule_engine import run_rules


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(config: dict, log_dir: Path) -> None:
    """
    Configure loguru for both console (INFO) and file (DEBUG) output.
    The log file is named with a timestamp for easy chronological sorting.
    """
    log_cfg = config.get("logging", {})
    level = log_cfg.get("level", "INFO").upper()

    # Remove loguru's default handler so we fully control output
    logger.remove()

    # Console: INFO and above, colorized
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File: DEBUG and above for full diagnostic output
    if log_cfg.get("log_to_file", True):
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"pipeline_{timestamp}.log"
        logger.add(
            log_file,
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} — {message}",
            rotation="50 MB",
            retention="30 days",
        )
        logger.info(f"Log file: {log_file}")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.yaml") -> dict:
    """Load and return the pipeline configuration from YAML file."""
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        # Try relative to this script's directory
        cfg_path = Path(__file__).parent / config_path

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(cfg_path) as fh:
        config = yaml.safe_load(fh)

    logger.debug(f"Loaded config from {cfg_path}")
    return config


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """
    Apply command-line argument overrides to the config dict.

    This allows quick parameter changes without editing config.yaml.
    """
    # Therapeutic area override
    if args.ta:
        config["filters"]["therapeutic_areas"] = [args.ta]
        logger.info(f"CLI override: therapeutic_areas → [{args.ta}]")

    # Overwrite override
    if args.overwrite:
        config["storage"]["overwrite_existing"] = True
        logger.info("CLI override: overwrite_existing → true")

    return config


# ---------------------------------------------------------------------------
# Synthetic study builder for --nct-ids mode
# ---------------------------------------------------------------------------

def build_stub_studies_from_nct_ids(nct_ids: list[str], config: dict) -> list[dict]:
    """
    When the user provides specific NCT IDs, we skip the query step and
    instead query the ClinicalTrials.gov API individually for each study.

    Returns a list of study record dicts (same format as step1_query output).
    """
    import time
    import requests
    from tenacity import retry, stop_after_attempt, wait_exponential

    api_cfg = config["clinicaltrials_api"]
    base_url = api_cfg["base_url"]
    delay = api_cfg["request_delay_seconds"]

    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    studies = []

    for i, nct_id in enumerate(nct_ids):
        logger.info(f"Fetching study info for {nct_id} from ClinicalTrials.gov")
        try:
            resp = session.get(
                f"{base_url}/{nct_id}",
                params={"format": "json"},
                timeout=30,
            )
            resp.raise_for_status()
            study_data = resp.json()

            # The single-study endpoint returns the study object directly
            protocol = study_data.get("protocolSection", {})
            id_module = protocol.get("identificationModule", {})
            status_module = protocol.get("statusModule", {})
            sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
            conditions_module = protocol.get("conditionsModule", {})
            design_module = protocol.get("designModule", {})
            doc_section = study_data.get("documentSection", {})
            large_doc_module = doc_section.get("largeDocumentModule", {})
            large_docs = large_doc_module.get("largeDocs", [])

            # All large docs included — step 2 picks the SAP
            studies.append({
                "nct_id": nct_id,
                "study_title": (
                    id_module.get("officialTitle") or id_module.get("briefTitle", "")
                ),
                "sponsor": sponsor_module.get("leadSponsor", {}).get("name", ""),
                "condition": ", ".join(conditions_module.get("conditions", [])),
                "therapeutic_area": conditions_module.get("conditions", [""])[0],
                "phase": ", ".join(design_module.get("phases", [])),
                "status": status_module.get("overallStatus", ""),
                "large_doc_info": large_docs,
            })

        except Exception as exc:
            logger.error(f"Failed to fetch study info for {nct_id}: {exc}")

        if i < len(nct_ids) - 1:
            time.sleep(delay)

    return studies


# ---------------------------------------------------------------------------
# Step execution with dry-run support
# ---------------------------------------------------------------------------

def should_run_step(step_num: int, steps_filter: list[int]) -> bool:
    """Return True if this step should be executed."""
    if not steps_filter:
        return True  # run all steps when no filter is given
    return step_num in steps_filter


# ---------------------------------------------------------------------------
# Main pipeline orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    """
    Execute the full SAP knowledge base pipeline.

    Steps:
    1. Query CT.gov for studies with SAP documents
    2. Download SAP PDFs
    3. Extract text from PDFs
    4. Parse with Claude into canonical JSON schema
    5. Update master_index.csv
    """
    # Load environment variables from .env if present (graceful if dotenv missing)
    try:
        load_dotenv()
    except Exception:
        pass  # dotenv is optional; key can be set directly in environment

    # Load config and apply CLI overrides
    config = load_config(args.config)
    config = apply_cli_overrides(config, args)

    # Set up storage directories and logging
    base_path = Path(config["storage"]["base_path"])
    log_dir = base_path / config["storage"]["subdirs"]["logs"]
    setup_logging(config, log_dir)

    logger.info("=" * 70)
    logger.info("StandardsGate Component 1 — SAP Knowledge Base Pipeline")
    logger.info("=" * 70)

    if args.dry_run:
        logger.info("DRY RUN MODE: No files will be downloaded or APIs called.")

    steps_filter = args.steps or []

    # -----------------------------------------------------------------------
    # STEP 1: Query ClinicalTrials.gov
    # -----------------------------------------------------------------------
    studies = []

    if args.nct_ids:
        logger.info(f"Using provided NCT IDs: {args.nct_ids}")
        if not args.dry_run:
            studies = build_stub_studies_from_nct_ids(args.nct_ids, config)
        else:
            # Dry run: create stub records so subsequent steps can log what they'd do
            studies = [{"nct_id": nct_id, "large_doc_info": [], "study_title": "", "sponsor": "",
                        "condition": "", "therapeutic_area": "", "phase": "", "status": ""}
                       for nct_id in args.nct_ids]
    elif should_run_step(1, steps_filter):
        logger.info("--- STEP 1: Querying ClinicalTrials.gov ---")
        if not args.dry_run:
            studies = step1_query.run(config)
        else:
            logger.info("[DRY RUN] Skipping actual API calls.")
            studies = []
    else:
        logger.info("Step 1 skipped (not in --steps filter).")

    logger.info(f"Step 1 result: {len(studies)} studies to process.")

    if not studies:
        if not args.dry_run:
            logger.warning("No studies found. Pipeline will exit early.")
            return
        else:
            logger.info("[DRY RUN] Continuing with empty study list for demonstration.")

    # -----------------------------------------------------------------------
    # STEP 2: Download SAP PDFs
    # -----------------------------------------------------------------------
    download_results = []

    if should_run_step(2, steps_filter):
        logger.info("--- STEP 2: Downloading SAP PDFs ---")
        if not args.dry_run:
            download_results = step2_download.run(config, studies)
        else:
            logger.info(f"[DRY RUN] Would download PDFs for {len(studies)} studies.")
            download_results = []
    else:
        logger.info("Step 2 skipped. Attempting to discover existing PDFs...")
        # Try to reconstruct download results from existing PDFs
        pdf_dir = base_path / config["storage"]["subdirs"]["raw_pdfs"]
        if pdf_dir.exists():
            for study in studies:
                nct_id = study["nct_id"]
                meta_path = pdf_dir / f"{nct_id}_metadata.json"
                pdf_path = pdf_dir / f"{nct_id}.pdf"
                if meta_path.exists():
                    with open(meta_path) as fh:
                        download_results.append(json.load(fh))
                elif pdf_path.exists():
                    download_results.append({**study, "pdf_path": str(pdf_path)})

    logger.info(f"Step 2 result: {len(download_results)} PDFs available.")

    # -----------------------------------------------------------------------
    # STEP 3: Extract text from PDFs
    # -----------------------------------------------------------------------
    extraction_results = []

    if should_run_step(3, steps_filter):
        logger.info("--- STEP 3: Extracting text from PDFs ---")
        if not args.dry_run and download_results:
            extraction_results = step3_extract.run(config, download_results)
        else:
            if args.dry_run:
                logger.info(f"[DRY RUN] Would extract text from {len(download_results)} PDFs.")
            extraction_results = []
    else:
        logger.info("Step 3 skipped. Attempting to discover existing text files...")
        text_dir = base_path / config["storage"]["subdirs"]["extracted_text"]
        if text_dir.exists():
            for meta in download_results:
                nct_id = meta.get("nct_id", "")
                txt_path = text_dir / f"{nct_id}.txt"
                if txt_path.exists():
                    word_count = len(txt_path.read_text(encoding="utf-8").split())
                    extraction_results.append({
                        **meta,
                        "text_path": str(txt_path),
                        "word_count": word_count,
                    })

    logger.info(f"Step 3 result: {len(extraction_results)} text files available.")

    # -----------------------------------------------------------------------
    # STEP 4: Parse with Claude → canonical JSON schema
    # -----------------------------------------------------------------------
    parse_results = []

    if should_run_step(4, steps_filter):
        logger.info("--- STEP 4: Parsing SAPs with Claude ---")

        # Validate API key before starting (fail fast rather than after step 3)
        if not args.dry_run:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                logger.error(
                    "ANTHROPIC_API_KEY not set. Set it in your .env file or environment. "
                    "Skipping Step 4."
                )
            elif extraction_results:
                parse_results = step4_parse.run(config, extraction_results)
            else:
                logger.warning("No extraction results to parse.")
        else:
            logger.info(f"[DRY RUN] Would parse {len(extraction_results)} SAPs with Claude.")
    else:
        logger.info("Step 4 skipped. Attempting to discover existing schema files...")
        schema_dir = base_path / config["storage"]["subdirs"]["parsed_schemas"]
        if schema_dir.exists():
            for meta in extraction_results:
                nct_id = meta.get("nct_id", "")
                schema_path = schema_dir / f"{nct_id}.json"
                if schema_path.exists():
                    try:
                        with open(schema_path) as fh:
                            schema = json.load(fh)
                        parse_results.append({
                            **meta,
                            "schema_path": str(schema_path),
                            "extraction_confidence": schema.get("extraction_confidence", "UNKNOWN"),
                            "open_questions_count": len(schema.get("open_questions", [])),
                            "schema": schema,
                        })
                    except Exception as exc:
                        logger.warning(f"[{nct_id}] Could not load schema: {exc}")

    # Run rule engine on any parsed schemas (even if step 4 was skipped)
    if parse_results:
        logger.info("--- RULE ENGINE: Deriving SDTM domains and ADaM datasets ---")
        schema_dir = base_path / config["storage"]["subdirs"]["parsed_schemas"]
        for result in parse_results:
            schema = result.get("schema")
            if schema:
                nct_id = result.get("nct_id", "UNKNOWN")
                # Run the deterministic rule engine
                updated_schema = run_rules(schema, config)
                result["schema"] = updated_schema

                # Persist the updated schema back to disk
                schema_path = schema_dir / f"{nct_id}.json"
                try:
                    with open(schema_path, "w", encoding="utf-8") as fh:
                        json.dump(updated_schema, fh, indent=2, ensure_ascii=False)
                    logger.debug(f"[{nct_id}] Schema updated with rule engine results.")
                except Exception as exc:
                    logger.warning(f"[{nct_id}] Could not save updated schema: {exc}")

    logger.info(f"Step 4 result: {len(parse_results)} schemas processed by rule engine.")

    # -----------------------------------------------------------------------
    # STEP 5: Update master_index.csv
    # -----------------------------------------------------------------------
    if should_run_step(5, steps_filter):
        logger.info("--- STEP 5: Updating master index ---")
        if not args.dry_run and parse_results:
            index_path = step5_index.run(config, parse_results)
            logger.info(f"Master index updated: {index_path}")
        elif args.dry_run:
            logger.info(f"[DRY RUN] Would update master index with {len(parse_results)} entries.")
        else:
            logger.warning("No parse results to index.")
    else:
        logger.info("Step 5 skipped.")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("Pipeline run complete.")
    logger.info(f"  Studies queried:     {len(studies)}")
    logger.info(f"  PDFs downloaded:     {len(download_results)}")
    logger.info(f"  Texts extracted:     {len(extraction_results)}")
    logger.info(f"  Schemas parsed:      {len(parse_results)}")

    if parse_results:
        high_conf = sum(1 for r in parse_results
                        if r.get("extraction_confidence") == "HIGH")
        med_conf = sum(1 for r in parse_results
                       if r.get("extraction_confidence") == "MEDIUM")
        low_conf = sum(1 for r in parse_results
                       if r.get("extraction_confidence") == "LOW")
        total_oqs = sum(r.get("open_questions_count", 0) for r in parse_results)
        logger.info(
            f"  Extraction quality:  HIGH={high_conf}, MED={med_conf}, LOW={low_conf}"
        )
        logger.info(f"  Total open questions: {total_oqs}")

    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "StandardsGate SAP Knowledge Base Pipeline\n"
            "Queries ClinicalTrials.gov, downloads SAP PDFs, extracts and parses them\n"
            "into a canonical JSON schema, and maintains a master_index.csv.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML file (default: config.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log what would be done without making API calls or downloading files.",
    )
    parser.add_argument(
        "--nct-ids",
        nargs="+",
        metavar="NCT_ID",
        help=(
            "Process specific NCT IDs directly (skips query step). "
            "Example: --nct-ids NCT12345678 NCT98765432"
        ),
    )
    parser.add_argument(
        "--ta",
        metavar="THERAPEUTIC_AREA",
        help=(
            "Override therapeutic area filter with a single value. "
            'Example: --ta "Breast Cancer"'
        ),
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        metavar="N",
        choices=[1, 2, 3, 4, 5],
        help=(
            "Run only specific pipeline steps (1-5). "
            "Example: --steps 3 4 5 to re-extract and re-parse existing PDFs."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files even if overwrite_existing=false in config.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
