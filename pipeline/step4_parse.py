"""
Step 4: Parse SAP text into canonical JSON schema using the Claude API.

Sends the extracted text (with a structured extraction prompt) to Claude and
receives back a JSON object conforming to the canonical SAP schema.

Key design decisions:
- Uses tenacity for robust retry logic (rate limits, transient errors)
- Loads the extraction prompt from file — never hardcodes prompt text here
- Truncates very long SAP texts to stay within Claude's context window while
  preserving the most important sections (front-loaded in SAPs)
- Validates that the response is parseable JSON before returning
- Rule engine (step 5) populates sdtm_domains_expected / adam_datasets_expected
  AFTER this step — so this module intentionally leaves them empty
"""

import json
import os
from datetime import date
from pathlib import Path
from typing import Optional
from loguru import logger
import anthropic
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import logging

# Maximum characters to send to Claude. SAPs can be very long; we take the
# first N characters which typically cover study design, populations, and
# primary endpoints — the most critical sections for CDISC planning.
# ~120,000 chars ≈ ~30,000 tokens, well within claude-sonnet context window.
MAX_TEXT_CHARS = 400_000  # ~100k tokens — well within Claude's 200k context window


def _load_prompt_template(prompt_file: str) -> str:
    """Load the extraction prompt template from disk."""
    prompt_path = Path(prompt_file)
    if not prompt_path.is_absolute():
        # Resolve relative to the project root (parent of this file's parent dir)
        project_root = Path(__file__).parent.parent
        prompt_path = project_root / prompt_file

    if not prompt_path.exists():
        raise FileNotFoundError(f"Extraction prompt file not found: {prompt_path}")

    return prompt_path.read_text(encoding="utf-8")


def _truncate_text(text: str, max_chars: int = MAX_TEXT_CHARS) -> str:
    """
    Truncate text to max_chars if necessary.

    Hard safety cap for extremely long documents (>400k chars / ~100k tokens).
    Combined protocol+SAP documents can be long and have statistical sections
    near the end — the limit is intentionally high to avoid cutting those sections.
    We add a notice so Claude knows the document was truncated.
    """
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    notice = (
        "\n\n[NOTE: SAP text was truncated at this point due to length. "
        "Extract what is available from the text above.]\n"
    )
    return truncated + notice


def _build_prompt(template: str, sap_text: str) -> str:
    """Substitute the SAP text into the prompt template."""
    # The template uses {SAP_TEXT} as placeholder
    return template.replace("{SAP_TEXT}", sap_text)


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=5, max=120),
    # Retry on rate limit errors and transient API errors
    retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APIStatusError)),
    before_sleep=before_sleep_log(logging.getLogger("tenacity"), logging.WARNING),
)
def _call_claude(
    client: anthropic.Anthropic,
    model: str,
    max_tokens: int,
    prompt: str,
) -> str:
    """
    Send the extraction prompt to Claude and return the raw text response.

    Retries automatically on rate limit (429) and transient 5xx errors.
    """
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )
    return message.content[0].text


def _parse_json_response(raw_response: str, nct_id: str) -> Optional[dict]:
    """
    Parse the JSON from Claude's response.

    Claude is instructed to return only JSON, but occasionally wraps it in
    markdown code fences. This function strips those if present.
    """
    text = raw_response.strip()

    # Strip markdown code fences if present (e.g. ```json ... ```)
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        inner_lines = []
        in_fence = False
        for line in lines:
            if line.strip().startswith("```") and not in_fence:
                in_fence = True
                continue
            elif line.strip() == "```" and in_fence:
                break
            elif in_fence:
                inner_lines.append(line)
        text = "\n".join(inner_lines)

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error(f"[{nct_id}] Failed to parse Claude response as JSON: {exc}")
        logger.debug(f"[{nct_id}] Raw response (first 500 chars): {raw_response[:500]}")
        return None


def _post_process_schema(schema: dict, meta: dict, source_doc: str) -> dict:
    """
    Post-process the extracted schema:
    - Fill in metadata fields that Claude may not know (nct_id, extraction_date)
    - Ensure required fields exist with defaults
    - Guarantee sdtm_domains_expected / adam_datasets_expected are empty (rule engine fills these)
    """
    # Ensure metadata section exists
    if "metadata" not in schema:
        schema["metadata"] = {}

    # Override/fill metadata from our known provenance
    schema["metadata"]["nct_id"] = meta.get("nct_id", schema["metadata"].get("nct_id", ""))
    schema["metadata"]["extraction_date"] = date.today().isoformat()
    schema["metadata"]["source_document"] = source_doc

    # Fill from download metadata if Claude left these blank
    if not schema["metadata"].get("study_title"):
        schema["metadata"]["study_title"] = meta.get("study_title", "")
    if not schema["metadata"].get("sponsor"):
        schema["metadata"]["sponsor"] = meta.get("sponsor", "")
    if not schema["metadata"].get("therapeutic_area"):
        schema["metadata"]["therapeutic_area"] = meta.get("therapeutic_area", "")
    if not schema["metadata"].get("indication"):
        schema["metadata"]["indication"] = meta.get("condition", "")
    if not schema["metadata"].get("phase"):
        schema["metadata"]["phase"] = meta.get("phase", "")

    # Reset to empty lists as a safeguard — the LLM recommendation engine
    # (rule_engine.py, Step 4b) populates these via a second Claude API call
    # after this step completes. Any values Claude may have extracted here
    # are intentionally discarded in favour of the structured LLM output.
    schema["sdtm_domains_expected"] = []
    schema["adam_datasets_expected"] = []

    # Ensure structural fields exist with sensible defaults
    schema.setdefault("endpoints", {"primary": [], "secondary": [], "exploratory": []})
    schema.setdefault("analysis_populations", [])
    schema.setdefault("open_questions", [])
    schema.setdefault("flags", [])
    schema.setdefault("non_standard_approaches", [])
    schema.setdefault("sap_section_refs", {})
    schema.setdefault("component_confidence", {
        "ADSL": None, "ADAE": None, "ADTTE": None, "ADEFF": None,
        "ADPC": None, "ADRS": None,
    })
    schema.setdefault("extraction_confidence", "LOW")
    schema.setdefault("extraction_notes", "")
    schema.setdefault("special_assessments", {
        "pharmacokinetics": False,
        "tumor_response": False,
        "patient_reported_outcomes": False,
        "biomarkers": False,
        "imaging": False,
        "ecg": False,
        "ophthalmology": False,
        "non_standard_endpoints": [],
    })
    schema.setdefault("statistical_methods", {
        "primary_analysis_method": None,
        "covariates": None,
        "visit_windowing": None,
        "multiplicity_adjustment": False,
        "missing_data_handling": "Not specified",
        "estimand_framework": False,
        "bayesian_elements": False,
        "sap_section": None,
    })

    return schema


def parse_study_sap(
    config: dict,
    nct_id: str,
    text_path: Path,
    schema_dir: Path,
    meta: dict,
) -> Optional[dict]:
    """
    Parse a single study's SAP text file into the canonical JSON schema.

    Returns the parsed schema dict, or None on failure.

    Args:
        config:      Full pipeline config dict.
        nct_id:      NCT identifier.
        text_path:   Path to the extracted .txt file.
        schema_dir:  Directory to save the JSON schema output.
        meta:        Download/extraction metadata dict (used for provenance).
    """
    storage_cfg = config["storage"]
    claude_cfg = config["claude_api"]
    overwrite = storage_cfg["overwrite_existing"]

    schema_path = schema_dir / f"{nct_id}.json"

    # Idempotency: skip if already parsed
    if schema_path.exists() and not overwrite:
        logger.info(f"[{nct_id}] Schema JSON already exists, skipping parse.")
        try:
            with open(schema_path) as fh:
                return json.load(fh)
        except Exception as exc:
            logger.warning(f"[{nct_id}] Could not read existing schema: {exc}. Re-parsing.")

    if not text_path.exists():
        logger.error(f"[{nct_id}] Text file not found: {text_path}")
        return None

    # Load and prepare the SAP text
    sap_text = text_path.read_text(encoding="utf-8")
    sap_text_truncated = _truncate_text(sap_text)

    if len(sap_text) > MAX_TEXT_CHARS:
        logger.info(
            f"[{nct_id}] SAP text truncated from {len(sap_text):,} to "
            f"{MAX_TEXT_CHARS:,} characters for Claude."
        )

    # Load prompt template
    prompt_template = _load_prompt_template(claude_cfg["extraction_prompt_file"])
    prompt = _build_prompt(prompt_template, sap_text_truncated)

    # Initialize Anthropic client (reads ANTHROPIC_API_KEY from env)
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable not set. "
            "Copy .env.example to .env and set your key."
        )

    client = anthropic.Anthropic(api_key=api_key)

    logger.info(
        f"[{nct_id}] Sending {len(sap_text_truncated):,} chars to Claude "
        f"({claude_cfg['model']})"
    )

    try:
        raw_response = _call_claude(
            client=client,
            model=claude_cfg["model"],
            max_tokens=claude_cfg["max_tokens"],
            prompt=prompt,
        )
    except Exception as exc:
        logger.error(f"[{nct_id}] Claude API call failed after retries: {exc}")
        return None

    # Parse the JSON response
    schema = _parse_json_response(raw_response, nct_id)
    if schema is None:
        return None

    # Post-process: fill provenance fields, ensure structural integrity
    source_doc = meta.get("original_filename", f"{nct_id}.pdf")
    schema = _post_process_schema(schema, meta, source_doc)

    # Save the schema to disk
    with open(schema_path, "w", encoding="utf-8") as fh:
        json.dump(schema, fh, indent=2, ensure_ascii=False)

    confidence = schema.get("extraction_confidence", "UNKNOWN")
    oq_count = len(schema.get("open_questions", []))
    logger.info(
        f"[{nct_id}] Parsed schema saved (confidence={confidence}, "
        f"open_questions={oq_count}) → {schema_path.name}"
    )

    return schema


def run(config: dict, extraction_results: list[dict]) -> list[dict]:
    """
    Main entry point for Step 4.

    Parses extracted SAP text files into canonical JSON schemas using Claude.

    Returns a list of enriched result dicts that add 'schema_path' and
    'extraction_confidence' fields.
    """
    storage_cfg = config["storage"]
    base_path = Path(storage_cfg["base_path"])
    schema_dir = base_path / storage_cfg["subdirs"]["parsed_schemas"]
    schema_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Step 4: Parsing {len(extraction_results)} SAPs with Claude → {schema_dir}"
    )

    enriched = []

    for i, meta in enumerate(extraction_results, 1):
        nct_id = meta.get("nct_id", "UNKNOWN")
        text_path_str = meta.get("text_path")

        if not text_path_str:
            logger.warning(f"[{nct_id}] No text_path in metadata, skipping.")
            continue

        text_path = Path(text_path_str)
        schema_path = schema_dir / f"{nct_id}.json"

        logger.info(f"  [{i}/{len(extraction_results)}] {nct_id}")

        schema = parse_study_sap(
            config=config,
            nct_id=nct_id,
            text_path=text_path,
            schema_dir=schema_dir,
            meta=meta,
        )

        if schema is None:
            logger.warning(f"[{nct_id}] Skipping due to parse failure.")
            continue

        enriched.append({
            **meta,
            "schema_path": str(schema_path),
            "extraction_confidence": schema.get("extraction_confidence", "UNKNOWN"),
            "open_questions_count": len(schema.get("open_questions", [])),
            "schema": schema,  # carry the schema through for rule engine
        })

    logger.info(
        f"Step 4 complete. Parsed {len(enriched)}/{len(extraction_results)} studies."
    )
    return enriched
