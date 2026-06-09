"""
CDISC Recommendation Engine: LLM-based SDTM/ADaM recommendations.

Replaces the previous deterministic IF-THEN rule engine with a second
Claude API call that reasons over the extracted SAP schema to produce
context-aware recommendations with clinical rationale.

Two-step LLM pipeline:
  Step 4  (step4_parse.py): SAP text → canonical schema
  Step 4b (this module):    canonical schema → SDTM/ADaM recommendations
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import anthropic
from loguru import logger
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Default prompt file path (relative to project root)
_DEFAULT_PROMPT_FILE = "prompts/sdtm_adam_recommendation_prompt.txt"


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def _load_prompt_template(prompt_file: str) -> str:
    """Load the recommendation prompt template from disk."""
    prompt_path = Path(prompt_file)
    if not prompt_path.is_absolute():
        # Resolve relative to the project root (parent of this file's parent dir)
        project_root = Path(__file__).parent.parent
        prompt_path = project_root / prompt_file

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Recommendation prompt file not found: {prompt_path}"
        )

    return prompt_path.read_text(encoding="utf-8")


def _build_prompt(template: str, schema: dict) -> str:
    """Substitute the schema JSON into the prompt template."""
    schema_json = json.dumps(schema, indent=2, ensure_ascii=False)
    return template.replace("{SCHEMA_JSON}", schema_json)


# ---------------------------------------------------------------------------
# Claude API call (with tenacity retry)
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=5, max=120),
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
    Send the recommendation prompt to Claude and return the raw text response.

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


# ---------------------------------------------------------------------------
# JSON response parsing
# ---------------------------------------------------------------------------

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
        logger.error(
            f"[{nct_id}] Failed to parse recommendation response as JSON: {exc}"
        )
        logger.debug(
            f"[{nct_id}] Raw recommendation response (first 500 chars): "
            f"{raw_response[:500]}"
        )
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_rules(schema: dict, config: dict) -> dict:
    """
    Run the LLM-based CDISC recommendation engine on an extracted SAP schema.

    Makes a second Claude API call (Step 4b) with the canonical schema as input,
    asking Claude to produce SDTM domain and ADaM dataset recommendations with
    clinical rationale, traceability, per-component confidence scores, and any
    new CDISC implementation questions.

    Populates:
    - schema["sdtm_domains_expected"]: list of SDTM domain recommendation dicts
    - schema["adam_datasets_expected"]: list of ADaM dataset recommendation dicts
    - schema["component_confidence"]: dict of per-dataset confidence scores
    - schema["cdisc_open_questions"]: new CDISC implementation questions from LLM
    - schema["implementation_notes"]: free-text implementation summary

    On any failure (API error, JSON parse failure), logs the error and returns
    the schema unchanged rather than crashing the pipeline.

    Args:
        schema: Canonical SAP schema dict produced by step4_parse.py.
        config: Full pipeline config dict. Must contain config["claude_api"]
                with keys "model" and "max_tokens".

    Returns:
        The updated schema dict.
    """
    nct_id = schema.get("metadata", {}).get("nct_id", "UNKNOWN")
    logger.info(f"[{nct_id}] Running LLM recommendation engine (Step 4b)")

    claude_cfg = config.get("claude_api", {})
    model = claude_cfg.get("model", "claude-sonnet-4-5")
    max_tokens = claude_cfg.get("max_tokens", 4096)

    # Load the recommendation prompt template
    prompt_file = claude_cfg.get(
        "recommendation_prompt_file", _DEFAULT_PROMPT_FILE
    )
    try:
        prompt_template = _load_prompt_template(prompt_file)
    except FileNotFoundError as exc:
        logger.error(f"[{nct_id}] Cannot load recommendation prompt: {exc}")
        return schema

    prompt = _build_prompt(prompt_template, schema)

    # Initialize Anthropic client
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error(
            f"[{nct_id}] ANTHROPIC_API_KEY not set — skipping LLM recommendations."
        )
        return schema

    client = anthropic.Anthropic(api_key=api_key)

    logger.info(
        f"[{nct_id}] Sending schema to Claude for SDTM/ADaM recommendations "
        f"({model})"
    )

    try:
        raw_response = _call_claude(
            client=client,
            model=model,
            max_tokens=max_tokens,
            prompt=prompt,
        )
    except Exception as exc:
        logger.error(
            f"[{nct_id}] Claude API call failed after retries (Step 4b): {exc}"
        )
        return schema

    # Parse the JSON response
    recommendations = _parse_json_response(raw_response, nct_id)
    if recommendations is None:
        logger.error(
            f"[{nct_id}] LLM recommendation response could not be parsed — "
            f"leaving schema unchanged."
        )
        return schema

    # --- Merge recommendations into the schema ---

    sdtm_domains = recommendations.get("sdtm_domains_expected", [])
    adam_datasets = recommendations.get("adam_datasets_expected", [])
    cdisc_open_questions = recommendations.get("cdisc_open_questions", [])
    implementation_notes = recommendations.get("implementation_notes", "")

    schema["sdtm_domains_expected"] = sdtm_domains
    schema["adam_datasets_expected"] = adam_datasets
    schema["cdisc_open_questions"] = cdisc_open_questions
    schema["implementation_notes"] = implementation_notes

    # Build component_confidence from adam_datasets confidence scores
    schema["component_confidence"] = {
        ds["dataset"]: ds.get("confidence")
        for ds in adam_datasets
        if ds.get("dataset")
    }

    # --- Logging ---

    logger.info(
        f"[{nct_id}] LLM recommendations: "
        f"{len(sdtm_domains)} SDTM domains, "
        f"{len(adam_datasets)} ADaM datasets, "
        f"{len(cdisc_open_questions)} new CDISC questions"
    )

    # Log HIGH severity new questions so they surface prominently
    high_severity_qs = [
        q for q in cdisc_open_questions
        if q.get("severity", "").upper() == "HIGH"
    ]
    if high_severity_qs:
        logger.warning(
            f"[{nct_id}] {len(high_severity_qs)} HIGH severity CDISC "
            f"implementation question(s):"
        )
        for q in high_severity_qs:
            dataset = q.get("dataset", q.get("category", "?"))
            logger.warning(f"[{nct_id}]   [{dataset}] {q.get('question', '')}")

    # Log confidence scores
    if schema["component_confidence"]:
        scores_str = ", ".join(
            f"{k}: {v}%"
            for k, v in sorted(schema["component_confidence"].items())
            if v is not None
        )
        if scores_str:
            logger.info(f"[{nct_id}] Component confidence: {scores_str}")

    return schema
