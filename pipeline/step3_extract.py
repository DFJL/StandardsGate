"""
Step 3: Extract text from SAP PDFs.

Preferred extractor: pdfplumber (handles multi-column layouts better, preserves
reading order more faithfully for clinical trial documents).
Fallback extractor: pypdf (pure-Python, works when pdfplumber fails).

Output: .txt file per study in the extracted_text subdirectory.
"""

from pathlib import Path
from typing import Optional
from loguru import logger


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _extract_with_pdfplumber(pdf_path: Path) -> Optional[str]:
    """
    Extract text from a PDF using pdfplumber.

    pdfplumber is preferred because it handles:
    - Multi-column layouts (clinical SAPs often have two-column sections)
    - Tables (exposure summaries, population definitions)
    - More accurate character-level positioning

    Returns the concatenated text of all pages, or None on failure.
    """
    try:
        import pdfplumber  # local import so pypdf-only environments still work
    except ImportError:
        logger.warning("pdfplumber not installed; will use pypdf fallback.")
        return None

    try:
        pages_text = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # extract_text() with x_tolerance/y_tolerance helps with
                # character-spaced headers common in SAP documents
                text = page.extract_text(x_tolerance=3, y_tolerance=3)
                if text:
                    pages_text.append(f"[PAGE {page_num}]\n{text}")
                else:
                    logger.debug(f"  Page {page_num} yielded no text (possibly image-based).")

        if not pages_text:
            logger.warning(f"pdfplumber extracted no text from {pdf_path.name}")
            return None

        return "\n\n".join(pages_text)

    except Exception as exc:
        logger.warning(f"pdfplumber failed on {pdf_path.name}: {exc}")
        return None


def _extract_with_pypdf(pdf_path: Path) -> Optional[str]:
    """
    Fallback text extraction using pypdf.

    Less accurate for multi-column layouts but works on a wider range of PDFs
    (including some that pdfplumber chokes on due to malformed xref tables).

    Returns extracted text or None on failure.
    """
    try:
        from pypdf import PdfReader  # pypdf ≥ 3.x uses this import path
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # older name
        except ImportError:
            logger.error("Neither pypdf nor PyPDF2 is installed.")
            return None

    try:
        reader = PdfReader(str(pdf_path))
        pages_text = []
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text and text.strip():
                pages_text.append(f"[PAGE {page_num}]\n{text}")

        if not pages_text:
            logger.warning(f"pypdf extracted no text from {pdf_path.name}")
            return None

        return "\n\n".join(pages_text)

    except Exception as exc:
        logger.warning(f"pypdf failed on {pdf_path.name}: {exc}")
        return None


def extract_text_from_pdf(pdf_path: Path) -> Optional[str]:
    """
    Extract text from a PDF, trying pdfplumber first and falling back to pypdf.

    Returns the extracted text string, or None if both extractors fail.
    """
    logger.debug(f"Attempting pdfplumber extraction for {pdf_path.name}")
    text = _extract_with_pdfplumber(pdf_path)

    if text is None:
        logger.info(f"Falling back to pypdf for {pdf_path.name}")
        text = _extract_with_pypdf(pdf_path)

    return text


# ---------------------------------------------------------------------------
# Per-study extraction
# ---------------------------------------------------------------------------

def extract_study_text(
    config: dict,
    nct_id: str,
    pdf_path: Path,
    text_dir: Path,
) -> Optional[Path]:
    """
    Extract text from a single study's SAP PDF and save as a .txt file.

    Returns the Path to the saved text file, or None on failure.

    Args:
        config:    Full pipeline config dict.
        nct_id:    NCT identifier (used for file naming and logging).
        pdf_path:  Path to the PDF file.
        text_dir:  Directory where the .txt output should be written.
    """
    overwrite = config["storage"]["overwrite_existing"]
    txt_path = text_dir / f"{nct_id}.txt"

    # Idempotency: skip if already extracted
    if txt_path.exists() and not overwrite:
        logger.info(f"[{nct_id}] Text file already exists, skipping extraction.")
        return txt_path

    if not pdf_path.exists():
        logger.error(f"[{nct_id}] PDF not found at {pdf_path}")
        return None

    logger.info(f"[{nct_id}] Extracting text from {pdf_path.name}")
    text = extract_text_from_pdf(pdf_path)

    if text is None:
        logger.error(f"[{nct_id}] Text extraction failed completely.")
        return None

    char_count = len(text)
    word_count = len(text.split())
    logger.info(
        f"[{nct_id}] Extracted {char_count:,} characters / {word_count:,} words"
    )

    # Sanity check: a real SAP should have at least a few thousand words
    if word_count < 500:
        logger.warning(
            f"[{nct_id}] Suspiciously short extraction ({word_count} words). "
            "Document may be scanned/image-based or nearly empty."
        )

    txt_path.write_text(text, encoding="utf-8")
    logger.info(f"[{nct_id}] Saved extracted text → {txt_path.name}")
    return txt_path


# ---------------------------------------------------------------------------
# Step entry point
# ---------------------------------------------------------------------------

def run(config: dict, download_results: list[dict]) -> list[dict]:
    """
    Main entry point for Step 3.

    For each successfully downloaded PDF, extract the text and save it.

    Returns a list of enriched result dicts that add 'text_path' and
    'word_count' fields to the incoming download_results entries.
    """
    storage_cfg = config["storage"]
    base_path = Path(storage_cfg["base_path"])
    text_dir = base_path / storage_cfg["subdirs"]["extracted_text"]
    text_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Step 3: Extracting text from {len(download_results)} PDFs → {text_dir}"
    )

    enriched = []

    for i, meta in enumerate(download_results, 1):
        nct_id = meta.get("nct_id", "UNKNOWN")
        pdf_path_str = meta.get("pdf_path")

        if not pdf_path_str:
            logger.warning(f"[{nct_id}] No pdf_path in metadata, skipping.")
            continue

        pdf_path = Path(pdf_path_str)
        logger.info(f"  [{i}/{len(download_results)}] {nct_id}")

        txt_path = extract_study_text(config, nct_id, pdf_path, text_dir)

        if txt_path is None:
            logger.warning(f"[{nct_id}] Skipping due to extraction failure.")
            continue

        # Count words for downstream reporting
        try:
            word_count = len(txt_path.read_text(encoding="utf-8").split())
        except Exception:
            word_count = 0

        enriched.append({
            **meta,
            "text_path": str(txt_path),
            "word_count": word_count,
        })

    logger.info(
        f"Step 3 complete. Extracted text for {len(enriched)}/{len(download_results)} studies."
    )
    return enriched
