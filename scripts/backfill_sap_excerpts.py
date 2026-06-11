"""
Backfill sap_section and sap_excerpt fields on existing KB schemas.

Reads each parsed schema from sap_knowledge_base/parsed_schemas/,
finds the matching extracted text in sap_knowledge_base/extracted_text/,
runs local fuzzy keyword matching (no API calls), and writes the fields
back to the schema JSON in-place.

Usage:
    python scripts/backfill_sap_excerpts.py
    python scripts/backfill_sap_excerpts.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

KB = ROOT / "sap_knowledge_base"
SCHEMAS_DIR = KB / "parsed_schemas"
TEXT_DIR = KB / "extracted_text"

# ── Same stop-words as app.py _find_sap_passage ──────────────────────────────
_STOP = {"the", "a", "an", "in", "of", "for", "to", "is", "are", "be",
         "was", "not", "this", "that", "with", "and", "or", "how", "what",
         "which", "whether", "does", "should", "will", "by", "at", "from"}

_SECTION_RE = re.compile(
    r"(?:^|\n)\s*(\d+(?:\.\d+)*\.?\s+[A-Z][^\n]{3,60})", re.MULTILINE
)


def _find_passage(text: str, query: str, window: int = 450) -> str | None:
    words = [w.strip("?.,:;()") for w in query.lower().split()
             if w.lower() not in _STOP and len(w) > 3]
    if not words:
        return None
    text_lower = text.lower()
    best_pos, best_hits = 0, 0
    step = max(1, len(text) // 400)
    for i in range(0, max(1, len(text) - window), step):
        chunk = text_lower[i: i + window]
        hits = sum(1 for w in words if w in chunk)
        if hits > best_hits:
            best_hits, best_pos = hits, i
    if best_hits == 0:
        return None
    start = max(0, best_pos)
    end = min(len(text), start + window)
    excerpt = text[start:end].strip()
    if ". " in excerpt[20:]:
        excerpt = excerpt[excerpt.index(". ", 20) + 2:]
    if ". " in excerpt[50:]:
        last = excerpt.rindex(". ", 0, -10)
        excerpt = excerpt[: last + 1]
    return ("…" if start > 0 else "") + excerpt + "…"


def _find_section(text: str, pos: int) -> str:
    """Return the nearest section heading at or before `pos`."""
    best_section = ""
    for m in _SECTION_RE.finditer(text):
        if m.start() <= pos:
            best_section = m.group(1).strip()
        else:
            break
    return best_section


def backfill_schema(schema_path: Path, text_path: Path, dry_run: bool) -> int:
    """Return number of questions updated."""
    with open(schema_path, encoding="utf-8") as fh:
        schema = json.load(fh)

    sap_text = text_path.read_text(encoding="utf-8")
    open_qs = schema.get("open_questions") or []
    updated = 0

    for q in open_qs:
        needs_excerpt = not q.get("sap_excerpt")
        needs_section = not q.get("sap_section")
        if not (needs_excerpt or needs_section):
            continue

        question = q.get("question") or q.get("category") or ""
        passage = _find_passage(sap_text, question)
        if not passage:
            continue

        # Estimate position for section lookup
        pos = max(0, sap_text.lower().find(
            next((w for w in question.lower().split() if len(w) > 5), ""), 0
        ))

        if needs_excerpt:
            q["sap_excerpt"] = passage
        if needs_section:
            section = _find_section(sap_text, pos)
            if section:
                q["sap_section"] = section
        updated += 1

    if updated > 0 and not dry_run:
        with open(schema_path, "w", encoding="utf-8") as fh:
            json.dump(schema, fh, ensure_ascii=False, indent=2)

    return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without writing files")
    args = parser.parse_args()

    if not SCHEMAS_DIR.exists():
        print(f"No schemas directory found at {SCHEMAS_DIR}")
        sys.exit(1)

    schema_files = sorted(SCHEMAS_DIR.glob("*.json"))
    print(f"Found {len(schema_files)} schema(s) in {SCHEMAS_DIR}")

    total_updated = 0
    for sf in schema_files:
        nct_id = sf.stem
        text_file = TEXT_DIR / f"{nct_id}.txt"
        if not text_file.exists():
            print(f"  [{nct_id}] No text file found — skipping")
            continue
        n = backfill_schema(sf, text_file, dry_run=args.dry_run)
        if n:
            tag = "(dry-run)" if args.dry_run else "✓"
            print(f"  [{nct_id}] {tag} Updated {n} open question(s)")
            total_updated += 1
        else:
            print(f"  [{nct_id}] Already has excerpts or no text matches — skipped")

    print(f"\nDone. {total_updated} schema(s) {'would be' if args.dry_run else 'were'} updated.")


if __name__ == "__main__":
    main()
