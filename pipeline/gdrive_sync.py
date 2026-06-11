"""
Google Drive sync module for Standards Gate.

Keeps the knowledge base (master_index.csv + parsed_schemas JSONs) persistent
across Streamlit Cloud redeploys by syncing to/from a Google Drive folder.

Credentials are read from (in order):
  1. Streamlit secrets: st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"]  (JSON string)
  2. Environment variable: GOOGLE_SERVICE_ACCOUNT_JSON

If neither is present, all operations are no-ops and is_configured() returns False.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def _gdrive_config() -> dict:
    """Load google_drive section from config.yaml."""
    try:
        import yaml
        config_path = Path(__file__).parent.parent / "config.yaml"
        with open(config_path, encoding="utf-8") as fh:
            return yaml.safe_load(fh).get("google_drive", {})
    except Exception as exc:
        logger.error(f"[gdrive_sync] Could not load config.yaml: {exc}")
        return {}
SCOPES = ["https://www.googleapis.com/auth/drive"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_credentials_json():
    """
    Return the service-account credentials from secrets or env, or None.
    May return a dict (if stored as TOML object in Streamlit secrets) or a str.
    """
    # 1. Streamlit secrets (available only when running inside Streamlit)
    try:
        import streamlit as st  # noqa: PLC0415 – optional import
        secret = st.secrets.get("GOOGLE_SERVICE_ACCOUNT_JSON")
        if secret:
            return secret  # may be dict or string — _get_drive_service handles both
    except Exception:
        pass  # Not running in Streamlit, or secrets not configured

    # 2. Environment variable
    env_val = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if env_val:
        return env_val

    return None


def _get_drive_service():
    """
    Build and return a Google Drive API service object.

    Returns None if credentials are unavailable (graceful degradation).
    """
    creds_json = _load_credentials_json()
    if not creds_json is None and creds_json == "":
        logger.error("[gdrive_sync] GOOGLE_SERVICE_ACCOUNT_JSON is empty.")
        return None
    if creds_json is None:
        logger.error("[gdrive_sync] GOOGLE_SERVICE_ACCOUNT_JSON not found in secrets or env.")
        return None

    try:
        from google.oauth2.service_account import Credentials  # noqa: PLC0415
        from googleapiclient.discovery import build  # noqa: PLC0415

        # Normalize to plain dict — Streamlit secrets returns AttrDict or str
        if isinstance(creds_json, str):
            creds_dict = json.loads(creds_json)
        else:
            # AttrDict / dict-like — round-trip through JSON to get a plain dict
            creds_dict = json.loads(json.dumps(dict(creds_json)))

        creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
        service = build("drive", "v3", credentials=creds, cache_discovery=False)
        logger.info("[gdrive_sync] Drive service built successfully.")
        return service
    except Exception as exc:
        logger.error(f"[gdrive_sync] Failed to build Drive service: {exc}", exc_info=True)
        return None


def _find_file_in_folder(drive_service, filename: str, parent_id: str) -> str | None:
    """Return the file ID of *filename* inside *parent_id*, or None if not found."""
    try:
        results = drive_service.files().list(
            q=f"name='{filename}' and '{parent_id}' in parents and trashed=false",
            fields="files(id, name)",
        ).execute()
        files = results.get("files", [])
        return files[0]["id"] if files else None
    except Exception as exc:
        logger.error(f"[gdrive_sync] Error querying Drive for '{filename}': {exc}")
        return None


def _upload_file(
    drive_service,
    local_path: Path,
    filename: str,
    parent_id: str,
    mime_type: str = "application/octet-stream",
) -> tuple[bool, str]:
    """
    Upload *local_path* to Drive under *parent_id* as *filename*.
    Returns (True, "") on success, (False, error_message) on failure.
    """
    from googleapiclient.http import MediaIoBaseUpload  # noqa: PLC0415

    try:
        data = local_path.read_bytes()
        media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mime_type, resumable=False)

        existing_id = _find_file_in_folder(drive_service, filename, parent_id)
        if existing_id:
            drive_service.files().update(
                fileId=existing_id,
                media_body=media,
            ).execute()
            logger.debug(f"[gdrive_sync] Updated '{filename}' (id={existing_id})")
        else:
            drive_service.files().create(
                body={"name": filename, "parents": [parent_id]},
                media_body=media,
            ).execute()
            logger.debug(f"[gdrive_sync] Created '{filename}' in folder {parent_id}")

        return True, ""
    except Exception as exc:
        msg = f"Failed to upload '{filename}': {exc}"
        logger.error(f"[gdrive_sync] {msg}")
        return False, msg


def _download_file(drive_service, file_id: str, dest_path: Path) -> bool:
    """
    Download *file_id* from Drive to *dest_path*.

    Returns True on success.
    """
    try:
        request = drive_service.files().get_media(fileId=file_id)
        data = request.execute()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(data)
        logger.debug(f"[gdrive_sync] Downloaded file_id={file_id} → {dest_path}")
        return True
    except Exception as exc:
        logger.error(f"[gdrive_sync] Failed to download file_id={file_id}: {exc}")
        return False


def _list_files_in_folder(drive_service, parent_id: str) -> list[dict]:
    """Return a list of {id, name} dicts for all non-trashed files in *parent_id*."""
    try:
        results = drive_service.files().list(
            q=f"'{parent_id}' in parents and trashed=false",
            fields="files(id, name)",
        ).execute()
        return results.get("files", [])
    except Exception as exc:
        logger.error(f"[gdrive_sync] Failed to list files in folder {parent_id}: {exc}")
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_configured() -> bool:
    """Return True if Google Drive credentials are available."""
    return _load_credentials_json() is not None


def _update_file_by_id(drive_service, file_id: str, data: bytes, mime_type: str) -> tuple[bool, str]:
    """Update an existing Drive file by ID. Never creates — avoids service-account quota issue."""
    from googleapiclient.http import MediaIoBaseUpload  # noqa: PLC0415
    try:
        media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mime_type, resumable=False)
        drive_service.files().update(fileId=file_id, media_body=media).execute()
        logger.debug(f"[gdrive_sync] Updated file id={file_id}")
        return True, ""
    except Exception as exc:
        msg = str(exc)
        logger.error(f"[gdrive_sync] Failed to update file id={file_id}: {msg}")
        return False, msg


def upload_knowledge_base(base_path: Path) -> bool:
    """
    Upload the knowledge base to Google Drive by updating pre-created files.

    - master_index.csv  → MASTER_INDEX_FILE_ID
    - parsed_schemas/*.json bundled → SCHEMAS_BUNDLE_FILE_ID  (single JSON dict)

    Returns True on success, raises RuntimeError with details on failure.
    """
    cfg = _gdrive_config()
    drive_service = _get_drive_service()
    if drive_service is None:
        raise RuntimeError("Could not build Drive service — check GOOGLE_SERVICE_ACCOUNT_JSON secret.")

    errors: list[str] = []

    # --- master_index.csv ---
    index_path = base_path / "master_index.csv"
    if index_path.exists():
        logger.info("[gdrive_sync] Uploading master_index.csv…")
        ok, err = _update_file_by_id(drive_service, cfg["master_index_file_id"], index_path.read_bytes(), "text/csv")
        if ok:
            logger.info("[gdrive_sync] master_index.csv updated.")
        else:
            errors.append(f"master_index.csv: {err}")
    else:
        logger.warning(f"[gdrive_sync] master_index.csv not found at {index_path} — skipping.")

    # --- schemas bundle (all JSONs merged into one file) ---
    schemas_dir = base_path / "parsed_schemas"
    if schemas_dir.is_dir():
        json_files = list(schemas_dir.glob("*.json"))
        if json_files:
            bundle = {}
            for jp in json_files:
                try:
                    bundle[jp.stem] = json.loads(jp.read_text(encoding="utf-8"))
                except Exception:
                    pass
            logger.info(f"[gdrive_sync] Uploading schemas bundle ({len(bundle)} schemas)…")
            ok, err = _update_file_by_id(
                drive_service, cfg["schemas_bundle_file_id"],
                json.dumps(bundle, ensure_ascii=False, indent=2).encode("utf-8"),
                "application/json",
            )
            if ok:
                logger.info("[gdrive_sync] schemas_bundle.json updated.")
            else:
                errors.append(f"schemas_bundle.json: {err}")
        else:
            logger.info("[gdrive_sync] No JSON schemas found — skipping bundle upload.")
    else:
        logger.warning(f"[gdrive_sync] parsed_schemas/ not found at {schemas_dir} — skipping.")

    if errors:
        raise RuntimeError("Drive upload errors:\n" + "\n".join(errors))

    return True


def download_knowledge_base(base_path: Path) -> bool:
    """
    Download the knowledge base from Google Drive.

    Downloads:
      - ``master_index.csv`` from the root folder  → ``base_path/master_index.csv``
      - All JSON files from parsed_schemas folder  → ``base_path/parsed_schemas/``

    Skips files that already exist locally (idempotent).
    Creates directories as needed.

    Returns True if at least one file was downloaded, False otherwise.
    """
    drive_service = _get_drive_service()
    if drive_service is None:
        logger.warning("[gdrive_sync] download_knowledge_base called but Drive is not configured.")
        return False

    cfg = _gdrive_config()
    any_downloaded = False
    base_path.mkdir(parents=True, exist_ok=True)

    # --- master_index.csv (always overwrite on cold start to get latest) ---
    index_dest = base_path / "master_index.csv"
    logger.info("[gdrive_sync] Downloading master_index.csv…")
    ok = _download_file(drive_service, cfg["master_index_file_id"], index_dest)
    if ok:
        logger.info("[gdrive_sync] master_index.csv downloaded.")
        any_downloaded = True
    else:
        logger.error("[gdrive_sync] Failed to download master_index.csv.")

    # --- schemas bundle → unpack into parsed_schemas/ ---
    schemas_dest = base_path / "parsed_schemas"
    schemas_dest.mkdir(parents=True, exist_ok=True)
    bundle_tmp = base_path / "_schemas_bundle_tmp.json"
    ok = _download_file(drive_service, cfg["schemas_bundle_file_id"], bundle_tmp)
    if ok:
        try:
            bundle = json.loads(bundle_tmp.read_text(encoding="utf-8"))
            for nct_id, schema in bundle.items():
                dest = schemas_dest / f"{nct_id}.json"
                dest.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
            bundle_tmp.unlink(missing_ok=True)
            logger.info(f"[gdrive_sync] Unpacked {len(bundle)} schemas from bundle.")
            if bundle:
                any_downloaded = True
        except Exception as exc:
            logger.error(f"[gdrive_sync] Failed to unpack schemas bundle: {exc}")
    else:
        logger.info("[gdrive_sync] schemas_bundle.json not found or empty — skipping.")

    return any_downloaded
