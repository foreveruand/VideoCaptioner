import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_MAP_FILE_NAME = ".workdir_name_map.json"
_MAP_LOCK = threading.Lock()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_key(source_path: str) -> str:
    normalized = os.path.normpath(os.path.abspath(source_path))
    return os.path.normcase(normalized)


def _map_file(work_dir: str) -> Path:
    return Path(work_dir) / _MAP_FILE_NAME


def _load_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "source_to_short": {}, "short_to_meta": {}}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        source_to_short = raw.get("source_to_short", {})
        short_to_meta = raw.get("short_to_meta", {})
        if not isinstance(source_to_short, dict) or not isinstance(short_to_meta, dict):
            raise ValueError("Invalid mapping schema")
        return {
            "version": 1,
            "source_to_short": source_to_short,
            "short_to_meta": short_to_meta,
        }
    except Exception:
        # Mapping damage should not block the pipeline; rebuild from scratch.
        return {"version": 1, "source_to_short": {}, "short_to_meta": {}}


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _build_short_name(prefix: str, source_key: str) -> str:
    digest = hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def get_or_create_work_dir_short_name(
    source_path: str, work_dir: str, prefix: str = "video"
) -> str:
    """Get or create a persistent short name for work-dir storage.

    The mapping is stored in ``<work_dir>/.workdir_name_map.json``.
    """
    source_path = str(source_path)
    source_key = _source_key(source_path)
    path = _map_file(work_dir)

    with _MAP_LOCK:
        mapping = _load_mapping(path)
        source_to_short = mapping["source_to_short"]
        short_to_meta = mapping["short_to_meta"]

        existing = source_to_short.get(source_key)
        if isinstance(existing, str) and existing:
            meta = short_to_meta.get(existing, {})
            if isinstance(meta, dict):
                meta["last_used_at"] = _utc_now_iso()
                short_to_meta[existing] = meta
                _atomic_write_json(path, mapping)
            return existing

        short_name = _build_short_name(prefix=prefix, source_key=source_key)
        if short_name in short_to_meta:
            # Rare collision handling.
            suffix = 1
            while f"{short_name}_{suffix}" in short_to_meta:
                suffix += 1
            short_name = f"{short_name}_{suffix}"

        source_to_short[source_key] = short_name
        short_to_meta[short_name] = {
            "source_path": source_path,
            "created_at": _utc_now_iso(),
            "last_used_at": _utc_now_iso(),
        }
        _atomic_write_json(path, mapping)
        return short_name
