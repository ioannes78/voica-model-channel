#!/usr/bin/env python3
import hashlib
import json
import pathlib
import re
import sys
from urllib.parse import urlparse

SHA256 = re.compile(r"^[0-9a-f]{64}$")
SAFE = re.compile(r"^[A-Za-z0-9._-]+$")
REQUIRED_BASELINE = {
    "modelId": "silero-vad-int8",
    "kind": "VAD",
    "version": "2025-07-11",
    "revision": 1,
    "sourceType": "BUILTIN_WITH_OVERRIDE",
}

def fail(message: str) -> None:
    raise SystemExit("manifest validation failed: " + message)

def compact_models(models) -> bytes:
    return json.dumps(
        models,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

def require_https(value: str, field: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        fail(field + " must be HTTPS: " + value)

def validate_file_entry(entry, model_id: str) -> None:
    for key in ("relativePath", "packagePath", "sizeBytes", "sha256"):
        if key not in entry:
            fail(model_id + " file missing " + key)
    for key in ("relativePath", "packagePath"):
        value = entry[key]
        if not isinstance(value, str) or not value or value.startswith(("/", "\\")):
            fail(model_id + " unsafe " + key)
        normalized = value.replace("\\", "/")
        if ":" in normalized or ".." in normalized.split("/"):
            fail(model_id + " unsafe " + key + ": " + value)
    if not isinstance(entry["sizeBytes"], int) or entry["sizeBytes"] < 0:
        fail(model_id + " invalid file size")
    if not SHA256.fullmatch(str(entry["sha256"]).lower()):
        fail(model_id + " invalid file sha256")

def validate_model(model) -> None:
    required = [
        "modelId", "kind", "displayName", "version", "revision", "runtimeId",
        "languages", "capabilities", "sourceType", "installedSizeBytes",
        "files", "compatibility", "license", "sourceUrl", "releaseChannel",
        "autoUpdateEligible", "deprecated", "criticalUpdate",
    ]
    for key in required:
        if key not in model:
            fail(str(model.get("modelId", "<unknown>")) + " missing " + key)

    model_id = model["modelId"]
    if not isinstance(model_id, str) or not SAFE.fullmatch(model_id):
        fail("unsafe modelId")
    if not isinstance(model["version"], str) or not SAFE.fullmatch(model["version"]):
        fail(model_id + " unsafe version")
    if not isinstance(model["revision"], int) or model["revision"] < 1:
        fail(model_id + " invalid revision")
    if not isinstance(model["languages"], list) or not model["languages"]:
        fail(model_id + " languages must be non-empty")
    if not isinstance(model["files"], list) or not model["files"]:
        fail(model_id + " files must be non-empty")

    speaker_role = model.get("speakerRole")
    if model["kind"] == "SPEAKER":
        if speaker_role not in ("DIARIZATION_SEGMENTATION", "EMBEDDING"):
            fail(model_id + " SPEAKER model requires a valid speakerRole")
    elif speaker_role is not None:
        fail(model_id + " speakerRole is only valid for SPEAKER models")

    for entry in model["files"]:
        validate_file_entry(entry, model_id)

    require_https(model["sourceUrl"], model_id + ".sourceUrl")
    if model.get("homepage"):
        require_https(model["homepage"], model_id + ".homepage")
    if model["license"].get("url"):
        require_https(model["license"]["url"], model_id + ".license.url")

    source_type = model["sourceType"]
    download = model.get("download")
    if source_type == "MANAGED_DOWNLOAD":
        if not isinstance(download, dict):
            fail(model_id + " managed model requires download")
    if download is not None:
        for key in ("packageFormat", "url", "mirrors", "sizeBytes", "sha256"):
            if key not in download:
                fail(model_id + " download missing " + key)
        require_https(download["url"], model_id + ".download.url")
        for mirror in download["mirrors"]:
            require_https(mirror, model_id + ".download.mirror")
        if not isinstance(download["sizeBytes"], int) or download["sizeBytes"] < 0:
            fail(model_id + " invalid download size")
        if not SHA256.fullmatch(str(download["sha256"]).lower()):
            fail(model_id + " invalid package sha256")

def validate(path: pathlib.Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("catalogVersion") != 1:
        fail("catalogVersion must be 1")
    if not isinstance(data.get("manifestVersion"), int) or data["manifestVersion"] < 1:
        fail("manifestVersion must be >= 1")
    if data.get("channel") != "production":
        fail("channel must be production")

    models = data.get("models")
    if not isinstance(models, list) or not models:
        fail("models must be a non-empty array")
    ids = [m.get("modelId") for m in models]
    if len(ids) != len(set(ids)):
        fail("duplicate modelId")

    actual = hashlib.sha256(compact_models(models)).hexdigest()
    declared = str(data.get("manifestDigest", "")).lower()
    if actual != declared:
        fail("manifestDigest mismatch: declared=" + declared + " actual=" + actual)

    baseline = next((m for m in models if m.get("modelId") == REQUIRED_BASELINE["modelId"]), None)
    if baseline is None:
        fail("Silero baseline removed")
    for key, expected in REQUIRED_BASELINE.items():
        if baseline.get(key) != expected:
            fail("Silero baseline changed " + key)

    for model in models:
        validate_model(model)

    print(path.as_posix() + ": OK (" + str(len(models)) + " models, digest " + actual + ")")

if __name__ == "__main__":
    paths = [pathlib.Path(p) for p in sys.argv[1:]]
    if not paths:
        paths = sorted(pathlib.Path("manifests").glob("*.json"))
    if not paths:
        fail("no manifests found")
    for path in paths:
        validate(path)
