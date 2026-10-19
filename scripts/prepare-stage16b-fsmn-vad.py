#!/usr/bin/env python3
import hashlib
import io
import json
import pathlib
import sys
import tarfile
import urllib.request

UPSTREAM_REVISION = "eeb66f413b94a33458f7edbe7f501b10f20fcc85"
MODEL_ID = "fsmn-vad-int8"
VERSION = "eeb66f4"
REVISION = 1
RUNTIME_ID = "voica-fsmn-ort"
RUNTIME_VERSION = "1.0.0"
RELEASE_TAG = "candidate-fsmn-vad-r1"
PACKAGE_NAME = "fsmn-vad-int8-r1.tar.bz2"
PACKAGE_ROOT = "fsmn-vad-int8-r1"
RELEASE_BASE = (
    "https://github.com/ioannes78/voica-model-channel/releases/download/"
    + RELEASE_TAG
)
UPSTREAM_BASE = (
    "https://huggingface.co/funasr/fsmn-vad-onnx/resolve/"
    + UPSTREAM_REVISION
)
UPSTREAM_PAGE = (
    "https://huggingface.co/funasr/fsmn-vad-onnx/tree/" + UPSTREAM_REVISION
)

FILES = [
    {
        "relativePath": "model_quant.onnx",
        "url": UPSTREAM_BASE + "/model_quant.onnx",
        "sizeBytes": 508876,
        "sha256": "9b28837838fce9685503c63139fadbad35d6c8ed485485dafdbb32e725969660",
    },
    {
        "relativePath": "vad.mvn",
        "url": UPSTREAM_BASE + "/vad.mvn",
        "sizeBytes": 8040,
        "sha256": "6820fef9687708c4fc3fab2530179c8fcea6262daa25514380056cd8f6eb1754",
    },
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "voica-model-channel-stage16b/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def deterministic_tar_bz2(path: pathlib.Path, files: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:bz2", format=tarfile.PAX_FORMAT) as archive:
        for relative_path in sorted(files):
            data = files[relative_path]
            info = tarfile.TarInfo(f"{PACKAGE_ROOT}/{relative_path}")
            info.size = len(data)
            info.mode = 0o644
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            archive.addfile(info, io.BytesIO(data))


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: prepare-stage16b-fsmn-vad.py OUT_DIR")

    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)

    downloaded: dict[str, bytes] = {}
    descriptors = []
    for source in FILES:
        data = download(source["url"])
        if len(data) != source["sizeBytes"]:
            raise SystemExit(
                f"upstream size mismatch for {source['relativePath']}: "
                f"expected {source['sizeBytes']}, got {len(data)}"
            )
        digest = sha256_bytes(data)
        if digest != source["sha256"]:
            raise SystemExit(
                f"upstream SHA-256 mismatch for {source['relativePath']}: "
                f"expected {source['sha256']}, got {digest}"
            )
        downloaded[source["relativePath"]] = data
        descriptors.append(
            {
                "relativePath": source["relativePath"],
                "sizeBytes": source["sizeBytes"],
                "sha256": source["sha256"],
                "packagePath": f"{PACKAGE_ROOT}/{source['relativePath']}",
            }
        )

    package = out / PACKAGE_NAME
    deterministic_tar_bz2(package, downloaded)
    package_size = package.stat().st_size
    package_sha = sha256_file(package)

    descriptor = {
        "modelId": MODEL_ID,
        "kind": "VAD",
        "displayName": "FSMN VAD INT8",
        "version": VERSION,
        "revision": REVISION,
        "runtimeId": RUNTIME_ID,
        "runtimeVersionMin": RUNTIME_VERSION,
        "runtimeVersionMax": None,
        "languages": ["und"],
        "capabilities": {
            "supportsStreaming": False,
            "supportsPartial": False,
            "supportsTokenTiming": False,
            "supportsLanguageDetection": False,
            "supportsConfidence": False,
            "supportsInverseTextNormalization": False,
            "supportsSecondPass": False,
            "supportsHotwords": False,
        },
        "sourceType": "MANAGED_DOWNLOAD",
        "builtinAssetPath": None,
        "download": {
            "packageFormat": "TAR_BZ2",
            "url": f"{RELEASE_BASE}/{PACKAGE_NAME}",
            "mirrors": [],
            "sizeBytes": package_size,
            "sha256": package_sha,
        },
        "installedSizeBytes": sum(item["sizeBytes"] for item in descriptors),
        "files": descriptors,
        "compatibility": {
            "abis": ["arm64-v8a"],
            "minSdk": 26,
            "appVersionMin": 92,
            "appVersionMax": None,
        },
        "license": {
            "id": "Apache-2.0",
            "url": "https://www.apache.org/licenses/LICENSE-2.0",
            "attribution": (
                "FunASR FSMN-VAD ONNX INT8; upstream revision " + UPSTREAM_REVISION
            ),
            "redistributionPolicy": "VOICA_MIRROR_ALLOWED",
        },
        "sourceUrl": UPSTREAM_PAGE,
        "homepage": "https://huggingface.co/funasr/fsmn-vad-onnx",
        "releaseChannel": "production",
        "autoUpdateEligible": False,
        "deprecated": False,
        "criticalUpdate": False,
        "quantization": "INT8",
    }

    descriptor_path = out / "model-descriptor.json"
    descriptor_path.write_text(
        json.dumps(descriptor, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    package_metadata = {
        "upstreamRevision": UPSTREAM_REVISION,
        "packageName": PACKAGE_NAME,
        "packageSizeBytes": package_size,
        "packageSha256": package_sha,
        "files": descriptors,
    }
    (out / "package-metadata.json").write_text(
        json.dumps(package_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(package_metadata, ensure_ascii=False, indent=2))
    print(json.dumps(descriptor, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
