#!/usr/bin/env python3
import hashlib
import json
import pathlib
import shutil
import sys
import tarfile
import tempfile
import urllib.request

MODELS = [
    {
        "key": "transducer",
        "modelId": "zipformer-large-zh-transducer-int8",
        "displayName": "Chinese Large Zipformer Transducer INT8",
        "version": "2025-06-30",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30.tar.bz2",
        "expectedSize": 132634597,
        "expectedSha256": "5a2832047ea1f97dd0dc595b816c230c4bafad65cfc0341fa57517cadc50afd0",
        "root": "sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30",
        "files": [
            ("encoder.int8.onnx", "encoder.int8.onnx"),
            ("decoder.onnx", "decoder.onnx"),
            ("joiner.int8.onnx", "joiner.int8.onnx"),
            ("tokens.txt", "tokens.txt"),
        ],
        "runtimeModelType": "zipformer2-transducer",
        "attribution": "k2-fsa / icefall Chinese Large streaming Zipformer Transducer INT8 2025-06-30",
    },
    {
        "key": "ctc",
        "modelId": "zipformer-large-zh-ctc-int8",
        "displayName": "Chinese Large Zipformer CTC INT8",
        "version": "2025-06-30",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-ctc-zh-int8-2025-06-30.tar.bz2",
        "expectedSize": 127965713,
        "expectedSha256": "f2ab7a5deb02717801f6a5b26c751b42f8a2db891b07f5b095e6da7442081448",
        "root": "sherpa-onnx-streaming-zipformer-ctc-zh-int8-2025-06-30",
        "files": [
            ("model.int8.onnx", "model.int8.onnx"),
            ("tokens.txt", "tokens.txt"),
        ],
        "runtimeModelType": "zipformer2-ctc",
        "attribution": "k2-fsa / icefall Chinese Large streaming Zipformer CTC INT8 2025-06-30",
    },
]

def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def download(url: str, target: pathlib.Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "voica-model-channel-stage13a"},
    )
    with urllib.request.urlopen(request, timeout=180) as response, target.open("wb") as dst:
        shutil.copyfileobj(response, dst, 1024 * 1024)

def inspect_files(cfg, archive: pathlib.Path):
    descriptors = []
    with tarfile.open(archive, "r:bz2") as tar:
        members = {m.name: m for m in tar.getmembers() if m.isfile()}
        for installed, source_name in cfg["files"]:
            package_path = cfg["root"] + "/" + source_name
            member = members.get(package_path)
            if member is None:
                available = sorted(name for name in members if name.endswith("/" + source_name))
                raise SystemExit(
                    cfg["modelId"] + " missing package path " + package_path
                    + "; matching paths=" + repr(available)
                )
            extracted = tar.extractfile(member)
            if extracted is None:
                raise SystemExit("cannot extract: " + package_path)
            h = hashlib.sha256()
            size = 0
            while True:
                chunk = extracted.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                h.update(chunk)
            descriptors.append({
                "relativePath": installed,
                "sizeBytes": size,
                "sha256": h.hexdigest(),
                "packagePath": package_path,
            })
    return descriptors

if len(sys.argv) != 3:
    raise SystemExit("usage: prepare-stage13a-streaming-asr.py REVISION OUT_DIR")

revision = int(sys.argv[1])
if revision < 1:
    raise SystemExit("revision must be >= 1")

out = pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

base_capabilities = {
    "supportsStreaming": True,
    "supportsPartial": True,
    "supportsTokenTiming": True,
    "supportsLanguageDetection": False,
    "supportsConfidence": False,
    "supportsInverseTextNormalization": False,
    "supportsSecondPass": False,
    "supportsHotwords": False,
    "executionMode": "TRUE_STREAMING",
    "timestampCapability": "TOKEN",
    "supportsLanguageForcing": False,
    "punctuationMode": "EXTERNAL",
    "supportedParameters": [
        "numThreads",
        "decodingMethod",
        "maxActivePaths",
    ],
}

descriptors = []
with tempfile.TemporaryDirectory() as temp_text:
    temp = pathlib.Path(temp_text)
    for cfg in MODELS:
        archive = temp / (cfg["key"] + ".tar.bz2")
        download(cfg["url"], archive)

        actual_size = archive.stat().st_size
        if actual_size != cfg["expectedSize"]:
            raise SystemExit(
                cfg["modelId"] + " upstream package size mismatch: "
                + str(actual_size)
            )

        package_sha = sha256(archive)
        if package_sha != cfg["expectedSha256"]:
            raise SystemExit(
                cfg["modelId"] + " upstream package SHA-256 mismatch: "
                + package_sha
            )

        files = inspect_files(cfg, archive)
        descriptor = {
            "modelId": cfg["modelId"],
            "kind": "ASR_STREAMING",
            "displayName": cfg["displayName"],
            "version": cfg["version"],
            "revision": revision,
            "runtimeId": "sherpa-onnx",
            "runtimeVersionMin": "1.13.8",
            "runtimeVersionMax": None,
            "runtimeModelType": cfg["runtimeModelType"],
            "quantization": "INT8",
            "languages": ["zh"],
            "capabilities": dict(base_capabilities),
            "sourceType": "MANAGED_DOWNLOAD",
            "builtinAssetPath": None,
            "download": {
                "packageFormat": "TAR_BZ2",
                "url": cfg["url"],
                "mirrors": [],
                "sizeBytes": actual_size,
                "sha256": package_sha,
            },
            "installedSizeBytes": sum(item["sizeBytes"] for item in files),
            "files": files,
            "compatibility": {
                "abis": ["arm64-v8a"],
                "minSdk": 26,
                "appVersionMin": 41,
                "appVersionMax": None,
            },
            "license": {
                "id": "Apache-2.0",
                "url": "https://github.com/k2-fsa/sherpa-onnx/blob/master/LICENSE",
                "attribution": cfg["attribution"],
                "redistributionPolicy": "UPSTREAM_ONLY",
            },
            "sourceUrl": cfg["url"],
            "homepage": "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html",
            "releaseChannel": "production",
            "autoUpdateEligible": False,
            "deprecated": False,
            "criticalUpdate": False,
        }
        descriptors.append(descriptor)
        (out / (cfg["key"] + "-model-descriptor.json")).write_text(
            json.dumps(descriptor, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

(out / "streaming-asr-model-descriptors.json").write_text(
    json.dumps(descriptors, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(descriptors, ensure_ascii=False, indent=2))
